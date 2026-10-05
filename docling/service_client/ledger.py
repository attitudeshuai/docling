# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""Local job ledger for resuming docling-serve tasks across process restarts.

The ledger is an opt-in, append-shaped JSON-lines file: one record per line,
the whole file rewritten atomically (temp file + ``os.replace``) under an
OS-level file lock so that several processes sharing one ledger still land a
single record per submission.

Design properties
------------------
- Intent is registered *before* the network submission and completed with the
  server-assigned task id as soon as the submission returns.
- A fresh ``intended`` record is considered owned by another process: peers
  wait for the task id instead of submitting again. An ``intended`` record
  older than ``takeover_timeout`` is treated as a dead owner and may be taken
  over.
- Lines that cannot be decoded, parsed or validated are skipped individually;
  the next rewrite repairs the file by dropping them.
- Terminal and orphaned records older than ``record_ttl`` can be purged.
- Nothing credential-bearing is persisted: callers sanitize every model before
  it reaches a record, and records themselves carry only reattachment hints
  (names, kinds, sizes), never request headers, API keys or raw settings.
"""

from __future__ import annotations

import json
import logging
import os
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

STATE_INTENDED = "intended"
STATE_SUBMITTED = "submitted"
STATE_SUCCEEDED = "succeeded"
STATE_FAILED = "failed"
STATE_ORPHANED = "orphaned"

TERMINAL_STATES: frozenset[str] = frozenset(
    {STATE_SUCCEEDED, STATE_FAILED, STATE_ORPHANED}
)
_TERMINAL_STATUS_TO_STATE: dict[str, str] = {
    "success": STATE_SUCCEEDED,
    "failure": STATE_FAILED,
}

DEFAULT_TAKEOVER_TIMEOUT_SECONDS = 30.0
DEFAULT_RECORD_TTL = timedelta(days=7)
DEFAULT_PEER_POLL_INTERVAL_SECONDS = 0.2


@dataclass(frozen=True, slots=True)
class JobLedgerConfig:
    """Configuration for the local job ledger.

    Attributes:
        path: Ledger file location; parent directories are created as needed.
        takeover_timeout: How long a fresh ``intended`` record is assumed to be
            a live submission by another process before it may be taken over.
        record_ttl: Age after which terminal and orphaned records are purged.
        auto_cleanup: Run a purge at most once per client, on first use.
        peer_poll_interval: Cadence for waiting on another process to finish
            its in-flight submission.
    """

    path: Path
    takeover_timeout: float = DEFAULT_TAKEOVER_TIMEOUT_SECONDS
    record_ttl: timedelta = DEFAULT_RECORD_TTL
    auto_cleanup: bool = True
    peer_poll_interval: float = DEFAULT_PEER_POLL_INTERVAL_SECONDS

    def __post_init__(self) -> None:
        if self.takeover_timeout <= 0:
            raise ValueError("takeover_timeout must be positive.")
        if self.record_ttl <= timedelta(0):
            raise ValueError("record_ttl must be positive.")
        if self.peer_poll_interval <= 0:
            raise ValueError("peer_poll_interval must be positive.")
        if not isinstance(self.path, Path):
            object.__setattr__(self, "path", Path(self.path))


@dataclass(slots=True)
class _LedgerRecord:
    fingerprint: str
    state: str
    token: str
    intent_at: datetime
    updated_at: datetime
    task_id: str | None = None
    task_status: str | None = None
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class _Owner:
    """This caller owns the intent and must perform the submission."""

    token: str


@dataclass(frozen=True, slots=True)
class _PeerWait:
    """Another process owns a fresh intent; wait for its task id."""


@dataclass(frozen=True, slots=True)
class _Existing:
    """A completed record exists; the caller must reattach to its task id."""

    record: _LedgerRecord


@dataclass(frozen=True, slots=True)
class _TargetMarker:
    """Kind-only stand-in for a target whose credentials are not stored.

    Used when reattaching after a restart: post-submission result fetching only
    needs the target kind, never the original credentials.
    """

    kind: str


class _FileLock:
    """Cross-process single-writer lock backed by a one-byte file lock."""

    def __init__(self, lock_path: Path) -> None:
        self._lock_path = lock_path
        self._fd: int | None = None

    def __enter__(self) -> _FileLock:
        self._lock_path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(str(self._lock_path), os.O_RDWR | os.O_CREAT)
        try:
            if os.name == "nt":
                import msvcrt

                os.lseek(fd, 0, os.SEEK_SET)
                # LK_LOCK blocks (retries for ~10s) instead of failing immediately.
                msvcrt.locking(fd, msvcrt.LK_LOCK, 1)
            else:
                import fcntl

                fcntl.flock(fd, fcntl.LOCK_EX)
        except BaseException:
            os.close(fd)
            raise
        self._fd = fd
        return self

    def __exit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None:
        fd = self._fd
        if fd is None:
            return
        try:
            if os.name == "nt":
                import msvcrt

                os.lseek(fd, 0, os.SEEK_SET)
                msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)
            self._fd = None


class _JobLedger:
    """Persistent, multi-process-safe storage for job records."""

    def __init__(self, config: JobLedgerConfig) -> None:
        self._config = config
        self._path = config.path
        self._lock = _FileLock(self._path.with_name(f".{self._path.name}.lock"))
        self._auto_purged = False

    @property
    def peer_poll_interval(self) -> float:
        return self._config.peer_poll_interval

    def auto_purge_once(self) -> None:
        """Purge expired records once per ledger instance, if enabled."""
        if self._auto_purged:
            return
        self._auto_purged = True
        if self._config.auto_cleanup:
            self.purge_expired()

    def begin(self, fingerprint: str) -> _Owner | _PeerWait | _Existing:
        """Register or look up the intent for ``fingerprint``.

        Returns:
            _Owner: the caller must submit and then call :meth:`complete`.
            _PeerWait: a fresh intent exists; re-check after sleeping.
            _Existing: a record with a task id exists; reattach.
        """
        with self._lock:
            records = self._read_locked()
            now = datetime.now(tz=timezone.utc)
            record = _find_record(records, fingerprint)
            if record is None:
                record = _LedgerRecord(
                    fingerprint=fingerprint,
                    state=STATE_INTENDED,
                    token=uuid.uuid4().hex,
                    intent_at=now,
                    updated_at=now,
                )
                records.append(record)
                self._write_locked(records)
                return _Owner(record.token)

            if record.state == STATE_INTENDED:
                age = (now - record.intent_at).total_seconds()
                if age < self._config.takeover_timeout:
                    return _PeerWait()
                # The owning process died mid-submission; take over.
                record.token = uuid.uuid4().hex
                record.intent_at = now
                record.updated_at = now
                record.task_id = None
                record.task_status = None
                record.details = {}
                self._write_locked(records)
                return _Owner(record.token)

            return _Existing(record)

    def complete(
        self,
        fingerprint: str,
        token: str,
        task_id: str,
        task_status: str,
        details: dict[str, Any],
    ) -> _LedgerRecord | None:
        """Fill the task id after a successful submission.

        Returns the record when ``token`` still owns it, otherwise ``None`` —
        the record was taken over by another process and the caller must
        discard its own submission and reattach.
        """
        with self._lock:
            records = self._read_locked()
            record = _find_record(records, fingerprint)
            if record is None or record.token != token:
                return None
            now = datetime.now(tz=timezone.utc)
            record.task_id = task_id
            record.task_status = task_status
            record.details = dict(details)
            record.state = _TERMINAL_STATUS_TO_STATE.get(task_status, STATE_SUBMITTED)
            record.updated_at = now
            self._write_locked(records)
            return record

    def abandon(self, fingerprint: str, token: str) -> None:
        """Remove an intent whose submission failed, if we still own it."""
        with self._lock:
            records = self._read_locked()
            record = _find_record(records, fingerprint)
            if (
                record is not None
                and record.token == token
                and record.state == STATE_INTENDED
            ):
                records.remove(record)
                self._write_locked(records)

    def note_status(
        self,
        fingerprint: str,
        task_id: str,
        task_status: str,
    ) -> None:
        """Persist the latest observed task status, including terminal ones."""
        with self._lock:
            records = self._read_locked()
            record = _find_record(records, fingerprint)
            if record is None or record.task_id != task_id:
                return
            now = datetime.now(tz=timezone.utc)
            record.task_status = task_status
            new_state = _TERMINAL_STATUS_TO_STATE.get(task_status)
            if new_state is not None and record.state not in TERMINAL_STATES:
                record.state = new_state
            record.updated_at = now
            self._write_locked(records)

    def mark_orphaned(self, fingerprint: str, task_id: str) -> None:
        """Mark a task the service no longer recognizes."""
        with self._lock:
            records = self._read_locked()
            record = _find_record(records, fingerprint)
            if record is None or record.task_id != task_id:
                return
            if record.state != STATE_ORPHANED:
                record.state = STATE_ORPHANED
                record.updated_at = datetime.now(tz=timezone.utc)
                self._write_locked(records)

    def get_record(self, fingerprint: str) -> _LedgerRecord | None:
        with self._lock:
            return _find_record(self._read_locked(), fingerprint)

    def purge_expired(self) -> int:
        """Drop terminal/orphaned records older than the TTL.

        Ancient ``intended`` records (dead owners never taken over) are removed
        too. Returns the number of records removed.
        """
        with self._lock:
            records = self._read_locked()
            now = datetime.now(tz=timezone.utc)
            cutoff = now - self._config.record_ttl
            kept: list[_LedgerRecord] = []
            for record in records:
                if record.state == STATE_INTENDED:
                    expired = record.intent_at < cutoff
                else:
                    expired = record.updated_at < cutoff
                if expired:
                    logger.info(
                        "Purging expired ledger record fp=%s state=%s task_id=%s",
                        record.fingerprint,
                        record.state,
                        record.task_id,
                    )
                    continue
                kept.append(record)
            removed = len(records) - len(kept)
            if removed:
                self._write_locked(kept)
            return removed

    # ------------------------------------------------------------------
    # Tolerant file I/O (callers already hold the lock)
    # ------------------------------------------------------------------

    def _read_locked(self) -> list[_LedgerRecord]:
        if not self._path.exists():
            return []
        try:
            raw = self._path.read_bytes()
        except OSError as exc:
            logger.warning("Cannot read job ledger %s: %s", self._path, exc)
            return []
        records: list[_LedgerRecord] = []
        skipped = 0
        for line_number, line in enumerate(raw.splitlines(), start=1):
            if not line.strip():
                continue
            try:
                text = line.decode("utf-8")
                obj = json.loads(text)
                records.append(_record_from_object(obj))
            except (
                UnicodeDecodeError,
                json.JSONDecodeError,
                TypeError,
                ValueError,
            ) as exc:
                # Bad or half-written record: skip it and keep processing the rest.
                skipped += 1
                logger.warning(
                    "Skipping invalid job ledger record at %s line %d: %s",
                    self._path,
                    line_number,
                    exc,
                )
        if skipped:
            # Caller already holds the lock; repair by rewriting without the bad lines.
            self._write_locked(records)
        return records

    def _write_locked(self, records: list[_LedgerRecord]) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = self._path.with_name(
            f".{self._path.name}.{os.getpid()}.{uuid.uuid4().hex[:8]}.tmp"
        )
        try:
            with tmp_path.open("w", encoding="utf-8", newline="\n") as handle:
                for record in records:
                    handle.write(
                        json.dumps(
                            _record_to_object(record),
                            ensure_ascii=False,
                            separators=(",", ":"),
                        )
                    )
                    handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp_path, self._path)
        except BaseException:
            try:
                tmp_path.unlink(missing_ok=True)
            except OSError:
                pass
            raise


def _find_record(
    records: list[_LedgerRecord], fingerprint: str
) -> _LedgerRecord | None:
    for record in records:
        if record.fingerprint == fingerprint:
            return record
    return None


def _record_from_object(obj: Any) -> _LedgerRecord:
    if not isinstance(obj, dict):
        raise TypeError("record must be a JSON object")
    required = ("fp", "st", "tok", "iat", "uat")
    for key in required:
        if key not in obj:
            raise ValueError(f"missing field {key!r}")
    details = obj.get("d")
    if details is not None and not isinstance(details, dict):
        raise ValueError("'d' must be an object")
    return _LedgerRecord(
        fingerprint=str(obj["fp"]),
        state=str(obj["st"]),
        token=str(obj["tok"]),
        intent_at=_parse_timestamp(obj["iat"]),
        updated_at=_parse_timestamp(obj["uat"]),
        task_id=obj.get("tid"),
        task_status=obj.get("stat"),
        details=dict(details or {}),
    )


def _record_to_object(record: _LedgerRecord) -> dict[str, Any]:
    return {
        "fp": record.fingerprint,
        "st": record.state,
        "tok": record.token,
        "iat": _format_timestamp(record.intent_at),
        "uat": _format_timestamp(record.updated_at),
        "tid": record.task_id,
        "stat": record.task_status,
        "d": record.details,
    }


def _parse_timestamp(value: Any) -> datetime:
    parsed = datetime.fromisoformat(str(value))
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed


def _format_timestamp(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat()
