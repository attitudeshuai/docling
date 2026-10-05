# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""Lifecycle-governed workspace for conversion intermediate artifacts.

When the workspace is enabled (see
``docling.datamodel.settings.WorkspaceSettings``), every converted document
receives a private subdirectory under a configured root. All temporary
directories and temporary files created by the backends and the audio/video
pipelines are allocated inside that subdirectory through the helpers in this
module.

Guarantees:

* Per-document directories are uniquely named and created with private
  permissions; concurrent conversions only ever see their own directory, and
  releasing one document cannot affect another.
* The directory is released on success, failure, and interruption alike.
* Directories left behind by a killed process are reclaimed on the next
  workspace startup, based on an owner-PID marker.
* Admission is refused (and either degraded to ordinary system temp files or
  turned into a policy error, depending on configuration) when the capacity
  limit or the free-space floor would be violated.
* Files produced from in-memory data are written via a staging name and
  atomically renamed, and outputs of external tools follow an explicit
  stage/commit protocol, so half-written files are never consumed as valid
  intermediate artifacts.

When the workspace is disabled (the default), every helper is an exact
pass-through to :mod:`tempfile`: no workspace directory is ever created and
the observable behavior is identical to not using this module.
"""

import json
import logging
import os
import re
import shutil
import tempfile
import threading
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import (
    IO,
    TYPE_CHECKING,
    Annotated,
    Any,
    Iterator,
    Optional,
    Protocol,
    Union,
)

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from docling.backend.abstract_backend import AbstractDocumentBackend
    from docling.datamodel.document import InputDocument

_log = logging.getLogger(__name__)

MARKER_FILENAME = "workspace.json"
ROOT_MARKER_FILENAME = ".docling-workspace"
SWEEP_LOCK_DIRNAME = ".sweep.lock"
SWEEP_LOCK_OWNER_FILENAME = "owner.json"
DOC_DIR_PREFIX = "doc-"
SCRATCH_DIRNAME = "scratch"
PART_SUFFIX = ".part"
_ROOT_MARKER_PAYLOAD = {"version": 1}


class WorkspaceLimitPolicy(str, Enum):
    """What to do when a workspace request violates the configured limits."""

    FALLBACK = "fallback"
    ERROR = "error"


class WorkspaceSettings(BaseModel):
    """Configuration for the intermediate-artifacts workspace.

    The workspace is disabled by default: no directories are created and
    conversion behaves exactly as without a workspace.

    When enabled, each converted document gets a private subdirectory under
    ``root`` that hosts every temporary directory and temporary file created
    by the backends and the audio/video pipelines. The subdirectory is
    removed when the conversion ends (success, failure, or interruption);
    directories left behind by abnormally terminated processes are reclaimed
    the next time a workspace starts.
    """

    enabled: Annotated[
        bool,
        Field(
            description=(
                "Turn on the intermediate-artifacts workspace. Disabled by "
                "default; when disabled no directory is ever created."
            )
        ),
    ] = False
    root: Annotated[
        Optional[Path],
        Field(
            description=(
                "Directory hosting the per-document workspaces. Created "
                "lazily on first use. When unset, "
                "`<system-temp>/docling-workspace` is used."
            )
        ),
    ] = None
    max_size_bytes: Annotated[
        Optional[int],
        Field(
            description=(
                "Capacity upper bound for the workspace root, in bytes. When the "
                "current usage plus the estimated size of a new document "
                "would exceed it, ``limit_policy`` decides what happens."
            ),
            ge=0,
        ),
    ] = None
    min_free_disk_bytes: Annotated[
        Optional[int],
        Field(
            description=(
                "Minimum free space, in bytes, that must remain on the "
                "filesystem hosting the workspace root while a new document "
                "is admitted. When unset, no free-space floor is enforced."
            ),
            ge=0,
        ),
    ] = None
    limit_policy: Annotated[
        WorkspaceLimitPolicy,
        Field(
            description=(
                "FALLBACK (default): log a warning and process the document "
                "with ordinary system temp files. ERROR: reject the document "
                "with a policy failure instead of admitting it."
            )
        ),
    ] = WorkspaceLimitPolicy.FALLBACK
    stale_dir_ttl_seconds: Annotated[
        float,
        Field(
            description=(
                "Grace period in seconds before an unmanaged document "
                "directory can be reclaimed. Protects directories that a "
                "concurrent process is currently initializing."
            ),
            ge=0.0,
        ),
    ] = 60.0


class WorkspaceError(RuntimeError):
    """Base class for workspace failures."""


class WorkspaceCapacityError(WorkspaceError):
    """Raised when a document cannot be admitted under the configured limits."""


@dataclass(frozen=True)
class _WorkspaceContext:
    """Owner metadata persisted in a document directory marker."""

    pid: int
    uuid: str
    created_at: str
    doc_name: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "pid": self.pid,
            "uuid": self.uuid,
            "created_at": self.created_at,
            "doc_name": self.doc_name,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "_WorkspaceContext":
        return cls(
            pid=int(payload["pid"]),
            uuid=str(payload["uuid"]),
            created_at=str(payload["created_at"]),
            doc_name=str(payload.get("doc_name", "")),
        )


@dataclass(frozen=True)
class StagedOutput:
    """An output path that an external tool writes before it is valid.

    The tool writes to ``part``; only after the caller verifies the result
    (return code, existence, non-zero size) does it call :meth:`commit`,
    which atomically publishes the file at ``final``. A file left at
    ``part`` can therefore never be mistaken for a valid intermediate
    artifact.
    """

    part: Path
    final: Path

    def commit(self) -> Path:
        """Atomically publish the staged file at its final location."""
        if not self.part.exists():
            raise WorkspaceError(f"Staged output does not exist: {self.part}")
        os.replace(self.part, self.final)
        return self.final

    def discard(self) -> None:
        """Remove the staged file if it is still around."""
        self.part.unlink(missing_ok=True)


def _pid_is_alive(pid: int) -> bool:
    """Return whether a process with ``pid`` currently exists."""
    if pid <= 0:
        return False
    if os.name == "nt":
        import ctypes

        # PROCESS_QUERY_LIMITED_INFORMATION; avoids elevated privileges.
        process_query_limited_information = 0x1000
        kernel32 = ctypes.windll.kernel32
        handle = kernel32.OpenProcess(process_query_limited_information, False, pid)
        if not handle:
            return False
        kernel32.CloseHandle(handle)
        return True

    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        # The process exists but is owned by another user.
        return True
    except OSError:
        return True
    return True


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_name(name: str) -> str:
    """Reduce an arbitrary document name to a single safe path component."""
    sanitized = re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("._-")
    return sanitized[:48] or "document"


def _directory_size(path: Path) -> int:
    """Sum the logical size of all files below ``path`` (best effort)."""
    total = 0
    for root, _dirs, files in os.walk(path):
        for filename in files:
            try:
                total += (Path(root) / filename).stat().st_size
            except OSError:
                # Files may disappear concurrently; skip them.
                continue
    return total


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    """Write ``data`` to ``path`` via a sibling staging file and rename."""
    staging = path.with_name(f".{path.name}.{uuid.uuid4().hex[:12]}{PART_SUFFIX}")
    try:
        with staging.open("wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(staging, path)
    finally:
        staging.unlink(missing_ok=True)


def _restrict_to_owner(path: Path) -> None:
    """Restrict a directory to the current user on POSIX systems."""
    if os.name == "posix":
        path.chmod(0o700)


class DocumentWorkspace:
    """Private intermediate-artifact directory of one document."""

    def __init__(self, path: Path, context: _WorkspaceContext) -> None:
        self.path = path
        self.scratch = path / SCRATCH_DIRNAME
        self._context = context
        self._released = False

    @property
    def released(self) -> bool:
        """Whether this workspace has already been released."""
        return self._released

    def mkdtemp(self, prefix: Optional[str] = None) -> Path:
        """Create a unique throwaway directory inside this workspace."""
        return Path(tempfile.mkdtemp(prefix=prefix, dir=self.scratch))

    def named_temp_file(
        self,
        mode: str = "w+b",
        suffix: Optional[str] = None,
        prefix: Optional[str] = None,
        delete: bool = True,
    ) -> IO[Any]:
        """Create a named temporary file inside this workspace."""
        return tempfile.NamedTemporaryFile(
            mode=mode,
            suffix=suffix,
            prefix=prefix,
            dir=self.scratch,
            delete=delete,
        )

    def atomic_write_bytes(self, filename: str, data: bytes) -> Path:
        """Write ``data`` into the scratch area using a staging rename.

        Returns the final path. Concurrent readers either see the complete
        file or no file at all.
        """
        final = self.scratch / _safe_name(filename)
        _atomic_write_bytes(final, data)
        return final

    def stage_output(self, suffix: str = "", prefix: str = "out-") -> StagedOutput:
        """Reserve ``final``/``part`` paths for an external tool output."""
        token = uuid.uuid4().hex[:12]
        final = self.scratch / f"{prefix}{token}{suffix}"
        part = final.with_name(final.name + PART_SUFFIX)
        return StagedOutput(part=part, final=final)

    def release(self) -> None:
        """Remove the whole document workspace. Idempotent."""
        if self._released:
            return
        self._released = True
        shutil.rmtree(self.path, ignore_errors=True)
        if self.path.exists():
            _log.warning(
                "Intermediate-artifact workspace %s could not be fully removed.",
                self.path,
            )


class _SweepLock:
    """Best-effort cross-process mutex around a startup sweep.

    Implemented with an ``O_EXCL`` directory so it works without third-party
    dependencies. A lock whose owner process is dead is reclaimed. Failure to
    acquire the lock simply skips the sweep (another process is sweeping).
    """

    def __init__(self, root: Path) -> None:
        self._lock_path = root / SWEEP_LOCK_DIRNAME
        self._acquired = False

    def __enter__(self) -> bool:
        try:
            self._lock_path.mkdir()
            self._acquired = True
        except FileExistsError:
            self._acquired = self._reclaim_stale_lock()
        if self._acquired:
            owner = {"pid": os.getpid(), "created_at": _utc_now_iso()}
            (self._lock_path / SWEEP_LOCK_OWNER_FILENAME).write_text(
                json.dumps(owner), encoding="utf-8"
            )
        return self._acquired

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        if self._acquired:
            shutil.rmtree(self._lock_path, ignore_errors=True)

    def _reclaim_stale_lock(self) -> bool:
        owner_file = self._lock_path / SWEEP_LOCK_OWNER_FILENAME
        try:
            owner = json.loads(owner_file.read_text(encoding="utf-8"))
            if _pid_is_alive(int(owner["pid"])):
                return False
        except (OSError, ValueError, KeyError):
            return False
        shutil.rmtree(self._lock_path, ignore_errors=True)
        try:
            self._lock_path.mkdir()
        except FileExistsError:
            return False
        return True


class WorkspaceManager:
    """Owns the workspace root and hands out per-document workspaces."""

    def __init__(self, options: WorkspaceSettings) -> None:
        self.options = options
        self._lock = threading.RLock()
        self._root: Optional[Path] = None

    @property
    def enabled(self) -> bool:
        return self.options.enabled

    def effective_root(self) -> Path:
        """Resolve the configured root without creating it."""
        if self.options.root is not None:
            return Path(self.options.root).expanduser()
        return Path(tempfile.gettempdir()) / "docling-workspace"

    def acquire(
        self, doc_name: str, estimated_bytes: int = 0
    ) -> Optional[DocumentWorkspace]:
        """Allocate a private workspace for one document.

        Returns ``None`` when the workspace is disabled or when the FALLBACK
        limit policy applies; callers then use ordinary system temp files.
        Raises :class:`WorkspaceCapacityError` when the ERROR limit policy
        applies.
        """
        if not self.options.enabled:
            return None

        with self._lock:
            root = self._ensure_root()
            try:
                self._enforce_limits(root, estimated_bytes=estimated_bytes)
            except WorkspaceCapacityError:
                if self.options.limit_policy is WorkspaceLimitPolicy.ERROR:
                    raise
                _log.warning(
                    "Intermediate-artifacts workspace at %s cannot admit %s "
                    "within the configured limits; falling back to the system "
                    "temporary directory for this document.",
                    root,
                    doc_name,
                )
                return None
            return self._create_workspace(root, doc_name=doc_name)

    def _ensure_root(self) -> Path:
        if self._root is not None:
            return self._root

        root = self.effective_root()
        created = not root.exists()
        root.mkdir(parents=True, exist_ok=True)
        if created:
            _restrict_to_owner(root)
            _atomic_write_bytes(
                root / ROOT_MARKER_FILENAME,
                json.dumps(_ROOT_MARKER_PAYLOAD).encode("utf-8"),
            )

        self._sweep_stale(root)
        self._root = root
        return root

    def _enforce_limits(self, root: Path, estimated_bytes: int) -> None:
        usage = shutil.disk_usage(root)

        if self.options.min_free_disk_bytes is not None:
            projected_free = usage.free - max(estimated_bytes, 0)
            if projected_free < self.options.min_free_disk_bytes:
                raise WorkspaceCapacityError(
                    f"Free space on {root} would drop to "
                    f"{projected_free} bytes, below the configured floor of "
                    f"{self.options.min_free_disk_bytes} bytes."
                )

        if self.options.max_size_bytes is not None:
            current_usage = _directory_size(root)
            projected_usage = current_usage + max(estimated_bytes, 0)
            if projected_usage > self.options.max_size_bytes:
                raise WorkspaceCapacityError(
                    f"Workspace usage under {root} would grow to "
                    f"{projected_usage} bytes, above the configured capacity "
                    f"limit of {self.options.max_size_bytes} bytes "
                    f"(current usage: {current_usage} bytes)."
                )

    def _create_workspace(self, root: Path, doc_name: str) -> DocumentWorkspace:
        context = _WorkspaceContext(
            pid=os.getpid(),
            uuid=uuid.uuid4().hex,
            created_at=_utc_now_iso(),
            doc_name=doc_name,
        )
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        safe_name = _safe_name(doc_name)

        path: Optional[Path] = None
        for _ in range(8):
            candidate = (
                root / f"{DOC_DIR_PREFIX}{timestamp}-{context.uuid[:12]}-{safe_name}"
            )
            try:
                candidate.mkdir()
            except FileExistsError:
                continue
            path = candidate
            break
        if path is None:
            raise WorkspaceError(
                f"Could not allocate a unique workspace directory under {root}."
            )

        _restrict_to_owner(path)
        # Publish the owner marker before anything else lands in the
        # directory: as soon as it exists, a concurrent sweep sees a
        # live owning PID and leaves the directory alone.
        _atomic_write_bytes(
            path / MARKER_FILENAME,
            json.dumps(context.to_dict()).encode("utf-8"),
        )
        (path / SCRATCH_DIRNAME).mkdir(exist_ok=True)
        _restrict_to_owner(path / SCRATCH_DIRNAME)
        return DocumentWorkspace(path=path, context=context)

    def _sweep_stale(self, root: Path) -> None:
        """Remove document directories left by abnormally exited processes."""
        with _SweepLock(root) as acquired:
            if not acquired:
                _log.debug(
                    "Skipping workspace sweep at %s: another process holds "
                    "the sweep lock.",
                    root,
                )
                return

            now = datetime.now(timezone.utc).timestamp()
            grace_period = max(0.0, self.options.stale_dir_ttl_seconds)
            for entry in root.iterdir():
                if not entry.is_dir() or not entry.name.startswith(DOC_DIR_PREFIX):
                    continue
                try:
                    age = now - entry.stat().st_mtime
                except OSError:
                    continue
                if age < grace_period:
                    # Leave freshly created directories alone: a concurrent
                    # process may still be writing their marker.
                    continue

                marker = entry / MARKER_FILENAME
                context = self._read_marker(marker)
                if context is None:
                    # No readable marker: never initialized or crashed
                    # mid-initialization; safe to reclaim past the grace.
                    shutil.rmtree(entry, ignore_errors=True)
                    continue
                if _pid_is_alive(context.pid):
                    # Owned by a live process (possibly on another host is
                    # out of scope; same machine only): leave untouched.
                    continue
                _log.info(
                    "Reclaiming stale intermediate-artifacts workspace %s "
                    "left by process %s.",
                    entry,
                    context.pid,
                )
                shutil.rmtree(entry, ignore_errors=True)

    @staticmethod
    def _read_marker(marker: Path) -> Optional[_WorkspaceContext]:
        try:
            payload = json.loads(marker.read_text(encoding="utf-8"))
            return _WorkspaceContext.from_dict(payload)
        except (OSError, ValueError, KeyError, TypeError):
            return None


# A document is processed by one worker thread at a time, so the active
# workspace is held in thread-local state and reached by code paths that do
# not receive the document explicitly (e.g. LibreOffice helper functions).
_thread_state = threading.local()


def _workspace_stack() -> list[DocumentWorkspace]:
    # threading.local has no documented default-value API; initialize the
    # per-thread stack lazily on first access.
    stack = getattr(_thread_state, "stack", None)
    if stack is None:
        stack = []
        _thread_state.stack = stack
    return stack


@contextmanager
def workspace_context(
    workspace: Optional[DocumentWorkspace],
) -> Iterator[None]:
    """Make ``workspace`` the active workspace for the current thread."""
    if workspace is None:
        yield
        return
    stack = _workspace_stack()
    stack.append(workspace)
    try:
        yield
    finally:
        stack.pop()


class WorkspaceOwner(Protocol):
    """Object carrying the document workspace (backend or input document)."""

    @property
    def workspace(self) -> Optional[DocumentWorkspace]: ...


# Anything a temp-allocation helper can resolve a workspace from: the
# workspace itself, or an object carrying one.
WorkspaceOwnerArg = Union[
    DocumentWorkspace,
    "InputDocument",
    "AbstractDocumentBackend",
]


def resolve_workspace(
    owner: Optional[WorkspaceOwnerArg] = None,
) -> Optional[DocumentWorkspace]:
    """Return the workspace tied to ``owner`` or active on this thread."""
    if owner is not None:
        if isinstance(owner, DocumentWorkspace):
            return owner
        return owner.workspace
    stack = _workspace_stack()
    return stack[-1] if stack else None


def workspace_mkdtemp(
    prefix: Optional[str] = None,
    *,
    owner: Optional[WorkspaceOwnerArg] = None,
) -> Path:
    """Drop-in replacement for :func:`tempfile.mkdtemp`.

    Allocates inside the document workspace when one is available, otherwise
    under the system temporary directory (the pre-workspace behavior).
    """
    workspace = resolve_workspace(owner)
    if workspace is not None:
        return workspace.mkdtemp(prefix=prefix)
    return Path(tempfile.mkdtemp(prefix=prefix))


def workspace_named_temp_file(
    mode: str = "w+b",
    suffix: Optional[str] = None,
    prefix: Optional[str] = None,
    delete: bool = True,
    *,
    owner: Optional[WorkspaceOwnerArg] = None,
) -> IO[Any]:
    """Drop-in replacement for :class:`tempfile.NamedTemporaryFile`."""
    workspace = resolve_workspace(owner)
    if workspace is not None:
        return workspace.named_temp_file(
            mode=mode, suffix=suffix, prefix=prefix, delete=delete
        )
    return tempfile.NamedTemporaryFile(
        mode=mode, suffix=suffix, prefix=prefix, delete=delete
    )


@contextmanager
def workspace_temp_directory(
    prefix: Optional[str] = None,
    *,
    owner: Optional[WorkspaceOwnerArg] = None,
) -> Iterator[Path]:
    """Context manager yielding a throwaway directory.

    Mirrors :class:`tempfile.TemporaryDirectory`: the directory is removed on
    exit. Inside a document workspace it is also removed when the whole
    document workspace is released, so a crash mid-block cannot leak it.
    """
    workspace = resolve_workspace(owner)
    if workspace is None:
        with tempfile.TemporaryDirectory(prefix=prefix) as directory:
            yield Path(directory)
        return

    directory = workspace.mkdtemp(prefix=prefix)
    try:
        yield directory
    finally:
        shutil.rmtree(directory, ignore_errors=True)


def workspace_materialize(
    data: bytes,
    suffix: str = "",
    *,
    owner: Optional[WorkspaceOwnerArg] = None,
) -> Path:
    """Write in-memory bytes to a path a subprocess can open.

    Uses an atomic staging rename inside a document workspace; outside one,
    uses :func:`tempfile.mkstemp`, which atomically creates a unique
    user-private file. The caller is responsible for removing the path once
    consumed (the document workspace release covers it on crashes).
    """
    workspace = resolve_workspace(owner)
    if workspace is not None:
        token = uuid.uuid4().hex[:12]
        return workspace.atomic_write_bytes(f"input-{token}{suffix}", data)

    descriptor, name = tempfile.mkstemp(suffix=suffix)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
    except BaseException:
        Path(name).unlink(missing_ok=True)
        raise
    return Path(name)
