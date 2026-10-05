# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

import json
import os
import shutil
import stat
import subprocess
import sys
import threading
import time
from io import BytesIO
from pathlib import Path

import pytest

from docling.datamodel.base_models import ConversionStatus, DocumentStream, InputFormat
from docling.document_converter import DocumentConverter
from docling.utils.workspace import (
    DOC_DIR_PREFIX,
    MARKER_FILENAME,
    SCRATCH_DIRNAME,
    WorkspaceCapacityError,
    WorkspaceError,
    WorkspaceLimitPolicy,
    WorkspaceManager,
    WorkspaceSettings,
    _pid_is_alive,
    resolve_workspace,
    workspace_context,
    workspace_materialize,
    workspace_mkdtemp,
    workspace_named_temp_file,
    workspace_temp_directory,
)

pytestmark = pytest.mark.cross_platform


def _settings(tmp_path: Path, **kwargs) -> WorkspaceSettings:
    kwargs.setdefault("stale_dir_ttl_seconds", 0)
    return WorkspaceSettings(enabled=True, root=tmp_path / "workspace", **kwargs)


def _doc_dirs(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return [
        p for p in root.iterdir() if p.is_dir() and p.name.startswith(DOC_DIR_PREFIX)
    ]


def _dead_pid() -> int:
    """A process ID that has provably exited and been fully reaped."""
    process = subprocess.Popen([sys.executable, "-c", "pass"])
    pid = process.pid
    process.wait()
    # Releasing our last reference closes the process handle, letting the
    # OS recycle the PID; on Windows OpenProcess() otherwise still
    # succeeds while the Popen object is alive.
    del process
    # Give the OS a moment to finish tearing down the process object.
    for _ in range(50):
        if not _pid_is_alive(pid):
            break
        time.sleep(0.05)
    assert not _pid_is_alive(pid)
    return pid


def _write_leftover(root: Path, name: str, pid: int | None) -> Path:
    directory = root / name
    (directory / SCRATCH_DIRNAME).mkdir(parents=True)
    (directory / SCRATCH_DIRNAME / "leftover.bin").write_bytes(b"stale")
    if pid is not None:
        marker = {
            "pid": pid,
            "uuid": "leftover-uuid",
            "created_at": "2000-01-01T00:00:00+00:00",
            "doc_name": "leftover",
        }
        (directory / MARKER_FILENAME).write_text(json.dumps(marker), encoding="utf-8")
    return directory


def test_disabled_workspace_creates_nothing(tmp_path: Path) -> None:
    root = tmp_path / "workspace"
    manager = WorkspaceManager(
        WorkspaceSettings(enabled=False, root=root, max_size_bytes=1)
    )

    assert manager.acquire("doc.csv", estimated_bytes=10**9) is None
    assert not root.exists()

    # Helpers are exact pass-throughs to the system temp area.
    directory = workspace_mkdtemp(prefix="ws_disabled_")
    try:
        assert directory.exists()
        assert not str(directory).startswith(str(root))
    finally:
        shutil.rmtree(directory, ignore_errors=True)

    materialized = workspace_materialize(b"abc", suffix=".bin")
    try:
        assert materialized.exists()
        assert not str(materialized).startswith(str(root))
    finally:
        materialized.unlink(missing_ok=True)


def test_workspace_lifecycle_and_marker(tmp_path: Path) -> None:
    manager = WorkspaceManager(_settings(tmp_path))

    workspace = manager.acquire("my report.csv", estimated_bytes=42)

    assert workspace.path.parent == manager.effective_root()
    assert "my_report.csv" in workspace.path.name
    assert workspace.path.name.startswith(DOC_DIR_PREFIX)
    assert workspace.scratch.is_dir()

    marker_path = workspace.path / MARKER_FILENAME
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    assert marker["pid"] == os.getpid()
    assert marker["doc_name"] == "my report.csv"

    if os.name == "posix":
        mode = stat.S_IMODE(workspace.path.stat().st_mode)
        assert mode == 0o700

    nested = workspace.mkdtemp(prefix="nested_")
    assert nested.parent == workspace.scratch

    handle = workspace.named_temp_file(suffix=".png")
    handle.close()
    assert Path(handle.name).parent == workspace.scratch

    output = workspace.atomic_write_bytes("result.bin", b"payload")
    assert output.read_bytes() == b"payload"
    assert list(workspace.scratch.glob(f"*{'.part'}")) == []

    workspace.release()
    assert workspace.released
    assert not workspace.path.exists()

    # Releasing again is a no-op.
    workspace.release()


def test_document_dirs_are_isolated_and_independent(tmp_path: Path) -> None:
    manager = WorkspaceManager(_settings(tmp_path))
    first = manager.acquire("first.csv")
    second = manager.acquire("second.csv")

    assert first.path != second.path
    (first.scratch / "a.tmp").write_bytes(b"a")
    (second.scratch / "b.tmp").write_bytes(b"b")

    first.release()
    assert not first.path.exists()
    assert second.path.exists()
    assert (second.scratch / "b.tmp").read_bytes() == b"b"

    second.release()
    assert _doc_dirs(manager.effective_root()) == []


def test_thread_local_workspace_is_per_thread(tmp_path: Path) -> None:
    manager = WorkspaceManager(_settings(tmp_path))
    first = manager.acquire("thread-a.csv")
    second = manager.acquire("thread-b.csv")
    seen: dict[str, object] = {}

    def run(workspace, label: str) -> None:
        with workspace_context(workspace):
            event = threading.Event()

            def nested() -> None:
                # A new thread without its own context sees no workspace.
                assert resolve_workspace() is None
                event.set()

            helper = threading.Thread(target=nested)
            helper.start()
            helper.join()
            event.wait(1)
            seen[label] = resolve_workspace()

    t1 = threading.Thread(target=run, args=(first, "a"))
    t2 = threading.Thread(target=run, args=(second, "b"))
    t1.start()
    t2.start()
    t1.join()
    t2.join()

    assert seen["a"] is first
    assert seen["b"] is second
    assert resolve_workspace() is None
    first.release()
    second.release()


def test_capacity_limit_fallback_and_error(tmp_path: Path) -> None:
    root = tmp_path / "workspace"
    fallback_manager = WorkspaceManager(
        WorkspaceSettings(
            enabled=True,
            root=root,
            max_size_bytes=200,
            limit_policy=WorkspaceLimitPolicy.FALLBACK,
            stale_dir_ttl_seconds=0,
        )
    )
    admitted = fallback_manager.acquire("first.bin", estimated_bytes=50)
    assert admitted is not None
    (admitted.scratch / "blob").write_bytes(b"x" * 500)

    # Current usage alone already exceeds the cap: degrade with a warning.
    assert fallback_manager.acquire("second.bin", estimated_bytes=0) is None
    assert admitted.path.exists()

    strict_manager = WorkspaceManager(
        WorkspaceSettings(
            enabled=True,
            root=root,
            max_size_bytes=200,
            limit_policy=WorkspaceLimitPolicy.ERROR,
            stale_dir_ttl_seconds=0,
        )
    )
    with pytest.raises(WorkspaceCapacityError, match="capacity"):
        strict_manager.acquire("third.bin", estimated_bytes=0)

    admitted.release()


def test_free_space_floor_enforced(tmp_path: Path) -> None:
    manager = WorkspaceManager(
        WorkspaceSettings(
            enabled=True,
            root=tmp_path / "workspace",
            min_free_disk_bytes=10**15,
            limit_policy=WorkspaceLimitPolicy.ERROR,
            stale_dir_ttl_seconds=0,
        )
    )
    with pytest.raises(WorkspaceCapacityError, match=r"[Ff]ree space"):
        manager.acquire("huge.bin", estimated_bytes=0)

    # FALLBACK policy degrades instead of raising.
    fallback = WorkspaceManager(
        WorkspaceSettings(
            enabled=True,
            root=tmp_path / "workspace-fallback",
            min_free_disk_bytes=10**15,
            limit_policy=WorkspaceLimitPolicy.FALLBACK,
            stale_dir_ttl_seconds=0,
        )
    )
    assert fallback.acquire("huge.bin", estimated_bytes=0) is None


def test_stale_directories_are_reclaimed_on_startup(tmp_path: Path) -> None:
    root = tmp_path / "workspace"
    root.mkdir()

    # A directory still owned by this (live) process must be preserved.
    live = _write_leftover(root, f"{DOC_DIR_PREFIX}live", pid=os.getpid())
    # A directory owned by a process that has exited is reclaimed.
    dead = _write_leftover(root, f"{DOC_DIR_PREFIX}dead", pid=_dead_pid())
    # A markerless directory past the grace period is treated as a crashed
    # initialization and reclaimed.
    markerless = _write_leftover(root, f"{DOC_DIR_PREFIX}nomarker", pid=None)

    manager = WorkspaceManager(
        WorkspaceSettings(enabled=True, root=root, stale_dir_ttl_seconds=0)
    )
    workspace = manager.acquire("fresh.csv")

    assert live.exists()
    assert not dead.exists()
    assert not markerless.exists()
    assert workspace.path.exists()

    shutil.rmtree(live, ignore_errors=True)
    workspace.release()


def test_fresh_markerless_directory_respects_grace_period(tmp_path: Path) -> None:
    root = tmp_path / "workspace"
    root.mkdir()
    in_flight = _write_leftover(root, f"{DOC_DIR_PREFIX}in-flight", pid=None)

    manager = WorkspaceManager(
        WorkspaceSettings(enabled=True, root=root, stale_dir_ttl_seconds=3600)
    )
    manager.acquire("fresh.csv")

    # Too young to be considered a leftover: another process may be writing
    # its marker right now.
    assert in_flight.exists()
    shutil.rmtree(in_flight, ignore_errors=True)


def test_staged_output_blocks_half_written_files(tmp_path: Path) -> None:
    manager = WorkspaceManager(_settings(tmp_path))
    workspace = manager.acquire("video.mp4")

    staged = workspace.stage_output(suffix=".wav", prefix="audio-")
    assert not staged.final.exists()

    # Simulate an external tool that only wrote part of its output.
    staged.part.write_bytes(b"RIFFpartial")
    assert not staged.final.exists()

    # A complete output is published atomically.
    staged.part.write_bytes(b"RIFFcomplete")
    final = staged.commit()
    assert final == staged.final
    assert final.read_bytes() == b"RIFFcomplete"
    assert not staged.part.exists()
    staged.discard()  # no-op after commit

    # Committing a missing staging file is an explicit failure.
    missing = workspace.stage_output(suffix=".wav")
    with pytest.raises(WorkspaceError):
        missing.commit()
    missing.discard()

    workspace.release()


def test_workspace_temp_directory_helper(tmp_path: Path) -> None:
    manager = WorkspaceManager(_settings(tmp_path))
    workspace = manager.acquire("dirs.csv")

    with workspace_temp_directory(owner=workspace) as directory:
        assert directory.parent == workspace.scratch
        assert directory.is_dir()
    assert not directory.exists()

    with workspace_context(workspace):
        with workspace_temp_directory(prefix="implicit_") as implicit:
            assert implicit.parent == workspace.scratch

    workspace.release()

    # Without a workspace the helper mirrors tempfile.TemporaryDirectory.
    with workspace_temp_directory(prefix="ws_outside_") as directory:
        assert directory.is_dir()
        assert tmp_path not in directory.parents
    assert not directory.exists()


def test_materialize_uses_workspace_when_active(tmp_path: Path) -> None:
    manager = WorkspaceManager(_settings(tmp_path))
    workspace = manager.acquire("stream.csv")

    with workspace_context(workspace):
        materialized = workspace_materialize(b"col1\n", suffix=".csv")
        assert materialized.parent == workspace.scratch
        assert materialized.read_bytes() == b"col1\n"

    workspace.release()
    assert not materialized.exists()


def _csv_converter(root: Path, **kwargs) -> DocumentConverter:
    return DocumentConverter(
        allowed_formats=[InputFormat.CSV],
        workspace=WorkspaceSettings(enabled=True, root=root, **kwargs),
    )


def test_conversion_releases_workspace_on_success(tmp_path: Path) -> None:
    root = tmp_path / "workspace"
    csv_path = tmp_path / "sample.csv"
    csv_path.write_text("a,b\n1,2\n", encoding="utf-8")

    converter = _csv_converter(root, stale_dir_ttl_seconds=0)
    result = converter.convert(csv_path)

    assert result.status == ConversionStatus.SUCCESS
    assert _doc_dirs(root) == []


def test_conversion_releases_workspace_on_failure(tmp_path: Path) -> None:
    root = tmp_path / "workspace"
    # Unparseable binary content (no newline, not any known container)
    # cannot be matched to a backend and is reported as a document-level
    # failure rather than raised.
    broken_path = tmp_path / "broken.csv"
    broken_path.write_bytes(b"\x00\x01\x02\xff")

    converter = _csv_converter(root, stale_dir_ttl_seconds=0)
    result = converter.convert(broken_path, raises_on_error=False)

    assert result.status == ConversionStatus.FAILURE
    assert _doc_dirs(root) == []


def test_skipped_document_releases_workspace(tmp_path: Path) -> None:
    root = tmp_path / "workspace"
    markdown_path = tmp_path / "note.md"
    markdown_path.write_text("# title\n", encoding="utf-8")

    # Markdown is not in the allowed formats: the workspace was still
    # admitted for the document and must be released on the skip branch.
    converter = _csv_converter(root, stale_dir_ttl_seconds=0)
    result = converter.convert(markdown_path, raises_on_error=False)

    assert result.status in {ConversionStatus.SKIPPED, ConversionStatus.FAILURE}
    assert _doc_dirs(root) == []


def test_conversion_releases_workspace_on_interruption(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from docling.backend.csv_backend import CsvDocumentBackend

    root = tmp_path / "workspace"
    csv_path = tmp_path / "sample.csv"
    csv_path.write_text("a,b\n1,2\n", encoding="utf-8")

    def interrupted_convert(self):
        raise KeyboardInterrupt

    monkeypatch.setattr(CsvDocumentBackend, "convert", interrupted_convert)
    converter = _csv_converter(root, stale_dir_ttl_seconds=0)

    with pytest.raises(KeyboardInterrupt):
        converter.convert(csv_path)

    assert _doc_dirs(root) == []


def test_error_policy_rejects_document(tmp_path: Path) -> None:
    root = tmp_path / "workspace"
    csv_path = tmp_path / "sample.csv"
    csv_path.write_text("a,b\n1,2\n", encoding="utf-8")

    converter = _csv_converter(
        root,
        max_size_bytes=1,
        limit_policy=WorkspaceLimitPolicy.ERROR,
        stale_dir_ttl_seconds=0,
    )
    result = converter.convert(csv_path, raises_on_error=False)

    assert result.status == ConversionStatus.FAILURE
    assert result.errors
    assert any("workspace" in e.error_message.lower() for e in result.errors)
    assert _doc_dirs(root) == []


def test_stream_conversion_uses_and_releases_workspace(tmp_path: Path) -> None:
    root = tmp_path / "workspace"
    converter = _csv_converter(root, stale_dir_ttl_seconds=0)
    stream = DocumentStream(name="sample.csv", stream=BytesIO(b"a,b\n1,2\n"))

    result = converter.convert(stream)

    assert result.status == ConversionStatus.SUCCESS
    assert _doc_dirs(root) == []


def test_named_temp_file_helper_outside_workspace() -> None:
    handle = workspace_named_temp_file(suffix=".txt", delete=False)
    try:
        handle.write(b"x")
        path = Path(handle.name)
        handle.close()
        assert path.exists()
    finally:
        path.unlink(missing_ok=True)


def test_settings_scope_enables_workspace(tmp_path: Path) -> None:
    from docling.datamodel.settings import scoped

    root = tmp_path / "workspace"
    csv_path = tmp_path / "sample.csv"
    csv_path.write_text("a,b\n1,2\n", encoding="utf-8")

    # No explicit workspace argument: global settings drive governance,
    # including startup sweep and per-document release.
    with scoped(
        workspace=WorkspaceSettings(enabled=True, root=root, stale_dir_ttl_seconds=0)
    ):
        converter = DocumentConverter(allowed_formats=[InputFormat.CSV])
        result = converter.convert(csv_path)

    assert result.status == ConversionStatus.SUCCESS
    assert _doc_dirs(root) == []


def test_concurrent_conversions_isolated_and_released(tmp_path: Path) -> None:
    from docling.datamodel.settings import BatchConcurrencySettings, scoped

    root = tmp_path / "workspace"
    csv_paths = []
    for index in range(4):
        path = tmp_path / f"sample_{index}.csv"
        path.write_text(f"a,b\n{index},2\n", encoding="utf-8")
        csv_paths.append(path)

    workspace_settings = WorkspaceSettings(
        enabled=True, root=root, stale_dir_ttl_seconds=0
    )
    with scoped(
        perf=BatchConcurrencySettings(doc_batch_size=4, doc_batch_concurrency=4),
        workspace=workspace_settings,
    ):
        converter = DocumentConverter(
            allowed_formats=[InputFormat.CSV],
            workspace=workspace_settings,
        )
        results = list(converter.convert_all(csv_paths, raises_on_error=False))

    assert len(results) == 4
    assert all(r.status == ConversionStatus.SUCCESS for r in results)
    # Every concurrent document had its own directory, all released.
    assert _doc_dirs(root) == []
