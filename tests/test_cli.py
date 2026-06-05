# Copyright (C) 2026 Lorenzo Benfenati
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tests for CLI option parsing and main() behaviour (no live Ollama)."""

import json
import sys
import textwrap
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
import ollama_symphony as _mod
from ollama_symphony import main, WorkflowConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_tasks(path: Path, content: str | None = None) -> Path:
    text = content or textwrap.dedent("""\
        ## Task 1: Do thing
        Body of task 1.
    """)
    path.write_text(text, encoding="utf-8")
    return path


def _write_workflow(path: Path) -> Path:
    path.write_text("You are an agent.\n", encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

def test_defaults(tmp_path, monkeypatch):
    """Unprovided options resolve to their documented defaults."""
    monkeypatch.chdir(tmp_path)
    _write_tasks(tmp_path / "TASKS.md")
    _write_workflow(tmp_path / "WORKFLOW.md")

    with patch.object(sys, "argv", ["ollama_symphony.py", "--dry-run"]):
        with patch("ollama_symphony.OllamaSymphonyRunner") as MockRunner:
            instance = MockRunner.return_value
            instance.run.return_value = True
            with pytest.raises(SystemExit) as exc:
                main()
            assert exc.value.code == 0

        _, kwargs = MockRunner.call_args
        # state path default
        state_arg = MockRunner.call_args[0][2]  # positional: tasks, config, state
        assert "TASKS.state.json" in str(state_arg.path)


def test_default_tasks_file_missing(tmp_path, monkeypatch):
    """Missing TASKS.md with default path → exit(1)."""
    monkeypatch.chdir(tmp_path)
    with patch.object(sys, "argv", ["ollama_symphony.py"]):
        with pytest.raises(SystemExit) as exc:
            main()
    assert exc.value.code == 1


# ---------------------------------------------------------------------------
# --tasks / --workflow / --state custom paths
# ---------------------------------------------------------------------------

def test_custom_tasks_path(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    tasks_file = tmp_path / "my_tasks.md"
    _write_tasks(tasks_file)
    _write_workflow(tmp_path / "WORKFLOW.md")

    with patch.object(sys, "argv", [
        "ollama_symphony.py", "--tasks", str(tasks_file), "--dry-run",
    ]):
        with patch("ollama_symphony.OllamaSymphonyRunner") as MockRunner:
            MockRunner.return_value.run.return_value = True
            with pytest.raises(SystemExit) as exc:
                main()
    assert exc.value.code == 0


def test_custom_workflow_path(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _write_tasks(tmp_path / "TASKS.md")
    wf_file = tmp_path / "my_workflow.md"
    wf_file.write_text(textwrap.dedent("""\
        ---
        agent:
          max_retries: 7
        ---
        Custom prompt.
    """), encoding="utf-8")

    with patch.object(sys, "argv", [
        "ollama_symphony.py", "--workflow", str(wf_file), "--dry-run",
    ]):
        with patch("ollama_symphony.OllamaSymphonyRunner") as MockRunner:
            MockRunner.return_value.run.return_value = True
            with pytest.raises(SystemExit) as exc:
                main()
    assert exc.value.code == 0
    # config passed to runner has the custom max_retries
    cfg_arg = MockRunner.call_args[0][1]
    assert cfg_arg.max_retries == 7


def test_custom_state_path(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _write_tasks(tmp_path / "TASKS.md")
    state_file = tmp_path / "custom.state.json"

    with patch.object(sys, "argv", [
        "ollama_symphony.py", "--state", str(state_file), "--dry-run",
    ]):
        with patch("ollama_symphony.OllamaSymphonyRunner") as MockRunner:
            MockRunner.return_value.run.return_value = True
            with pytest.raises(SystemExit) as exc:
                main()
    assert exc.value.code == 0
    state_arg = MockRunner.call_args[0][2]
    assert state_arg.path == state_file.resolve() or state_arg.path == state_file


# ---------------------------------------------------------------------------
# --reset
# ---------------------------------------------------------------------------

def test_reset_deletes_state_file(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _write_tasks(tmp_path / "TASKS.md")
    state_file = tmp_path / "TASKS.state.json"
    state_file.write_text('{"started_at": "x", "tasks": {}}', encoding="utf-8")

    with patch.object(sys, "argv", [
        "ollama_symphony.py", "--state", str(state_file), "--reset", "--dry-run",
    ]):
        with patch("ollama_symphony.OllamaSymphonyRunner") as MockRunner:
            MockRunner.return_value.run.return_value = True
            with pytest.raises(SystemExit) as exc:
                main()
    assert exc.value.code == 0
    # file should have been deleted and re-created empty (no pre-existing tasks)
    reloaded = json.loads(state_file.read_text())
    assert reloaded.get("tasks") == {}


def test_reset_noop_when_no_state_file(tmp_path, monkeypatch):
    """--reset with no existing state file should not crash."""
    monkeypatch.chdir(tmp_path)
    _write_tasks(tmp_path / "TASKS.md")
    state_file = tmp_path / "nonexistent.state.json"
    assert not state_file.exists()

    with patch.object(sys, "argv", [
        "ollama_symphony.py", "--state", str(state_file), "--reset", "--dry-run",
    ]):
        with patch("ollama_symphony.OllamaSymphonyRunner") as MockRunner:
            MockRunner.return_value.run.return_value = True
            with pytest.raises(SystemExit) as exc:
                main()
    assert exc.value.code == 0


# ---------------------------------------------------------------------------
# --dry-run
# ---------------------------------------------------------------------------

def test_dry_run_does_not_invoke_ollama(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _write_tasks(tmp_path / "TASKS.md")

    with patch.object(sys, "argv", ["ollama_symphony.py", "--dry-run"]):
        with patch("ollama.Client") as MockOllama:
            with patch("ollama_symphony.OllamaSymphonyRunner") as MockRunner:
                MockRunner.return_value.run.return_value = True
                with pytest.raises(SystemExit):
                    main()
            MockOllama.assert_not_called()


def test_dry_run_flag_passed_to_runner(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _write_tasks(tmp_path / "TASKS.md")

    with patch.object(sys, "argv", ["ollama_symphony.py", "--dry-run"]):
        with patch("ollama_symphony.OllamaSymphonyRunner") as MockRunner:
            MockRunner.return_value.run.return_value = True
            with pytest.raises(SystemExit):
                main()
    _, kwargs = MockRunner.call_args
    # dry_run passed as keyword or positional
    call_args = MockRunner.call_args
    dry_run_val = (
        call_args.kwargs.get("dry_run")
        if call_args.kwargs.get("dry_run") is not None
        else call_args[0][3] if len(call_args[0]) > 3 else None
    )
    assert dry_run_val is True


# ---------------------------------------------------------------------------
# --verbose / -v
# ---------------------------------------------------------------------------

def test_verbose_flag(tmp_path, monkeypatch, capfd):
    import logging
    monkeypatch.chdir(tmp_path)
    _write_tasks(tmp_path / "TASKS.md")

    with patch.object(sys, "argv", ["ollama_symphony.py", "--verbose", "--dry-run"]):
        with patch("ollama_symphony.OllamaSymphonyRunner") as MockRunner:
            MockRunner.return_value.run.return_value = True
            with patch("ollama_symphony.setup_logging") as mock_setup:
                with pytest.raises(SystemExit):
                    main()
    mock_setup.assert_called_once_with(True)


def test_v_short_flag(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _write_tasks(tmp_path / "TASKS.md")

    with patch.object(sys, "argv", ["ollama_symphony.py", "-v", "--dry-run"]):
        with patch("ollama_symphony.OllamaSymphonyRunner") as MockRunner:
            MockRunner.return_value.run.return_value = True
            with patch("ollama_symphony.setup_logging") as mock_setup:
                with pytest.raises(SystemExit):
                    main()
    mock_setup.assert_called_once_with(True)


def test_no_verbose_flag(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _write_tasks(tmp_path / "TASKS.md")

    with patch.object(sys, "argv", ["ollama_symphony.py", "--dry-run"]):
        with patch("ollama_symphony.OllamaSymphonyRunner") as MockRunner:
            MockRunner.return_value.run.return_value = True
            with patch("ollama_symphony.setup_logging") as mock_setup:
                with pytest.raises(SystemExit):
                    main()
    mock_setup.assert_called_once_with(False)


# ---------------------------------------------------------------------------
# --list-models
# ---------------------------------------------------------------------------

def test_list_models_exits_zero(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    with patch.object(sys, "argv", ["ollama_symphony.py", "--list-models"]):
        with patch("ollama_symphony.list_models") as mock_lm:
            with pytest.raises(SystemExit) as exc:
                main()
    assert exc.value.code == 0
    mock_lm.assert_called_once()


def test_list_models_no_tasks_needed(tmp_path, monkeypatch):
    """--list-models exits before reading TASKS.md, so missing file is OK."""
    monkeypatch.chdir(tmp_path)
    # no TASKS.md created

    with patch.object(sys, "argv", ["ollama_symphony.py", "--list-models"]):
        with patch("ollama_symphony.list_models"):
            with pytest.raises(SystemExit) as exc:
                main()
    assert exc.value.code == 0


# ---------------------------------------------------------------------------
# --check
# ---------------------------------------------------------------------------

def test_check_exits_zero_on_success(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    with patch.object(sys, "argv", ["ollama_symphony.py", "--check"]):
        with patch("ollama_symphony.check_connectivity", return_value=True) as mock_cc:
            with pytest.raises(SystemExit) as exc:
                main()
    assert exc.value.code == 0
    mock_cc.assert_called_once()


def test_check_exits_one_on_failure(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    with patch.object(sys, "argv", ["ollama_symphony.py", "--check"]):
        with patch("ollama_symphony.check_connectivity", return_value=False):
            with pytest.raises(SystemExit) as exc:
                main()
    assert exc.value.code == 1


def test_check_no_tasks_needed(tmp_path, monkeypatch):
    """--check exits before reading TASKS.md."""
    monkeypatch.chdir(tmp_path)

    with patch.object(sys, "argv", ["ollama_symphony.py", "--check"]):
        with patch("ollama_symphony.check_connectivity", return_value=True):
            with pytest.raises(SystemExit) as exc:
                main()
    assert exc.value.code == 0


# ---------------------------------------------------------------------------
# Invalid / missing TASKS.md
# ---------------------------------------------------------------------------

def test_tasks_file_not_found(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    with patch.object(sys, "argv", [
        "ollama_symphony.py", "--tasks", str(tmp_path / "nope.md"),
    ]):
        with pytest.raises(SystemExit) as exc:
            main()
    assert exc.value.code == 1


def test_tasks_file_no_headings(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    bad = tmp_path / "TASKS.md"
    bad.write_text("# Just a top-level header\nNo ## tasks here.\n", encoding="utf-8")

    with patch.object(sys, "argv", ["ollama_symphony.py"]):
        with pytest.raises(SystemExit) as exc:
            main()
    assert exc.value.code == 1
