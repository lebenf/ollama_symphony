"""Tests for tool execution, ReAct loop, and runner (no live Ollama needed)."""

import logging
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
from ollama_symphony import (
    Task, WorkflowConfig, TaskState, ToolCall,
    _safe_path, execute_tool,
    _tool_read_file, _tool_write_file, _tool_run_shell, _tool_list_directory,
    build_prompt, run_react_loop, OllamaSymphonyRunner, StateStore,
)

_LOG = logging.getLogger("test")


# ---------------------------------------------------------------------------
# _safe_path — path traversal
# ---------------------------------------------------------------------------

def test_safe_path_normal(tmp_path):
    p = _safe_path("src/main.py", str(tmp_path))
    assert p == (tmp_path / "src" / "main.py").resolve()


def test_safe_path_traversal_blocked(tmp_path):
    with pytest.raises(ValueError, match="Path traversal"):
        _safe_path("../../etc/passwd", str(tmp_path))


def test_safe_path_absolute_traversal_blocked(tmp_path):
    with pytest.raises(ValueError, match="Path traversal"):
        _safe_path("/etc/passwd", str(tmp_path))


def test_safe_path_dot_traversal_blocked(tmp_path):
    with pytest.raises(ValueError, match="Path traversal"):
        _safe_path("../sibling", str(tmp_path))


# ---------------------------------------------------------------------------
# read_file
# ---------------------------------------------------------------------------

def test_read_file_success(tmp_path):
    cfg = WorkflowConfig(working_dir=str(tmp_path))
    (tmp_path / "hello.txt").write_text("hello world", encoding="utf-8")
    result = _tool_read_file("hello.txt", cfg)
    assert result.success is True
    assert result.output == "hello world"


def test_read_file_not_found(tmp_path):
    cfg = WorkflowConfig(working_dir=str(tmp_path))
    result = _tool_read_file("missing.txt", cfg)
    assert result.success is False
    assert "not found" in result.output.lower()


def test_read_file_traversal_blocked(tmp_path):
    cfg = WorkflowConfig(working_dir=str(tmp_path))
    result = _tool_read_file("../../etc/passwd", cfg)
    assert result.success is False
    assert "traversal" in result.output.lower()


# ---------------------------------------------------------------------------
# write_file
# ---------------------------------------------------------------------------

def test_write_file_success(tmp_path):
    cfg = WorkflowConfig(working_dir=str(tmp_path))
    result = _tool_write_file("out.txt", "content", cfg)
    assert result.success is True
    assert (tmp_path / "out.txt").read_text() == "content"


def test_write_file_creates_dirs(tmp_path):
    cfg = WorkflowConfig(working_dir=str(tmp_path))
    result = _tool_write_file("a/b/c.txt", "nested", cfg)
    assert result.success is True
    assert (tmp_path / "a" / "b" / "c.txt").read_text() == "nested"


def test_write_file_traversal_blocked(tmp_path):
    cfg = WorkflowConfig(working_dir=str(tmp_path))
    result = _tool_write_file("../evil.txt", "pwned", cfg)
    assert result.success is False
    assert "traversal" in result.output.lower()


# ---------------------------------------------------------------------------
# run_shell
# ---------------------------------------------------------------------------

def test_run_shell_success(tmp_path):
    cfg = WorkflowConfig(working_dir=str(tmp_path))
    result = _tool_run_shell("echo hello", cfg, _LOG)
    assert result.success is True
    assert "hello" in result.output


def test_run_shell_failure(tmp_path):
    cfg = WorkflowConfig(working_dir=str(tmp_path))
    result = _tool_run_shell("exit 1", cfg, _LOG)
    assert result.success is False
    assert result.exit_code == 1


def test_run_shell_timeout(tmp_path):
    cfg = WorkflowConfig(working_dir=str(tmp_path), shell_timeout_s=1)
    result = _tool_run_shell("sleep 10", cfg, _LOG)
    assert result.success is False
    assert "timed out" in result.output.lower()


# ---------------------------------------------------------------------------
# list_directory
# ---------------------------------------------------------------------------

def test_list_directory(tmp_path):
    cfg = WorkflowConfig(working_dir=str(tmp_path))
    (tmp_path / "file.txt").write_text("x")
    (tmp_path / "subdir").mkdir()
    result = _tool_list_directory(".", cfg)
    assert result.success is True
    assert "file.txt" in result.output
    assert "subdir/" in result.output


def test_list_directory_traversal_blocked(tmp_path):
    cfg = WorkflowConfig(working_dir=str(tmp_path))
    result = _tool_list_directory("../../", cfg)
    assert result.success is False


# ---------------------------------------------------------------------------
# execute_tool dispatch
# ---------------------------------------------------------------------------

def test_execute_tool_unknown():
    cfg = WorkflowConfig()
    call = ToolCall(name="nonexistent_tool", arguments={})
    result = execute_tool(call, cfg, _LOG)
    assert result.success is False
    assert "Unknown tool" in result.output


def test_execute_tool_task_complete():
    cfg = WorkflowConfig()
    call = ToolCall(name="task_complete", arguments={"summary": "done"})
    result = execute_tool(call, cfg, _LOG)
    assert result.success is True
    assert result.output == "done"


# ---------------------------------------------------------------------------
# build_prompt
# ---------------------------------------------------------------------------

def test_build_prompt_no_completed():
    task = Task(index=0, title="My task", body="Do something.")
    cfg = WorkflowConfig()
    prompt = build_prompt(task, cfg, [])
    assert "## Current task: My task" in prompt
    assert "Do something." in prompt


def test_build_prompt_with_completed():
    task = Task(index=1, title="Second task", body="Do more.")
    done = Task(index=0, title="First task", body="Already done.")
    cfg = WorkflowConfig()
    prompt = build_prompt(task, cfg, [done])
    assert "First task" in prompt
    assert "Previously completed" in prompt
    assert "## Current task: Second task" in prompt


# ---------------------------------------------------------------------------
# run_react_loop — dry-run (no Ollama)
# ---------------------------------------------------------------------------

def test_react_loop_dry_run():
    task = Task(index=0, title="T", body="Do it.")
    cfg = WorkflowConfig()
    success, output = run_react_loop(task, cfg, "prompt", dry_run=True)
    assert success is True
    assert "dry-run" in output


# ---------------------------------------------------------------------------
# run_react_loop — mocked Ollama
# ---------------------------------------------------------------------------

def test_react_loop_task_complete_via_tool(tmp_path):
    """Ollama returns task_complete tool call — loop succeeds."""
    task = Task(index=0, title="T", body="Do it.")
    cfg = WorkflowConfig(working_dir=str(tmp_path))

    mock_tc = MagicMock()
    mock_tc.function.name = "task_complete"
    mock_tc.function.arguments = {"summary": "All done"}

    mock_msg = MagicMock()
    mock_msg.content = ""
    mock_msg.tool_calls = [mock_tc]

    mock_response = MagicMock()
    mock_response.message = mock_msg

    mock_client = MagicMock()
    mock_client.chat.return_value = mock_response

    with patch("ollama.Client", return_value=mock_client):
        success, output = run_react_loop(task, cfg, "prompt")

    assert success is True
    assert output == "All done"


def test_react_loop_max_iterations(tmp_path):
    """No task_complete after max_iterations — loop fails."""
    task = Task(index=0, title="T", body="Do it.")
    cfg = WorkflowConfig(working_dir=str(tmp_path), max_iterations=2)

    mock_msg = MagicMock()
    mock_msg.content = "thinking..."
    mock_msg.tool_calls = None

    mock_response = MagicMock()
    mock_response.message = mock_msg

    mock_client = MagicMock()
    mock_client.chat.return_value = mock_response

    with patch("ollama.Client", return_value=mock_client):
        success, output = run_react_loop(task, cfg, "prompt")

    assert success is False
    assert "Max iterations" in output


def test_react_loop_ollama_error(tmp_path):
    """Ollama raises exception — loop returns failure."""
    task = Task(index=0, title="T", body="Do it.")
    cfg = WorkflowConfig(working_dir=str(tmp_path))

    mock_client = MagicMock()
    mock_client.chat.side_effect = Exception("connection refused")

    with patch("ollama.Client", return_value=mock_client):
        success, output = run_react_loop(task, cfg, "prompt")

    assert success is False
    assert "Ollama error" in output


# ---------------------------------------------------------------------------
# OllamaSymphonyRunner — dry-run (no Ollama)
# ---------------------------------------------------------------------------

def test_runner_dry_run(tmp_path):
    tasks = [Task(index=0, title="T1", body="body")]
    cfg = WorkflowConfig(working_dir=str(tmp_path))
    state = StateStore(tmp_path / "state.json")
    state.init_run("TASKS.md")

    runner = OllamaSymphonyRunner(tasks, cfg, state, dry_run=True)
    success = runner.run()

    assert success is True
    assert state.is_completed("T1")


def test_runner_skips_completed(tmp_path):
    tasks = [Task(index=0, title="T1", body="body")]
    cfg = WorkflowConfig(working_dir=str(tmp_path))
    state = StateStore(tmp_path / "state.json")
    state.init_run("TASKS.md")
    state.set("T1", TaskState(status="completed", attempts=1,
                              completed_at="2024-01-01T00:00:00+00:00"))

    runner = OllamaSymphonyRunner(tasks, cfg, state, dry_run=True)
    success = runner.run()

    assert success is True


def test_runner_multiple_tasks_dry_run(tmp_path):
    tasks = [
        Task(index=0, title="T1", body="first"),
        Task(index=1, title="T2", body="second"),
    ]
    cfg = WorkflowConfig(working_dir=str(tmp_path))
    state = StateStore(tmp_path / "state.json")
    state.init_run("TASKS.md")

    runner = OllamaSymphonyRunner(tasks, cfg, state, dry_run=True)
    success = runner.run()

    assert success is True
    assert state.is_completed("T1")
    assert state.is_completed("T2")
