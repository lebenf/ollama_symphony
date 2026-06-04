"""Tests for tool execution functions and ToolExecutor."""

import sys
import textwrap
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
from ollama_symphony import (
    WorkflowConfig, ToolCall, ToolResult,
    _safe_path,
    execute_run_shell, execute_read_file, execute_write_file, execute_list_directory,
    build_tool_schemas, ToolExecutor,
)


# ---------------------------------------------------------------------------
# _safe_path
# ---------------------------------------------------------------------------

def test_safe_path_normal(tmp_path):
    result = _safe_path(tmp_path, "subdir/file.txt")
    assert str(result).startswith(str(tmp_path))


def test_safe_path_traversal_raises(tmp_path):
    with pytest.raises(ValueError, match="Path traversal blocked"):
        _safe_path(tmp_path, "../outside.txt")


def test_safe_path_double_traversal_raises(tmp_path):
    with pytest.raises(ValueError, match="Path traversal blocked"):
        _safe_path(tmp_path, "subdir/../../outside.txt")


# ---------------------------------------------------------------------------
# execute_run_shell
# ---------------------------------------------------------------------------

def test_run_shell_success(tmp_path):
    result = execute_run_shell("echo hello", tmp_path, timeout_s=10)
    assert result.success is True
    assert result.exit_code == 0
    assert "hello" in result.output


def test_run_shell_failure(tmp_path):
    result = execute_run_shell("exit 1", tmp_path, timeout_s=10)
    assert result.success is False
    assert result.exit_code == 1


def test_run_shell_timeout(tmp_path):
    result = execute_run_shell("sleep 10", tmp_path, timeout_s=1)
    assert result.success is False
    assert result.exit_code == -1
    assert "timed out" in result.output.lower()


def test_run_shell_output_format(tmp_path):
    result = execute_run_shell("echo hello", tmp_path, timeout_s=10)
    assert "exit_code:" in result.output
    assert "stdout:" in result.output
    assert "stderr:" in result.output


# ---------------------------------------------------------------------------
# execute_read_file
# ---------------------------------------------------------------------------

def test_read_file_existing(tmp_path):
    f = tmp_path / "hello.txt"
    f.write_text("hello world", encoding="utf-8")
    result = execute_read_file("hello.txt", tmp_path)
    assert result.success is True
    assert "hello world" in result.output


def test_read_file_not_found(tmp_path):
    result = execute_read_file("missing.txt", tmp_path)
    assert result.success is False
    assert "not found" in result.output.lower()


def test_read_file_traversal_blocked(tmp_path):
    result = execute_read_file("../secret.txt", tmp_path)
    assert result.success is False
    assert "traversal" in result.output.lower() or "blocked" in result.output.lower()


def test_read_file_truncation(tmp_path):
    f = tmp_path / "big.txt"
    f.write_text("x" * 10000, encoding="utf-8")
    result = execute_read_file("big.txt", tmp_path)
    assert result.success is True
    assert "truncated" in result.output


# ---------------------------------------------------------------------------
# execute_write_file
# ---------------------------------------------------------------------------

def test_write_file_creates(tmp_path):
    result = execute_write_file("new.txt", "content", tmp_path)
    assert result.success is True
    assert (tmp_path / "new.txt").read_text() == "content"


def test_write_file_overwrites(tmp_path):
    f = tmp_path / "existing.txt"
    f.write_text("old content")
    result = execute_write_file("existing.txt", "new content", tmp_path)
    assert result.success is True
    assert f.read_text() == "new content"


def test_write_file_creates_parent_dirs(tmp_path):
    result = execute_write_file("a/b/c/file.txt", "nested", tmp_path)
    assert result.success is True
    assert (tmp_path / "a" / "b" / "c" / "file.txt").exists()


def test_write_file_traversal_blocked(tmp_path):
    result = execute_write_file("../escape.txt", "bad", tmp_path)
    assert result.success is False
    assert "traversal" in result.output.lower() or "blocked" in result.output.lower()


# ---------------------------------------------------------------------------
# execute_list_directory
# ---------------------------------------------------------------------------

def test_list_directory_existing(tmp_path):
    (tmp_path / "file.txt").write_text("x")
    (tmp_path / "subdir").mkdir()
    result = execute_list_directory(".", tmp_path)
    assert result.success is True
    assert "file.txt" in result.output
    assert "subdir" in result.output


def test_list_directory_empty(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    result = execute_list_directory("empty", tmp_path)
    assert result.success is True
    assert "empty" in result.output.lower()


def test_list_directory_not_found(tmp_path):
    result = execute_list_directory("nonexistent", tmp_path)
    assert result.success is False


def test_list_directory_default_path(tmp_path):
    (tmp_path / "readme.md").write_text("x")
    result = execute_list_directory("", tmp_path)
    assert result.success is True
    assert "readme.md" in result.output


# ---------------------------------------------------------------------------
# build_tool_schemas
# ---------------------------------------------------------------------------

def test_build_tool_schemas_task_complete_always_present():
    cfg = WorkflowConfig(enabled_tools=[])
    schemas = build_tool_schemas(cfg)
    names = [s["function"]["name"] for s in schemas]
    assert "task_complete" in names


def test_build_tool_schemas_respects_enabled():
    cfg = WorkflowConfig(enabled_tools=["run_shell", "read_file"])
    schemas = build_tool_schemas(cfg)
    names = [s["function"]["name"] for s in schemas]
    assert "run_shell" in names
    assert "read_file" in names
    assert "write_file" not in names
    assert "list_directory" not in names
    assert "task_complete" in names


def test_build_tool_schemas_no_duplicate_task_complete():
    cfg = WorkflowConfig(enabled_tools=["task_complete"])
    schemas = build_tool_schemas(cfg)
    names = [s["function"]["name"] for s in schemas]
    assert names.count("task_complete") == 1


def test_build_tool_schemas_structure():
    cfg = WorkflowConfig(enabled_tools=["run_shell"])
    schemas = build_tool_schemas(cfg)
    for schema in schemas:
        assert "type" in schema
        assert "function" in schema
        assert "name" in schema["function"]
        assert "parameters" in schema["function"]


# ---------------------------------------------------------------------------
# ToolExecutor
# ---------------------------------------------------------------------------

def test_tool_executor_run_shell(tmp_path):
    cfg = WorkflowConfig(working_dir=str(tmp_path))
    executor = ToolExecutor(cfg)
    result = executor.execute(ToolCall(name="run_shell", arguments={"command": "echo hi"}))
    assert result.success is True
    assert "hi" in result.output


def test_tool_executor_unknown_tool(tmp_path):
    cfg = WorkflowConfig(working_dir=str(tmp_path))
    executor = ToolExecutor(cfg)
    result = executor.execute(ToolCall(name="nonexistent_tool", arguments={}))
    assert result.success is False
    assert "Unknown tool" in result.output


def test_tool_executor_task_complete(tmp_path):
    cfg = WorkflowConfig(working_dir=str(tmp_path))
    executor = ToolExecutor(cfg)
    result = executor.execute(ToolCall(name="task_complete", arguments={"summary": "done"}))
    # task_complete is intercepted by the loop, but executor returns success
    assert result.success is True
