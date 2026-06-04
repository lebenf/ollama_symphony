"""Tests for parse_tasks, parse_workflow, _slugify, and StateStore."""

import json
import textwrap
from pathlib import Path

import pytest

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from ollama_symphony import (
    Task, WorkflowConfig, TaskState,
    parse_tasks, parse_workflow, _slugify, StateStore,
)


# ---------------------------------------------------------------------------
# _slugify
# ---------------------------------------------------------------------------

def test_slugify_simple():
    assert _slugify("Hello World") == "Hello_World"

def test_slugify_special_chars():
    assert _slugify("Task 1: Setup/Init") == "Task_1__Setup_Init"

def test_slugify_already_clean():
    assert _slugify("my-task.v2") == "my-task.v2"

def test_slugify_strips_underscores():
    assert _slugify("  spaces  ") == "spaces"


# ---------------------------------------------------------------------------
# parse_tasks
# ---------------------------------------------------------------------------

SAMPLE_TASKS_MD = textwrap.dedent("""\
    # Project Tasks

    ## Task 1: First task

    Body of the first task.
    Multi-line body.

    ## Task 2: Second task

    Body of the second task.
""")


def test_parse_tasks_count(tmp_path):
    f = tmp_path / "TASKS.md"
    f.write_text(SAMPLE_TASKS_MD, encoding="utf-8")
    tasks = parse_tasks(f)
    assert len(tasks) == 2


def test_parse_tasks_titles(tmp_path):
    f = tmp_path / "TASKS.md"
    f.write_text(SAMPLE_TASKS_MD, encoding="utf-8")
    tasks = parse_tasks(f)
    assert tasks[0].title == "Task 1: First task"
    assert tasks[1].title == "Task 2: Second task"


def test_parse_tasks_body(tmp_path):
    f = tmp_path / "TASKS.md"
    f.write_text(SAMPLE_TASKS_MD, encoding="utf-8")
    tasks = parse_tasks(f)
    assert "Multi-line body" in tasks[0].body
    assert "second task" in tasks[1].body


def test_parse_tasks_index(tmp_path):
    f = tmp_path / "TASKS.md"
    f.write_text(SAMPLE_TASKS_MD, encoding="utf-8")
    tasks = parse_tasks(f)
    assert tasks[0].index == 0
    assert tasks[1].index == 1


def test_parse_tasks_slug_generated(tmp_path):
    f = tmp_path / "TASKS.md"
    f.write_text(SAMPLE_TASKS_MD, encoding="utf-8")
    tasks = parse_tasks(f)
    assert tasks[0].slug != ""
    assert tasks[1].slug != ""
    assert tasks[0].slug != tasks[1].slug


def test_parse_tasks_empty_raises(tmp_path):
    f = tmp_path / "TASKS.md"
    f.write_text("# Just a header, no tasks\n", encoding="utf-8")
    with pytest.raises(ValueError, match="No tasks found"):
        parse_tasks(f)


# ---------------------------------------------------------------------------
# parse_workflow — no front matter
# ---------------------------------------------------------------------------

def test_parse_workflow_no_frontmatter(tmp_path):
    f = tmp_path / "WORKFLOW.md"
    f.write_text("You are a dev agent.\n", encoding="utf-8")
    cfg = parse_workflow(f)
    assert isinstance(cfg, WorkflowConfig)
    assert "dev agent" in cfg.system_prompt
    assert cfg.max_retries == 3
    assert cfg.ollama_model == "qwen2.5-coder:7b"


# ---------------------------------------------------------------------------
# parse_workflow — with front matter, agent + git fields
# ---------------------------------------------------------------------------

WORKFLOW_AGENT_GIT = textwrap.dedent("""\
    ---
    agent:
      max_retries: 5
      retry_delay_s: 20
      turn_timeout_s: 300
      max_iterations: 10
    git:
      commit_after_each_task: false
      commit_message_template: "chore: {task_title}"
    ---
    Custom system prompt.
""")


def test_parse_workflow_agent_fields(tmp_path):
    f = tmp_path / "WORKFLOW.md"
    f.write_text(WORKFLOW_AGENT_GIT, encoding="utf-8")
    cfg = parse_workflow(f)
    assert cfg.max_retries == 5
    assert cfg.retry_delay_s == 20.0
    assert cfg.turn_timeout_s == 300
    assert cfg.max_iterations == 10


def test_parse_workflow_git_fields(tmp_path):
    f = tmp_path / "WORKFLOW.md"
    f.write_text(WORKFLOW_AGENT_GIT, encoding="utf-8")
    cfg = parse_workflow(f)
    assert cfg.commit_after_each_task is False
    assert cfg.commit_message_template == "chore: {task_title}"


def test_parse_workflow_system_prompt(tmp_path):
    f = tmp_path / "WORKFLOW.md"
    f.write_text(WORKFLOW_AGENT_GIT, encoding="utf-8")
    cfg = parse_workflow(f)
    assert "Custom system prompt" in cfg.system_prompt


# ---------------------------------------------------------------------------
# parse_workflow — ollama + tools fields
# ---------------------------------------------------------------------------

WORKFLOW_OLLAMA_TOOLS = textwrap.dedent("""\
    ---
    ollama:
      hosts:
        - http://host1:11434
        - http://host2:11434
      model: llama3.1:8b
      timeout_s: 60
      temperature: 0.5
      context_window: 4096
      num_ctx: 4096
    tools:
      enabled:
        - run_shell
        - read_file
      shell_timeout_s: 15
      working_dir: "/tmp/work"
    ---
    Ollama agent prompt.
""")


def test_parse_workflow_ollama_hosts(tmp_path):
    f = tmp_path / "WORKFLOW.md"
    f.write_text(WORKFLOW_OLLAMA_TOOLS, encoding="utf-8")
    cfg = parse_workflow(f)
    assert cfg.ollama_hosts == ["http://host1:11434", "http://host2:11434"]


def test_parse_workflow_ollama_model(tmp_path):
    f = tmp_path / "WORKFLOW.md"
    f.write_text(WORKFLOW_OLLAMA_TOOLS, encoding="utf-8")
    cfg = parse_workflow(f)
    assert cfg.ollama_model == "llama3.1:8b"


def test_parse_workflow_ollama_numerics(tmp_path):
    f = tmp_path / "WORKFLOW.md"
    f.write_text(WORKFLOW_OLLAMA_TOOLS, encoding="utf-8")
    cfg = parse_workflow(f)
    assert cfg.ollama_timeout_s == 60
    assert cfg.ollama_temperature == 0.5
    assert cfg.ollama_context_window == 4096
    assert cfg.ollama_num_ctx == 4096


def test_parse_workflow_tools_enabled(tmp_path):
    f = tmp_path / "WORKFLOW.md"
    f.write_text(WORKFLOW_OLLAMA_TOOLS, encoding="utf-8")
    cfg = parse_workflow(f)
    assert cfg.enabled_tools == ["run_shell", "read_file"]


def test_parse_workflow_tools_shell_timeout(tmp_path):
    f = tmp_path / "WORKFLOW.md"
    f.write_text(WORKFLOW_OLLAMA_TOOLS, encoding="utf-8")
    cfg = parse_workflow(f)
    assert cfg.shell_timeout_s == 15
    assert cfg.working_dir == "/tmp/work"


# ---------------------------------------------------------------------------
# StateStore
# ---------------------------------------------------------------------------

def test_state_store_init(tmp_path):
    store = StateStore(tmp_path / "state.json")
    store.init_run("TASKS.md")
    assert (tmp_path / "state.json").exists()


def test_state_store_set_get(tmp_path):
    store = StateStore(tmp_path / "state.json")
    store.init_run("TASKS.md")
    state = TaskState(status="completed", attempts=1, completed_at="2024-01-01T00:00:00+00:00")
    store.set("my_task", state)
    loaded = store.get("my_task")
    assert loaded is not None
    assert loaded.status == "completed"
    assert loaded.attempts == 1


def test_state_store_is_completed(tmp_path):
    store = StateStore(tmp_path / "state.json")
    store.init_run("TASKS.md")
    assert store.is_completed("missing_task") is False
    store.set("t1", TaskState(status="completed"))
    assert store.is_completed("t1") is True
    store.set("t2", TaskState(status="failed"))
    assert store.is_completed("t2") is False


def test_state_store_resume(tmp_path):
    """State persists across StateStore instances (simulates resume after restart)."""
    state_file = tmp_path / "state.json"
    store1 = StateStore(state_file)
    store1.init_run("TASKS.md")
    store1.set("task_one", TaskState(status="completed", attempts=1))

    store2 = StateStore(state_file)
    assert store2.is_completed("task_one") is True


def test_state_store_unknown_slug_returns_none(tmp_path):
    store = StateStore(tmp_path / "state.json")
    store.init_run("TASKS.md")
    assert store.get("nonexistent") is None


def test_state_store_corrupted_file(tmp_path):
    """Corrupted state file should not crash — falls back to empty state."""
    state_file = tmp_path / "state.json"
    state_file.write_text("{ invalid json }", encoding="utf-8")
    store = StateStore(state_file)  # should not raise
    store.init_run("TASKS.md")
    assert store.get("anything") is None
