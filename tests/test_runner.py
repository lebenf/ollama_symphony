# Copyright (C) 2026 Lorenzo Benfenati
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tests for OllamaSymphonyRunner. Mocks ReactLoop.run — no Ollama server needed."""

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
from ollama_symphony import (
    Task, WorkflowConfig, TaskState, StateStore,
    OllamaSymphonyRunner,
)


def make_tasks(n: int = 2) -> list[Task]:
    return [Task(index=i, title=f"Task {i+1}", body=f"Body {i+1}") for i in range(n)]


def make_runner(tasks, tmp_path, dry_run=False, **cfg_kwargs) -> OllamaSymphonyRunner:
    config = WorkflowConfig(
        commit_after_each_task=False,  # no git in tests
        retry_delay_s=0,
        **cfg_kwargs,
    )
    state = StateStore(tmp_path / "state.json")
    state.init_run("TASKS.md")
    return OllamaSymphonyRunner(tasks, config, state, dry_run=dry_run)


# ---------------------------------------------------------------------------
# All tasks succeed
# ---------------------------------------------------------------------------

def test_runner_all_tasks_succeed(tmp_path):
    tasks = make_tasks(2)
    runner = make_runner(tasks, tmp_path)

    with patch.object(runner, "_run_single_attempt", return_value=(True, "done")):
        result = runner.run()

    assert result is True
    assert runner.state.is_completed(tasks[0].slug)
    assert runner.state.is_completed(tasks[1].slug)


# ---------------------------------------------------------------------------
# Task fails after all retries
# ---------------------------------------------------------------------------

def test_runner_task_fails_all_retries(tmp_path):
    tasks = make_tasks(2)
    runner = make_runner(tasks, tmp_path, max_retries=1)

    with patch.object(runner, "_run_single_attempt", return_value=(False, "error")):
        result = runner.run()

    assert result is False


# ---------------------------------------------------------------------------
# Resume: already completed tasks are skipped
# ---------------------------------------------------------------------------

def test_runner_resume_skips_completed(tmp_path):
    tasks = make_tasks(2)
    state = StateStore(tmp_path / "state.json")
    state.init_run("TASKS.md")
    # Mark task 0 as already completed
    state.set(tasks[0].slug, TaskState(status="completed", attempts=1))

    config = WorkflowConfig(commit_after_each_task=False, retry_delay_s=0)
    runner = OllamaSymphonyRunner(tasks, config, state, dry_run=False)

    call_log = []

    def mock_attempt(task, completed_so_far, attempt):
        call_log.append(task.title)
        return True, "done"

    with patch.object(runner, "_run_single_attempt", side_effect=mock_attempt):
        result = runner.run()

    assert result is True
    # Only task 1 (index 1) should have been executed
    assert len(call_log) == 1
    assert "Task 2" in call_log[0]


# ---------------------------------------------------------------------------
# dry-run: no Ollama calls
# ---------------------------------------------------------------------------

def test_runner_dry_run_no_ollama(tmp_path):
    tasks = make_tasks(1)
    runner = make_runner(tasks, tmp_path, dry_run=True)

    with patch.object(runner._client, "chat") as mock_chat:
        result = runner.run()

    assert result is True
    mock_chat.assert_not_called()
