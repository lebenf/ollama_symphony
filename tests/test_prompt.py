"""Tests for build_task_prompt and build_initial_messages."""

import pytest
from ollama_symphony import Task, build_task_prompt, build_initial_messages


def _task(title: str, body: str = "Do the thing.", index: int = 0) -> Task:
    return Task(index=index, title=title, body=body)


# ---------------------------------------------------------------------------
# build_task_prompt
# ---------------------------------------------------------------------------

class TestBuildTaskPrompt:
    def test_basic_no_completed_no_retry(self):
        task = _task("Add feature X", "Implement X.")
        result = build_task_prompt(task, [], attempt=1)
        assert "## Current task: Add feature X" in result
        assert "Implement X." in result
        assert "Previously completed" not in result
        assert "retry attempt" not in result

    def test_completed_tasks_listed(self):
        done = [_task("Task A"), _task("Task B")]
        task = _task("Task C")
        result = build_task_prompt(task, done, attempt=1)
        assert "## Previously completed tasks (already committed):" in result
        assert "- Task A" in result
        assert "- Task B" in result
        assert "Do not redo these" in result

    def test_retry_note_appears_on_attempt_2(self):
        task = _task("Task X")
        result = build_task_prompt(task, [], attempt=2)
        assert "retry attempt 2" in result
        assert "Previous attempt failed" in result

    def test_retry_note_absent_on_attempt_1(self):
        task = _task("Task X")
        result = build_task_prompt(task, [], attempt=1)
        assert "retry attempt" not in result

    def test_retry_note_shows_correct_attempt_number(self):
        task = _task("Task X")
        result = build_task_prompt(task, [], attempt=3)
        assert "retry attempt 3" in result

    def test_current_task_title_always_present(self):
        task = _task("My Special Task")
        result = build_task_prompt(task, [], attempt=1)
        assert "## Current task: My Special Task" in result

    def test_task_body_always_present(self):
        task = _task("T", body="Write some code here.")
        result = build_task_prompt(task, [], attempt=1)
        assert "Write some code here." in result

    def test_completed_and_retry_together(self):
        done = [_task("Old Task")]
        task = _task("New Task", "New body.")
        result = build_task_prompt(task, done, attempt=2)
        assert "Previously completed" in result
        assert "- Old Task" in result
        assert "retry attempt 2" in result
        assert "## Current task: New Task" in result
        assert "New body." in result

    def test_empty_completed_list(self):
        task = _task("Solo Task")
        result = build_task_prompt(task, [], attempt=1)
        assert "Previously completed" not in result


# ---------------------------------------------------------------------------
# build_initial_messages
# ---------------------------------------------------------------------------

class TestBuildInitialMessages:
    def test_returns_two_messages(self):
        task = _task("Task A")
        msgs = build_initial_messages(task, [], system_prompt="You are an agent.", attempt=1)
        assert len(msgs) == 2

    def test_first_message_is_system(self):
        task = _task("Task A")
        msgs = build_initial_messages(task, [], system_prompt="System prompt here.", attempt=1)
        assert msgs[0]["role"] == "system"
        assert msgs[0]["content"] == "System prompt here."

    def test_second_message_is_user(self):
        task = _task("Task A")
        msgs = build_initial_messages(task, [], system_prompt="SP", attempt=1)
        assert msgs[1]["role"] == "user"

    def test_user_message_content_matches_build_task_prompt(self):
        task = _task("Task B", "Body text.")
        completed = [_task("Task A")]
        system = "Be helpful."
        attempt = 2
        msgs = build_initial_messages(task, completed, system_prompt=system, attempt=attempt)
        expected_user = build_task_prompt(task, completed, attempt)
        assert msgs[1]["content"] == expected_user

    def test_system_prompt_not_in_user_message(self):
        task = _task("Task")
        system = "UNIQUE_SYSTEM_CONTENT_XYZ"
        msgs = build_initial_messages(task, [], system_prompt=system, attempt=1)
        assert system not in msgs[1]["content"]

    def test_task_title_in_user_message(self):
        task = _task("My Task Title")
        msgs = build_initial_messages(task, [], system_prompt="SP", attempt=1)
        assert "My Task Title" in msgs[1]["content"]
