"""Tests for ReactLoop: run(), helpers, context truncation."""

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
from ollama_symphony import (
    OllamaClient,
    OllamaConnectionError,
    ReactLoop,
    Task,
    ToolCall,
    ToolExecutor,
    ToolResult,
    WorkflowConfig,
    build_task_prompt,
    build_initial_messages,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _config(**kwargs) -> WorkflowConfig:
    return WorkflowConfig(
        max_iterations=kwargs.pop("max_iterations", 5),
        ollama_context_window=kwargs.pop("ollama_context_window", 8192),
        **kwargs,
    )


def _make_loop(max_iterations=5, **cfg_kwargs):
    config = _config(max_iterations=max_iterations, **cfg_kwargs)
    client = MagicMock(spec=OllamaClient)
    executor = MagicMock(spec=ToolExecutor)
    return ReactLoop(client, executor, config), client, executor


def _response(content="", tool_calls=None):
    msg = {"role": "assistant", "content": content}
    if tool_calls is not None:
        msg["tool_calls"] = tool_calls
    return {"message": msg}


def _tc_dict(name, **args):
    return {"function": {"name": name, "arguments": args}}


# ---------------------------------------------------------------------------
# run() — task_complete path
# ---------------------------------------------------------------------------

def test_run_task_complete_first_iteration():
    loop, client, executor = _make_loop()
    client.chat.return_value = _response(tool_calls=[_tc_dict("task_complete", summary="done")])

    success, summary = loop.run([{"role": "user", "content": "go"}], tools=[])

    assert success is True
    assert summary == "done"
    client.chat.assert_called_once()
    executor.execute.assert_not_called()


def test_run_task_complete_after_tool_calls():
    loop, client, executor = _make_loop()
    executor.execute.return_value = ToolResult(
        tool_name="run_shell", success=True, output="ok", exit_code=0
    )
    client.chat.side_effect = [
        _response(tool_calls=[_tc_dict("run_shell", command="ls")]),
        _response(tool_calls=[_tc_dict("task_complete", summary="all done")]),
    ]

    success, summary = loop.run([{"role": "user", "content": "go"}], tools=[])

    assert success is True
    assert summary == "all done"
    assert client.chat.call_count == 2
    executor.execute.assert_called_once()


def test_run_task_complete_summary_empty_string():
    loop, client, executor = _make_loop()
    client.chat.return_value = _response(tool_calls=[_tc_dict("task_complete", summary="")])

    success, summary = loop.run([], tools=[])

    assert success is True
    assert summary == ""


# ---------------------------------------------------------------------------
# run() — max_iterations exhausted
# ---------------------------------------------------------------------------

def test_run_max_iterations_reached():
    loop, client, executor = _make_loop(max_iterations=3)
    executor.execute.return_value = ToolResult(
        tool_name="run_shell", success=True, output="ok"
    )
    client.chat.return_value = _response(tool_calls=[_tc_dict("run_shell", command="ls")])

    success, msg = loop.run([{"role": "user", "content": "go"}], tools=[])

    assert success is False
    assert "max_iterations" in msg
    assert "3" in msg
    assert client.chat.call_count == 3


# ---------------------------------------------------------------------------
# run() — connection error
# ---------------------------------------------------------------------------

def test_run_ollama_connection_error():
    loop, client, executor = _make_loop()
    client.chat.side_effect = OllamaConnectionError("all hosts down")

    success, msg = loop.run([], tools=[])

    assert success is False
    assert "Ollama unreachable" in msg


def test_run_generic_exception():
    loop, client, executor = _make_loop()
    client.chat.side_effect = RuntimeError("boom")

    success, msg = loop.run([], tools=[])

    assert success is False
    assert "Ollama error" in msg


# ---------------------------------------------------------------------------
# run() — no tool calls (implicit completion)
# ---------------------------------------------------------------------------

def test_run_no_tool_calls_returns_content():
    loop, client, executor = _make_loop()
    client.chat.return_value = _response(content="Here is the answer.")

    success, summary = loop.run([], tools=[])

    assert success is True
    assert summary == "Here is the answer."


def test_run_no_tool_calls_empty_content():
    loop, client, executor = _make_loop()
    client.chat.return_value = _response(content="")

    success, summary = loop.run([], tools=[])

    assert success is True
    assert summary == "completed (no summary)"


def test_run_no_tool_calls_truncates_long_content():
    loop, client, executor = _make_loop()
    long_text = "x" * 500
    client.chat.return_value = _response(content=long_text)

    success, summary = loop.run([], tools=[])

    assert success is True
    assert len(summary) == 300


# ---------------------------------------------------------------------------
# run() — message history grows correctly
# ---------------------------------------------------------------------------

def test_run_appends_tool_results_to_history():
    loop, client, executor = _make_loop()
    executor.execute.return_value = ToolResult(
        tool_name="read_file", success=True, output="file content"
    )
    client.chat.side_effect = [
        _response(tool_calls=[_tc_dict("read_file", path="foo.py")]),
        _response(tool_calls=[_tc_dict("task_complete", summary="done")]),
    ]

    loop.run([{"role": "user", "content": "start"}], tools=[])

    assert client.chat.call_count == 2
    # Second call should have assistant + tool messages appended
    second_call_args = client.chat.call_args_list[1]
    second_call_msgs = second_call_args[0][0]  # first positional arg
    roles = [m["role"] for m in second_call_msgs]
    assert "assistant" in roles
    assert "tool" in roles


# ---------------------------------------------------------------------------
# _extract_message
# ---------------------------------------------------------------------------

def test_extract_message_dict_response():
    loop, *_ = _make_loop()
    msg = {"role": "assistant", "content": "hi"}
    assert loop._extract_message({"message": msg}) == msg


def test_extract_message_object_with_dict():
    loop, *_ = _make_loop()
    inner = {"role": "assistant", "content": "hello"}

    class FakeResponse:
        message = inner

    assert loop._extract_message({"message": inner}) == inner


def test_extract_message_object_style():
    loop, *_ = _make_loop()

    msg = MagicMock()
    msg.__dict__ = {"role": "assistant", "content": "hi"}

    result = loop._extract_message({"message": msg})
    assert result["role"] == "assistant"
    assert result["content"] == "hi"


def test_extract_message_missing_returns_empty():
    loop, *_ = _make_loop()
    assert loop._extract_message({}) == {}
    assert loop._extract_message({"message": {}}) == {}


# ---------------------------------------------------------------------------
# _extract_tool_calls
# ---------------------------------------------------------------------------

def test_extract_tool_calls_present():
    loop, *_ = _make_loop()
    tcs = [{"function": {"name": "foo"}}]
    assert loop._extract_tool_calls({"tool_calls": tcs}) == tcs


def test_extract_tool_calls_none_returns_empty():
    loop, *_ = _make_loop()
    assert loop._extract_tool_calls({}) == []
    assert loop._extract_tool_calls({"tool_calls": None}) == []


def test_extract_tool_calls_non_list_returns_empty():
    loop, *_ = _make_loop()
    assert loop._extract_tool_calls({"tool_calls": "bad"}) == []


# ---------------------------------------------------------------------------
# _parse_tool_call
# ---------------------------------------------------------------------------

def test_parse_tool_call_dict_style():
    loop, *_ = _make_loop()
    raw = {"function": {"name": "run_shell", "arguments": {"command": "ls"}}}
    tc = loop._parse_tool_call(raw)
    assert tc.name == "run_shell"
    assert tc.arguments == {"command": "ls"}


def test_parse_tool_call_object_style():
    loop, *_ = _make_loop()

    class Fn:
        name = "write_file"
        arguments = {"path": "x.py", "content": "print()"}

    class Raw:
        function = Fn()

    tc = loop._parse_tool_call(Raw())
    assert tc.name == "write_file"
    assert tc.arguments == {"path": "x.py", "content": "print()"}


def test_parse_tool_call_json_string_args():
    loop, *_ = _make_loop()
    args_str = json.dumps({"command": "echo hi"})
    raw = {"function": {"name": "run_shell", "arguments": args_str}}
    tc = loop._parse_tool_call(raw)
    assert tc.arguments == {"command": "echo hi"}


def test_parse_tool_call_invalid_json_args():
    loop, *_ = _make_loop()
    raw = {"function": {"name": "run_shell", "arguments": "{not json}"}}
    tc = loop._parse_tool_call(raw)
    assert tc.arguments == {}


def test_parse_tool_call_none_args():
    loop, *_ = _make_loop()
    raw = {"function": {"name": "run_shell", "arguments": None}}
    tc = loop._parse_tool_call(raw)
    assert tc.arguments == {}


# ---------------------------------------------------------------------------
# _format_tool_result
# ---------------------------------------------------------------------------

def test_format_tool_result_success():
    loop, *_ = _make_loop()
    result = ToolResult(tool_name="run_shell", success=True, output="ok")
    formatted = loop._format_tool_result(result)
    assert "[run_shell] success" in formatted
    assert "ok" in formatted


def test_format_tool_result_error():
    loop, *_ = _make_loop()
    result = ToolResult(tool_name="read_file", success=False, output="not found")
    formatted = loop._format_tool_result(result)
    assert "[read_file] error" in formatted
    assert "not found" in formatted


# ---------------------------------------------------------------------------
# _estimate_tokens / _should_truncate
# ---------------------------------------------------------------------------

def test_estimate_tokens_proportional_to_size():
    loop, *_ = _make_loop()
    short_msgs = [{"role": "user", "content": "hi"}]
    long_msgs = [{"role": "user", "content": "x" * 1000}]
    assert loop._estimate_tokens(long_msgs) > loop._estimate_tokens(short_msgs)


def test_should_truncate_false_when_small():
    loop, *_ = _make_loop(ollama_context_window=8192)
    msgs = [{"role": "user", "content": "hi"}]
    assert loop._should_truncate(msgs) is False


def test_should_truncate_true_when_large():
    loop, *_ = _make_loop(ollama_context_window=100)
    # 85% of 100 = 85 tokens; each token ~4 chars → need >340 chars
    msgs = [{"role": "user", "content": "x" * 500}]
    assert loop._should_truncate(msgs) is True


# ---------------------------------------------------------------------------
# _truncate_context
# ---------------------------------------------------------------------------

def test_truncate_context_preserves_head_and_tail():
    loop, *_ = _make_loop()
    msgs = [{"role": "system", "content": "sys"}] + \
           [{"role": "user", "content": "first"}] + \
           [{"role": "assistant", "content": f"msg{i}"} for i in range(10)]

    truncated = loop._truncate_context(msgs)

    assert truncated[0]["content"] == "sys"
    assert truncated[1]["content"] == "first"
    # tail = last 6 of original
    assert truncated[2:] == msgs[-6:]


def test_truncate_context_short_list_unchanged():
    loop, *_ = _make_loop()
    msgs = [{"role": "user", "content": f"msg{i}"} for i in range(8)]
    assert loop._truncate_context(msgs) == msgs


def test_truncate_context_total_length():
    loop, *_ = _make_loop()
    msgs = [{"role": "system", "content": "sys"}] + \
           [{"role": "user", "content": "first"}] + \
           [{"role": "assistant", "content": f"extra{i}"} for i in range(20)]

    truncated = loop._truncate_context(msgs)
    assert len(truncated) == 8  # head(2) + tail(6)


# ---------------------------------------------------------------------------
# run() — context truncation triggered
# ---------------------------------------------------------------------------

def test_run_triggers_truncation_when_context_large():
    loop, client, executor = _make_loop(ollama_context_window=1)
    # context_window=1 → threshold = 0 tokens → always truncate
    client.chat.return_value = _response(tool_calls=[_tc_dict("task_complete", summary="ok")])

    # Need 9+ messages to trigger truncation logic
    msgs = [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}] + \
           [{"role": "assistant", "content": f"x{i}"} for i in range(8)]

    success, summary = loop.run(msgs, tools=[])

    assert success is True


# ---------------------------------------------------------------------------
# run() — unknown tool continues loop
# ---------------------------------------------------------------------------

def test_run_unknown_tool_continues(tmp_path):
    config = WorkflowConfig(working_dir=str(tmp_path), max_iterations=5)
    client = MagicMock(spec=OllamaClient)
    client.chat.side_effect = [
        _response(tool_calls=[_tc_dict("nonexistent_tool", arg="x")]),
        _response(tool_calls=[_tc_dict("task_complete", summary="recovered")]),
    ]
    executor = ToolExecutor(config)
    loop = ReactLoop(client, executor, config)

    messages = [{"role": "system", "content": "sys"}, {"role": "user", "content": "task"}]
    success, summary = loop.run(messages, [])

    assert success is True
    assert "recovered" in summary


# ---------------------------------------------------------------------------
# build_task_prompt
# ---------------------------------------------------------------------------

def _make_task(title="Test task", body="Do something") -> Task:
    return Task(index=0, title=title, body=body)


def test_build_task_prompt_first_attempt():
    task = _make_task(title="My task", body="Do something")
    prompt = build_task_prompt(task, [], attempt=1)
    assert "My task" in prompt
    assert "Do something" in prompt
    assert "retry" not in prompt.lower()


def test_build_task_prompt_with_completed():
    completed = [_make_task(title="Previous task")]
    task = _make_task(title="Current task", body="Body")
    prompt = build_task_prompt(task, completed, attempt=1)
    assert "Previous task" in prompt
    assert "Current task" in prompt


def test_build_task_prompt_retry():
    task = _make_task()
    prompt = build_task_prompt(task, [], attempt=2)
    assert "retry attempt 2" in prompt.lower() or "attempt 2" in prompt
