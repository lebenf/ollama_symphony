# Task 3 — OllamaClient, ReAct Loop, Runner e CLI

### Prerequisiti

I Task 1 e 2 devono essere completati:
- `ollama_symphony.py` contiene dataclass, parsing, `StateStore`, tool registry, `ToolExecutor`
- `pytest tests/test_parsing.py tests/test_tools.py` → tutti PASSED

---

### Obiettivo

Completare `ollama_symphony.py` con:
1. `OllamaClient` — comunicazione con Ollama API, round-robin multi-host
2. `ReactLoop` — loop Reason+Act per un singolo task
3. `OllamaSymphonyRunner` — orchestrazione dei task con retry e git
4. CLI completa con tutti gli argomenti
5. Test suite completa senza server Ollama attivo

---

### Cosa aggiungere a `ollama_symphony.py`

Aggiungi queste sezioni **prima** del `main()`, nell'ordine indicato.

---

## 1. Eccezione custom

```python
# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class OllamaConnectionError(Exception):
    """Raised when all configured Ollama hosts are unreachable."""
```

---

## 2. `OllamaClient`

```python
# ---------------------------------------------------------------------------
# Ollama client with round-robin host rotation
# ---------------------------------------------------------------------------

class OllamaClient:
    """
    Wraps the ollama Python client with round-robin across multiple hosts.
    Falls back to the next host on ConnectionError or timeout.
    """

    def __init__(self, config: WorkflowConfig):
        self._hosts = config.ollama_hosts
        self._model = config.ollama_model
        self._config = config
        self._current_host_idx = 0
        self._log = logging.getLogger("ollama_symphony.client")

    def _next_host(self) -> str:
        host = self._hosts[self._current_host_idx]
        self._current_host_idx = (self._current_host_idx + 1) % len(self._hosts)
        return host

    def chat(self, messages: list[dict], tools: list[dict]) -> dict:
        """
        Call Ollama /api/chat with tool calling.
        Tries each host in rotation; raises OllamaConnectionError if all fail.
        Returns the raw response dict from the ollama library.
        """
        import ollama

        last_exc: Exception | None = None
        for _ in range(len(self._hosts)):
            host = self._next_host()
            try:
                self._log.debug("ollama_request host=%s model=%s", host, self._model)
                client = ollama.Client(host=host)
                response = client.chat(
                    model=self._model,
                    messages=messages,
                    tools=tools,
                    stream=False,
                    options={
                        "temperature": self._config.ollama_temperature,
                        "num_ctx": self._config.ollama_num_ctx,
                    },
                )
                # ollama library returns an object; convert to dict for uniform handling
                if hasattr(response, "model_dump"):
                    return response.model_dump()
                if hasattr(response, "__dict__"):
                    return dict(response)
                return response  # type: ignore
            except Exception as exc:
                self._log.warning("ollama_host_error host=%s error=%r", host, str(exc))
                last_exc = exc

        raise OllamaConnectionError(
            f"All Ollama hosts unreachable: {self._hosts}. Last error: {last_exc}"
        )

    def list_models(self) -> list[str]:
        """List models available on the current host."""
        import ollama
        host = self._hosts[self._current_host_idx]
        try:
            client = ollama.Client(host=host)
            response = client.list()
            models = response.get("models", []) if isinstance(response, dict) else []
            return [m.get("name", "") for m in models if m.get("name")]
        except Exception as exc:
            self._log.warning("list_models failed host=%s error=%r", host, str(exc))
            return []
```

---

## 3. `ReactLoop`

```python
# ---------------------------------------------------------------------------
# ReAct loop
# ---------------------------------------------------------------------------

class ReactLoop:
    """
    Implements the Reason+Act loop for a single task.

    Sends messages to Ollama, intercepts tool_calls, executes them via
    ToolExecutor, and feeds results back until task_complete is called
    or max_iterations is reached.
    """

    def __init__(
        self,
        client: OllamaClient,
        executor: ToolExecutor,
        config: WorkflowConfig,
    ):
        self._client = client
        self._executor = executor
        self._config = config
        self._log = logging.getLogger("ollama_symphony.react")

    def run(self, messages: list[dict], tools: list[dict]) -> tuple[bool, str]:
        """
        Execute the ReAct loop.
        Returns (success: bool, summary_or_error: str).
        """
        for iteration in range(1, self._config.max_iterations + 1):
            # Context window check before sending
            if self._should_truncate(messages):
                messages = self._truncate_context(messages)
                self._log.warning("context_truncated iteration=%d", iteration)

            try:
                response = self._client.chat(messages, tools)
            except OllamaConnectionError as exc:
                return False, f"Ollama unreachable: {exc}"
            except Exception as exc:
                return False, f"Ollama error: {exc}"

            # Extract message from response
            msg = self._extract_message(response)
            tool_calls = self._extract_tool_calls(msg)

            self._log.debug(
                "react_iteration=%d tool_calls=%d", iteration, len(tool_calls)
            )

            if not tool_calls:
                # No tool calls: model gave a plain text response.
                # Treat as implicit completion with a warning.
                content = msg.get("content", "") if isinstance(msg, dict) else str(msg)
                self._log.warning(
                    "react_no_tool_calls iteration=%d — treating as implicit completion",
                    iteration,
                )
                return True, content[:300] if content else "completed (no summary)"

            # Process each tool call
            task_done = False
            task_summary = ""
            for tc_raw in tool_calls:
                tool_call = self._parse_tool_call(tc_raw)

                if tool_call.name == "task_complete":
                    task_summary = tool_call.arguments.get("summary", "")
                    self._log.info("task_complete summary=%r", task_summary)
                    task_done = True
                    # Still add the tool result to messages for protocol completeness
                    messages = self._append_assistant_turn(messages, msg)
                    messages = self._append_tool_result(
                        messages,
                        tool_call.name,
                        ToolResult(tool_name="task_complete", success=True, output="acknowledged"),
                    )
                    break  # stop processing further tool calls in this iteration

                # Execute the tool
                result = self._executor.execute(tool_call)
                self._log.debug(
                    "tool_exec name=%s success=%s exit_code=%s",
                    tool_call.name,
                    result.success,
                    result.exit_code,
                )

                # Append assistant message + tool result to history
                messages = self._append_assistant_turn(messages, msg)
                messages = self._append_tool_result(messages, tool_call.name, result)

            if task_done:
                return True, task_summary

        return False, f"max_iterations ({self._config.max_iterations}) reached without task_complete"

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _extract_message(self, response: dict) -> dict:
        """Extract the assistant message dict from the Ollama response."""
        if isinstance(response, dict):
            msg = response.get("message", {})
            if isinstance(msg, dict):
                return msg
            # Handle object-style message
            if hasattr(msg, "__dict__"):
                return vars(msg)
        return {}

    def _extract_tool_calls(self, message: dict) -> list:
        """Extract tool_calls list from the message dict."""
        tc = message.get("tool_calls")
        if tc is None:
            return []
        if isinstance(tc, list):
            return tc
        return []

    def _parse_tool_call(self, tc_raw) -> ToolCall:
        """Normalize a raw tool_call entry (dict or object) into ToolCall."""
        if isinstance(tc_raw, dict):
            fn = tc_raw.get("function", {})
            name = fn.get("name", "") if isinstance(fn, dict) else getattr(fn, "name", "")
            args = fn.get("arguments", {}) if isinstance(fn, dict) else getattr(fn, "arguments", {})
        else:
            # object-style
            fn = getattr(tc_raw, "function", None)
            name = getattr(fn, "name", "") if fn else ""
            args = getattr(fn, "arguments", {}) if fn else {}

        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                args = {}

        return ToolCall(name=name, arguments=args or {})

    def _append_assistant_turn(self, messages: list[dict], msg: dict) -> list[dict]:
        """Append the assistant message to the conversation history."""
        return messages + [{"role": "assistant", **msg}]

    def _append_tool_result(
        self, messages: list[dict], tool_name: str, result: ToolResult
    ) -> list[dict]:
        """Append a tool result message to the conversation history."""
        return messages + [{"role": "tool", "content": self._format_tool_result(result)}]

    def _format_tool_result(self, result: ToolResult) -> str:
        status = "success" if result.success else "error"
        return f"[{result.tool_name}] {status}\n{result.output}"

    def _estimate_tokens(self, messages: list[dict]) -> int:
        """Rough token estimate: characters / 4."""
        return len(json.dumps(messages, ensure_ascii=False)) // 4

    def _should_truncate(self, messages: list[dict]) -> bool:
        return self._estimate_tokens(messages) > int(self._config.ollama_context_window * 0.85)

    def _truncate_context(self, messages: list[dict]) -> list[dict]:
        """
        Keep: system message (index 0) + first user message (index 1) + last 6 messages.
        """
        if len(messages) <= 8:
            return messages
        head = messages[:2]   # system + initial user
        tail = messages[-6:]  # last 6 exchanges
        return head + tail
```

---

## 4. Prompt assembly

```python
# ---------------------------------------------------------------------------
# Prompt assembly
# ---------------------------------------------------------------------------

def build_task_prompt(
    task: "Task",
    completed_tasks: list["Task"],
    attempt: int,
) -> str:
    """Build the user-facing prompt for a single task."""
    parts: list[str] = []

    if completed_tasks:
        parts.append("## Previously completed tasks (already committed):")
        for t in completed_tasks:
            parts.append(f"- {t.title}")
        parts.append("Do not redo these. Focus only on the current task.\n")

    if attempt > 1:
        parts.append(
            f"NOTE: This is retry attempt {attempt}. Previous attempt failed. "
            "Review what might have gone wrong and try a different approach.\n"
        )

    parts.append(f"## Current task: {task.title}\n")
    parts.append(task.body)

    return "\n".join(parts)


def build_initial_messages(
    task: "Task",
    completed_tasks: list["Task"],
    system_prompt: str,
    attempt: int,
) -> list[dict]:
    """Build the initial messages list for the ReAct loop."""
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": build_task_prompt(task, completed_tasks, attempt)},
    ]
```

---

## 5. Git helpers

Copia da `symphony.py` (già testata, non modificare):

```python
# ---------------------------------------------------------------------------
# Git helpers
# ---------------------------------------------------------------------------

def git_commit(message: str, logger: logging.Logger) -> bool:
    """Stage all changes and create a commit. Returns True on success."""
    try:
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True, text=True, check=True,
        )
        if not status.stdout.strip():
            logger.info("git: nothing to commit, skipping")
            return True
        subprocess.run(["git", "add", "-A"], check=True, capture_output=True)
        subprocess.run(
            ["git", "commit", "-m", message],
            check=True, capture_output=True,
        )
        logger.info("git: committed — %s", message)
        return True
    except subprocess.CalledProcessError as exc:
        stderr = exc.stderr.decode(errors="replace").strip() if exc.stderr else ""
        logger.error("git commit failed: %s", stderr or exc)
        return False
```

---

## 6. `OllamaSymphonyRunner`

```python
# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

class OllamaSymphonyRunner:
    def __init__(
        self,
        tasks: list[Task],
        config: WorkflowConfig,
        state: StateStore,
        dry_run: bool = False,
    ):
        self.tasks = tasks
        self.config = config
        self.state = state
        self.dry_run = dry_run
        self.log = logging.getLogger("ollama_symphony")
        self._client = OllamaClient(config)
        self._executor = ToolExecutor(config)

    def run(self) -> bool:
        total = len(self.tasks)
        self.log.info(
            "Symphony start — %d task(s) to process  hosts=%s model=%s",
            total, self.config.ollama_hosts, self.config.ollama_model,
        )

        completed_so_far: list[Task] = []

        for task in self.tasks:
            if self.state.is_completed(task.slug):
                self.log.info(
                    "task_status=skipped index=%d title=%r (already completed)",
                    task.index + 1, task.title,
                )
                completed_so_far.append(task)
                continue

            success = self._run_task_with_retries(task, completed_so_far)

            if success:
                completed_so_far.append(task)
            else:
                self.log.error(
                    "task_status=fatal index=%d title=%r — stopping run",
                    task.index + 1, task.title,
                )
                return False

        self.log.info("Symphony complete — all %d task(s) succeeded", total)
        return True

    def _run_task_with_retries(self, task: Task, completed_so_far: list[Task]) -> bool:
        max_attempts = self.config.max_retries + 1
        task_state = self.state.get(task.slug) or TaskState(status="pending")

        for attempt in range(1, max_attempts + 1):
            self.log.info(
                "task_start index=%d title=%r attempt=%d/%d",
                task.index + 1, task.title, attempt, max_attempts,
            )

            success, summary_or_error = self._run_single_attempt(
                task, completed_so_far, attempt
            )
            task_state.attempts = attempt

            if success:
                if self.config.commit_after_each_task:
                    commit_msg = self.config.commit_message_template.format(
                        task_title=task.title,
                        task_index=task.index + 1,
                    )
                    if not self.dry_run:
                        committed = git_commit(commit_msg, self.log)
                        if not committed:
                            self.log.warning(
                                "git commit failed for task %r — continuing", task.title
                            )

                task_state.status = "completed"
                task_state.error = None
                task_state.completed_at = _now_iso()
                self.state.set(task.slug, task_state)

                self.log.info(
                    "task_status=completed index=%d title=%r attempt=%d",
                    task.index + 1, task.title, attempt,
                )
                return True

            task_state.status = "failed"
            task_state.error = summary_or_error[:500]
            self.state.set(task.slug, task_state)

            self.log.warning(
                "task_status=failed index=%d title=%r attempt=%d error=%r",
                task.index + 1, task.title, attempt, summary_or_error[:200],
            )

            if attempt < max_attempts:
                self.log.info("Retrying in %.0fs…", self.config.retry_delay_s)
                if not self.dry_run:
                    time.sleep(self.config.retry_delay_s)

        return False

    def _run_single_attempt(
        self, task: Task, completed_so_far: list[Task], attempt: int
    ) -> tuple[bool, str]:
        if self.dry_run:
            self.log.info("[dry-run] Would run task %r via Ollama ReAct loop", task.title)
            return True, "[dry-run]"

        messages = build_initial_messages(
            task, completed_so_far, self.config.system_prompt, attempt
        )
        tools = build_tool_schemas(self.config)
        loop = ReactLoop(self._client, self._executor, self.config)
        return loop.run(messages, tools)
```

---

## 7. CLI completa — sostituisce il placeholder `main()`

Rimuovi il placeholder `main()` creato nel Task 1 e sostituiscilo con:

```python
# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Ollama Symphony — run tasks sequentially via local LLM models.",
    )
    parser.add_argument("--tasks", default="TASKS.md", metavar="FILE",
                        help="Path to TASKS.md (default: TASKS.md)")
    parser.add_argument("--workflow", default="WORKFLOW.md", metavar="FILE",
                        help="Path to WORKFLOW.md (default: WORKFLOW.md)")
    parser.add_argument("--state", default="TASKS.state.json", metavar="FILE",
                        help="Path to state file (default: TASKS.state.json)")
    parser.add_argument("--reset", action="store_true",
                        help="Ignore saved state and restart from task 1")
    parser.add_argument("--dry-run", action="store_true",
                        help="Parse and log everything but do not invoke Ollama or git")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Enable debug logging")
    parser.add_argument("--list-models", action="store_true",
                        help="List models available on all configured Ollama hosts, then exit")
    parser.add_argument("--check", action="store_true",
                        help="Validate config and Ollama connectivity, then exit (0=ok, 1=error)")
    args = parser.parse_args()

    setup_logging(args.verbose)
    log = logging.getLogger("ollama_symphony")

    # --- Load workflow config ---
    config = WorkflowConfig()
    workflow_path = Path(args.workflow)
    if workflow_path.exists():
        try:
            config = parse_workflow(workflow_path)
            log.info("Loaded workflow config from %s", workflow_path)
        except Exception as exc:
            log.error("Failed to parse %s: %s", workflow_path, exc)
            sys.exit(1)
    else:
        log.info("No %s found — using defaults", workflow_path)

    # --- --list-models ---
    if args.list_models:
        client = OllamaClient(config)
        for host in config.ollama_hosts:
            config_copy = WorkflowConfig(**{
                **config.__dict__,
                "ollama_hosts": [host],
            })
            c = OllamaClient(config_copy)
            models = c.list_models()
            print(f"{host}: {', '.join(models) if models else '(no models or unreachable)'}")
        sys.exit(0)

    # --- --check ---
    if args.check:
        all_ok = True
        for host in config.ollama_hosts:
            config_copy = WorkflowConfig(**{**config.__dict__, "ollama_hosts": [host]})
            c = OllamaClient(config_copy)
            models = c.list_models()
            if models:
                log.info("check host=%s status=ok models=%d", host, len(models))
            else:
                log.error("check host=%s status=unreachable_or_empty", host)
                all_ok = False
        sys.exit(0 if all_ok else 1)

    # --- Load tasks ---
    tasks_path = Path(args.tasks)
    if not tasks_path.exists():
        log.error("Tasks file not found: %s", tasks_path)
        sys.exit(1)

    try:
        tasks = parse_tasks(tasks_path)
    except ValueError as exc:
        log.error("%s", exc)
        sys.exit(1)

    log.info("Loaded %d task(s) from %s", len(tasks), tasks_path)

    # --- State ---
    state_path = Path(args.state)
    if args.reset and state_path.exists():
        state_path.unlink()
        log.info("State file reset")

    state = StateStore(state_path)
    state.init_run(str(tasks_path))

    # --- Run ---
    runner = OllamaSymphonyRunner(tasks, config, state, dry_run=args.dry_run)
    success = runner.run()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
```

---

## 8. Test suite

### `tests/test_react_loop.py`

```python
"""Tests for ReactLoop. All tests use mocked OllamaClient — no real Ollama server needed."""

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
from ollama_symphony import (
    WorkflowConfig, ToolCall, ToolResult,
    OllamaClient, ToolExecutor, ReactLoop,
    build_task_prompt, build_initial_messages,
    Task,
)


def make_config(**kwargs) -> WorkflowConfig:
    defaults = dict(max_iterations=5, ollama_context_window=8192)
    defaults.update(kwargs)
    return WorkflowConfig(**defaults)


def make_response(tool_calls=None, content="") -> dict:
    """Build a minimal Ollama response dict."""
    msg: dict = {"role": "assistant", "content": content}
    if tool_calls:
        msg["tool_calls"] = tool_calls
    return {"message": msg}


def make_tool_call(name: str, **arguments) -> dict:
    return {"function": {"name": name, "arguments": arguments}}


def make_task(title="Test task", body="Do something") -> Task:
    return Task(index=0, title=title, body=body)


# ---------------------------------------------------------------------------
# task_complete on first iteration
# ---------------------------------------------------------------------------

def test_react_task_complete_first_iteration(tmp_path):
    config = make_config(working_dir=str(tmp_path))
    client = MagicMock(spec=OllamaClient)
    client.chat.return_value = make_response(
        tool_calls=[make_tool_call("task_complete", summary="all done")]
    )
    executor = ToolExecutor(config)
    loop = ReactLoop(client, executor, config)

    messages = [{"role": "system", "content": "sys"}, {"role": "user", "content": "task"}]
    success, summary = loop.run(messages, [])

    assert success is True
    assert "all done" in summary
    assert client.chat.call_count == 1


# ---------------------------------------------------------------------------
# tool call then task_complete
# ---------------------------------------------------------------------------

def test_react_tool_call_then_complete(tmp_path):
    config = make_config(working_dir=str(tmp_path))
    client = MagicMock(spec=OllamaClient)

    # First call: run_shell
    # Second call: task_complete
    client.chat.side_effect = [
        make_response(tool_calls=[make_tool_call("run_shell", command="echo hi")]),
        make_response(tool_calls=[make_tool_call("task_complete", summary="done after shell")]),
    ]

    executor = ToolExecutor(config)
    loop = ReactLoop(client, executor, config)

    messages = [{"role": "system", "content": "sys"}, {"role": "user", "content": "task"}]
    success, summary = loop.run(messages, [])

    assert success is True
    assert "done after shell" in summary
    assert client.chat.call_count == 2


# ---------------------------------------------------------------------------
# max_iterations exceeded
# ---------------------------------------------------------------------------

def test_react_max_iterations_exceeded(tmp_path):
    config = make_config(working_dir=str(tmp_path), max_iterations=3)
    client = MagicMock(spec=OllamaClient)
    # Always returns a tool call that is NOT task_complete
    client.chat.return_value = make_response(
        tool_calls=[make_tool_call("run_shell", command="echo loop")]
    )

    executor = ToolExecutor(config)
    loop = ReactLoop(client, executor, config)

    messages = [{"role": "system", "content": "sys"}, {"role": "user", "content": "task"}]
    success, error = loop.run(messages, [])

    assert success is False
    assert "max_iterations" in error
    assert client.chat.call_count == 3


# ---------------------------------------------------------------------------
# unknown tool — loop continues, tool_complete eventually called
# ---------------------------------------------------------------------------

def test_react_unknown_tool_continues(tmp_path):
    config = make_config(working_dir=str(tmp_path), max_iterations=5)
    client = MagicMock(spec=OllamaClient)
    client.chat.side_effect = [
        make_response(tool_calls=[make_tool_call("nonexistent_tool", arg="x")]),
        make_response(tool_calls=[make_tool_call("task_complete", summary="recovered")]),
    ]

    executor = ToolExecutor(config)
    loop = ReactLoop(client, executor, config)

    messages = [{"role": "system", "content": "sys"}, {"role": "user", "content": "task"}]
    success, summary = loop.run(messages, [])

    assert success is True
    assert "recovered" in summary


# ---------------------------------------------------------------------------
# context truncation
# ---------------------------------------------------------------------------

def test_truncate_context_keeps_head_and_tail(tmp_path):
    config = make_config(working_dir=str(tmp_path))
    loop = ReactLoop(MagicMock(), ToolExecutor(config), config)

    # Build 20 messages: system, user, then 18 tool exchanges
    messages = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "initial task"},
    ]
    for i in range(18):
        messages.append({"role": "assistant", "content": f"step {i}"})

    truncated = loop._truncate_context(messages)

    assert truncated[0]["role"] == "system"
    assert truncated[1]["role"] == "user"
    # last 6 original messages should be preserved
    assert truncated[-1] == messages[-1]
    assert truncated[-6] == messages[-6]
    assert len(truncated) == 2 + 6  # head + tail


def test_truncate_context_short_messages_unchanged(tmp_path):
    config = make_config(working_dir=str(tmp_path))
    loop = ReactLoop(MagicMock(), ToolExecutor(config), config)

    messages = [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}]
    assert loop._truncate_context(messages) == messages


# ---------------------------------------------------------------------------
# build_task_prompt
# ---------------------------------------------------------------------------

def test_build_task_prompt_first_attempt():
    task = make_task(title="My task", body="Do something")
    prompt = build_task_prompt(task, [], attempt=1)
    assert "My task" in prompt
    assert "Do something" in prompt
    assert "retry" not in prompt.lower()


def test_build_task_prompt_with_completed():
    completed = [make_task(title="Previous task")]
    task = make_task(title="Current task", body="Body")
    prompt = build_task_prompt(task, completed, attempt=1)
    assert "Previous task" in prompt
    assert "Current task" in prompt


def test_build_task_prompt_retry():
    task = make_task()
    prompt = build_task_prompt(task, [], attempt=2)
    assert "retry attempt 2" in prompt.lower() or "attempt 2" in prompt


# ---------------------------------------------------------------------------
# OllamaConnectionError propagated
# ---------------------------------------------------------------------------

def test_react_connection_error_returns_failure(tmp_path):
    from ollama_symphony import OllamaConnectionError
    config = make_config(working_dir=str(tmp_path))
    client = MagicMock(spec=OllamaClient)
    client.chat.side_effect = OllamaConnectionError("all hosts down")

    loop = ReactLoop(client, ToolExecutor(config), config)
    messages = [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}]
    success, error = loop.run(messages, [])

    assert success is False
    assert "unreachable" in error.lower() or "all hosts" in error.lower()
```

### `tests/test_runner.py`

```python
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
```

---

## 9. Verifica finale

```bash
# Suite completa
pytest tests/ -v --tb=short
```

Output atteso: tutti i test `PASSED`.

Verifica che il runner sia avviabile:

```bash
python ollama_symphony.py --dry-run --verbose
python ollama_symphony.py --help
```

---

### Criteri di completamento

- [ ] `OllamaClient` con round-robin e fallback su host multipli
- [ ] `ReactLoop` implementa correttamente il loop ReAct
- [ ] Context truncation mantiene system + user iniziale + ultimi 6
- [ ] `OllamaSymphonyRunner` con retry, resume e git commit
- [ ] CLI completa: `--check`, `--list-models`, `--dry-run`, `--reset`, `--verbose`
- [ ] `pytest tests/ -v` → tutti PASSED (test_parsing, test_tools, test_react_loop, test_runner)
- [ ] `--dry-run` non invoca Ollama né git
- [ ] `README.md` aggiornato con sezioni: "Differenze da symphony.py", "Multi-host Ollama", "Modelli consigliati", tabella "Tool disponibili"
