#!/usr/bin/env python3
"""
Ollama Symphony — sequential task runner for local Ollama models.

Reads tasks from TASKS.md, executes them via a ReAct loop with tool calling,
commits results to git. Compatible with symphony.py state files.

Usage:
    python ollama_symphony.py [--tasks TASKS.md] [--workflow WORKFLOW.md] [--dry-run]
"""

import argparse
import json
import logging
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import yaml  # pip install pyyaml


# ---------------------------------------------------------------------------
# Domain model
# ---------------------------------------------------------------------------

@dataclass
class Task:
    index: int          # 0-based position in file
    title: str          # text of the ## heading
    body: str           # markdown body of the task
    slug: str = ""      # sanitized key used in state file

    def __post_init__(self):
        if not self.slug:
            self.slug = _slugify(self.title)


@dataclass
class WorkflowConfig:
    # agent settings
    max_retries: int = 3
    retry_delay_s: float = 10.0
    turn_timeout_s: int = 600
    max_iterations: int = 20        # max ReAct iterations per task

    # git settings
    commit_after_each_task: bool = True
    commit_message_template: str = "feat: {task_title}"

    # ollama settings
    ollama_hosts: list[str] = field(default_factory=lambda: ["http://localhost:11434"])
    ollama_model: str = "qwen2.5-coder:7b"
    ollama_timeout_s: int = 120
    ollama_temperature: float = 0.2
    ollama_context_window: int = 8192
    ollama_num_ctx: int = 8192

    # tool settings
    enabled_tools: list[str] = field(default_factory=lambda: [
        "run_shell", "read_file", "write_file", "list_directory"
    ])
    shell_timeout_s: int = 30
    working_dir: str = "."

    # system prompt
    system_prompt: str = (
        "You are an autonomous development agent working on a software project.\n"
        "For each task assigned to you:\n"
        "1. Read the task description carefully.\n"
        "2. Implement the required code changes using the available tools.\n"
        "3. Write the tests specified in the task.\n"
        "4. Run the test suite and iterate until all tests pass.\n"
        "5. Do not ask for confirmation — proceed autonomously.\n"
        "6. When the task is complete, call the `task_complete` tool with a brief summary."
    )


@dataclass
class TaskState:
    status: str           # "pending" | "completed" | "failed"
    attempts: int = 0
    error: Optional[str] = None
    completed_at: Optional[str] = None


@dataclass
class ToolCall:
    name: str
    arguments: dict


@dataclass
class ToolResult:
    tool_name: str
    success: bool
    output: str           # stdout / file content / error message
    exit_code: Optional[int] = None


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class OllamaConnectionError(Exception):
    """Raised when all configured Ollama hosts are unreachable."""


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


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def _slugify(text: str) -> str:
    """Convert a task title to a safe dict key."""
    return re.sub(r"[^A-Za-z0-9._-]", "_", text).strip("_")


def parse_tasks(path: Path) -> list[Task]:
    """
    Parse TASKS.md into an ordered list of Task objects.
    Each ## heading starts a new task; the body is everything until the next ##.
    Compatible with symphony.py TASKS.md format.
    """
    content = path.read_text(encoding="utf-8")
    tasks: list[Task] = []

    parts = re.split(r"^##\s+(.+)$", content, flags=re.MULTILINE)
    it = iter(parts[1:])
    for index, (title, body) in enumerate(zip(it, it)):
        tasks.append(Task(index=index, title=title.strip(), body=body.strip()))

    if not tasks:
        raise ValueError(f"No tasks found in {path}. Use ## headings to define tasks.")

    return tasks


def parse_workflow(path: Path) -> WorkflowConfig:
    """
    Parse WORKFLOW.md with optional YAML front matter.
    Extends symphony.py format with ollama and tools sections.
    Returns WorkflowConfig.
    """
    content = path.read_text(encoding="utf-8")
    config = WorkflowConfig()

    if content.startswith("---"):
        end = content.find("\n---", 3)
        if end != -1:
            front_matter = content[3:end].strip()
            body = content[end + 4:].strip()
            data = yaml.safe_load(front_matter) or {}

            agent_cfg = data.get("agent", {})
            if "max_retries" in agent_cfg:
                config.max_retries = int(agent_cfg["max_retries"])
            if "retry_delay_s" in agent_cfg:
                config.retry_delay_s = float(agent_cfg["retry_delay_s"])
            if "turn_timeout_s" in agent_cfg:
                config.turn_timeout_s = int(agent_cfg["turn_timeout_s"])
            if "max_iterations" in agent_cfg:
                config.max_iterations = int(agent_cfg["max_iterations"])

            git_cfg = data.get("git", {})
            if "commit_after_each_task" in git_cfg:
                config.commit_after_each_task = bool(git_cfg["commit_after_each_task"])
            if "commit_message_template" in git_cfg:
                config.commit_message_template = str(git_cfg["commit_message_template"])

            ollama_cfg = data.get("ollama", {})
            if "hosts" in ollama_cfg:
                config.ollama_hosts = list(ollama_cfg["hosts"])
            if "model" in ollama_cfg:
                config.ollama_model = str(ollama_cfg["model"])
            if "timeout_s" in ollama_cfg:
                config.ollama_timeout_s = int(ollama_cfg["timeout_s"])
            if "temperature" in ollama_cfg:
                config.ollama_temperature = float(ollama_cfg["temperature"])
            if "context_window" in ollama_cfg:
                config.ollama_context_window = int(ollama_cfg["context_window"])
            if "num_ctx" in ollama_cfg:
                config.ollama_num_ctx = int(ollama_cfg["num_ctx"])

            tools_cfg = data.get("tools", {})
            if "enabled" in tools_cfg:
                config.enabled_tools = list(tools_cfg["enabled"])
            if "shell_timeout_s" in tools_cfg:
                config.shell_timeout_s = int(tools_cfg["shell_timeout_s"])
            if "working_dir" in tools_cfg:
                config.working_dir = str(tools_cfg["working_dir"])

            if body:
                config.system_prompt = body
        else:
            config.system_prompt = content.strip()
    else:
        body = content.strip()
        if body:
            config.system_prompt = body

    return config


# ---------------------------------------------------------------------------
# State persistence
# ---------------------------------------------------------------------------

class StateStore:
    """
    Persists task progress to a JSON file so runs can be resumed.
    Schema-compatible with symphony.py state files.
    """

    def __init__(self, path: Path):
        self.path = path
        self._data: dict = {}
        self._load()

    def _load(self):
        if self.path.exists():
            try:
                self._data = json.loads(self.path.read_text(encoding="utf-8"))
            except Exception:
                self._data = {}

    def _save(self):
        self.path.write_text(
            json.dumps(self._data, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def init_run(self, tasks_file: str):
        if "started_at" not in self._data:
            self._data["started_at"] = _now_iso()
        self._data["tasks_file"] = tasks_file
        if "tasks" not in self._data:
            self._data["tasks"] = {}
        self._save()

    def get(self, slug: str) -> Optional[TaskState]:
        raw = self._data.get("tasks", {}).get(slug)
        if raw is None:
            return None
        return TaskState(**raw)

    def set(self, slug: str, state: TaskState):
        self._data.setdefault("tasks", {})[slug] = {
            "status": state.status,
            "attempts": state.attempts,
            "error": state.error,
            "completed_at": state.completed_at,
        }
        self._save()

    def is_completed(self, slug: str) -> bool:
        s = self.get(slug)
        return s is not None and s.status == "completed"


# ---------------------------------------------------------------------------
# Tool definitions (Ollama function-calling format)
# ---------------------------------------------------------------------------

ALL_TOOL_DEFS: dict[str, dict] = {
    "run_shell": {
        "type": "function",
        "function": {
            "name": "run_shell",
            "description": (
                "Execute a shell command and return stdout + stderr. "
                "Use for running tests, git commands, installing packages, etc."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "Shell command to execute"},
                },
                "required": ["command"],
            },
        },
    },
    "read_file": {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read the contents of a file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Relative path to the file"},
                },
                "required": ["path"],
            },
        },
    },
    "write_file": {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write content to a file, creating it and any parent dirs if needed.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Relative path to the file"},
                    "content": {"type": "string", "description": "Content to write"},
                },
                "required": ["path", "content"],
            },
        },
    },
    "list_directory": {
        "type": "function",
        "function": {
            "name": "list_directory",
            "description": "List files and directories at the given relative path.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Relative path to list (default: '.')",
                        "default": ".",
                    },
                },
                "required": [],
            },
        },
    },
    "task_complete": {
        "type": "function",
        "function": {
            "name": "task_complete",
            "description": "Signal that the current task is complete. Call this when all work is done.",
            "parameters": {
                "type": "object",
                "properties": {
                    "summary": {
                        "type": "string",
                        "description": "Brief summary of what was accomplished",
                    },
                },
                "required": ["summary"],
            },
        },
    },
}


# ---------------------------------------------------------------------------
# Tool execution
# ---------------------------------------------------------------------------

def execute_run_shell(command: str, working_dir: Path, timeout_s: int) -> ToolResult:
    try:
        result = subprocess.run(
            command, shell=True, capture_output=True, text=True,
            timeout=timeout_s, cwd=str(working_dir),
        )
        output = (
            f"exit_code: {result.returncode}\n"
            f"stdout: {result.stdout.strip()}\n"
            f"stderr: {result.stderr.strip()}"
        )
        return ToolResult(
            tool_name="run_shell",
            success=result.returncode == 0,
            output=output,
            exit_code=result.returncode,
        )
    except subprocess.TimeoutExpired:
        return ToolResult(
            tool_name="run_shell", success=False,
            output=f"Command timed out after {timeout_s}s",
            exit_code=-1,
        )
    except Exception as exc:
        return ToolResult(tool_name="run_shell", success=False, output=str(exc))


_READ_FILE_MAX_CHARS = 8000


def execute_read_file(path_str: str, working_dir: Path) -> ToolResult:
    try:
        safe = _safe_path(working_dir, path_str)
        content = safe.read_text(encoding="utf-8")
        if len(content) > _READ_FILE_MAX_CHARS:
            content = content[:_READ_FILE_MAX_CHARS] + f"\n[truncated — {len(content)} total chars]"
        return ToolResult(tool_name="read_file", success=True, output=content)
    except ValueError as exc:
        return ToolResult(tool_name="read_file", success=False, output=str(exc))
    except FileNotFoundError:
        return ToolResult(
            tool_name="read_file", success=False, output=f"File not found: {path_str}",
        )
    except Exception as exc:
        return ToolResult(tool_name="read_file", success=False, output=str(exc))


def execute_write_file(path_str: str, content: str, working_dir: Path) -> ToolResult:
    try:
        safe = _safe_path(working_dir, path_str)
        safe.parent.mkdir(parents=True, exist_ok=True)
        safe.write_text(content, encoding="utf-8")
        return ToolResult(
            tool_name="write_file", success=True,
            output=f"Written {len(content)} chars to {path_str}",
        )
    except ValueError as exc:
        return ToolResult(tool_name="write_file", success=False, output=str(exc))
    except Exception as exc:
        return ToolResult(tool_name="write_file", success=False, output=str(exc))


def execute_list_directory(path_str: str, working_dir: Path) -> ToolResult:
    try:
        safe = _safe_path(working_dir, path_str or ".")
        entries = sorted(safe.iterdir(), key=lambda p: (p.is_file(), p.name))
        if not entries:
            return ToolResult(tool_name="list_directory", success=True, output="(empty directory)")
        lines = [f"{e.name}{'/' if e.is_dir() else ''}" for e in entries]
        return ToolResult(tool_name="list_directory", success=True, output="\n".join(lines))
    except ValueError as exc:
        return ToolResult(tool_name="list_directory", success=False, output=str(exc))
    except FileNotFoundError:
        return ToolResult(
            tool_name="list_directory", success=False, output=f"Directory not found: {path_str}",
        )
    except Exception as exc:
        return ToolResult(tool_name="list_directory", success=False, output=str(exc))


class ToolExecutor:
    """Dispatches tool calls to the appropriate implementation function."""

    def __init__(self, config: WorkflowConfig):
        self.config = config
        self._working_dir = Path(config.working_dir).resolve()

    def execute(self, tool_call: ToolCall) -> ToolResult:
        """Execute a tool call and return the result."""
        match tool_call.name:
            case "run_shell":
                command = tool_call.arguments.get("command", "")
                return execute_run_shell(command, self._working_dir, self.config.shell_timeout_s)
            case "read_file":
                path = tool_call.arguments.get("path", "")
                return execute_read_file(path, self._working_dir)
            case "write_file":
                path = tool_call.arguments.get("path", "")
                content = tool_call.arguments.get("content", "")
                return execute_write_file(path, content, self._working_dir)
            case "list_directory":
                path = tool_call.arguments.get("path", ".")
                return execute_list_directory(path, self._working_dir)
            case "task_complete":
                return ToolResult(
                    tool_name="task_complete",
                    success=True,
                    output="[task_complete signal — handled by runner]",
                )
            case _:
                return ToolResult(
                    tool_name=tool_call.name,
                    success=False,
                    output=f"Unknown tool: {tool_call.name!r}",
                )


def _safe_path(working_dir: Path, user_path: str) -> Path:
    """
    Resolve user_path relative to working_dir and verify it stays inside.
    Raises ValueError if the resolved path escapes working_dir.
    """
    resolved = (working_dir / user_path).resolve()
    root = working_dir.resolve()
    if not str(resolved).startswith(str(root) + os.sep) and resolved != root:
        raise ValueError(f"Path traversal blocked: {user_path!r} resolves outside working dir")
    return resolved


def execute_tool(
    call: ToolCall,
    config: "WorkflowConfig",
    logger: logging.Logger,
) -> ToolResult:
    """Dispatch a ToolCall to the appropriate handler."""
    name = call.name
    args = call.arguments

    if name == "run_shell":
        return _tool_run_shell(args.get("command", ""), config, logger)
    elif name == "read_file":
        return _tool_read_file(args.get("path", ""), config)
    elif name == "write_file":
        return _tool_write_file(args.get("path", ""), args.get("content", ""), config)
    elif name == "list_directory":
        return _tool_list_directory(args.get("path", "."), config)
    elif name == "task_complete":
        return ToolResult(tool_name="task_complete", success=True, output=args.get("summary", ""))
    else:
        return ToolResult(tool_name=name, success=False, output=f"Unknown tool: {name}")


def _tool_run_shell(command: str, config: "WorkflowConfig", logger: logging.Logger) -> ToolResult:
    logger.debug("run_shell: %s", command)
    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=config.shell_timeout_s,
            cwd=config.working_dir,
        )
        output = result.stdout
        if result.stderr:
            output += "\n[stderr]\n" + result.stderr
        return ToolResult(
            tool_name="run_shell",
            success=result.returncode == 0,
            output=output.strip(),
            exit_code=result.returncode,
        )
    except subprocess.TimeoutExpired:
        return ToolResult(
            tool_name="run_shell", success=False,
            output=f"Command timed out after {config.shell_timeout_s}s",
        )
    except Exception as exc:
        return ToolResult(tool_name="run_shell", success=False, output=str(exc))


def _tool_read_file(path_str: str, config: "WorkflowConfig") -> ToolResult:
    try:
        safe = _safe_path(Path(config.working_dir), path_str)
        content = safe.read_text(encoding="utf-8")
        return ToolResult(tool_name="read_file", success=True, output=content)
    except ValueError as exc:
        return ToolResult(tool_name="read_file", success=False, output=str(exc))
    except FileNotFoundError:
        return ToolResult(
            tool_name="read_file", success=False, output=f"File not found: {path_str}",
        )
    except Exception as exc:
        return ToolResult(tool_name="read_file", success=False, output=str(exc))


def _tool_write_file(path_str: str, content: str, config: "WorkflowConfig") -> ToolResult:
    try:
        safe = _safe_path(Path(config.working_dir), path_str)
        safe.parent.mkdir(parents=True, exist_ok=True)
        safe.write_text(content, encoding="utf-8")
        return ToolResult(
            tool_name="write_file", success=True,
            output=f"Written {len(content)} chars to {path_str}",
        )
    except ValueError as exc:
        return ToolResult(tool_name="write_file", success=False, output=str(exc))
    except Exception as exc:
        return ToolResult(tool_name="write_file", success=False, output=str(exc))


def _tool_list_directory(path_str: str, config: "WorkflowConfig") -> ToolResult:
    try:
        safe = _safe_path(Path(config.working_dir), path_str or ".")
        entries = sorted(safe.iterdir(), key=lambda p: (p.is_file(), p.name))
        lines = [f"{e.name}{'/' if e.is_dir() else ''}" for e in entries]
        return ToolResult(tool_name="list_directory", success=True, output="\n".join(lines))
    except ValueError as exc:
        return ToolResult(tool_name="list_directory", success=False, output=str(exc))
    except FileNotFoundError:
        return ToolResult(
            tool_name="list_directory", success=False, output=f"Directory not found: {path_str}",
        )
    except Exception as exc:
        return ToolResult(tool_name="list_directory", success=False, output=str(exc))


# ---------------------------------------------------------------------------
# Prompt assembly (identical to symphony.py)
# ---------------------------------------------------------------------------

def build_prompt(
    task: Task,
    config: "WorkflowConfig",
    completed_tasks: list[Task],
) -> str:
    """Compose the full prompt for one task, including completed-task context."""
    parts: list[str] = [config.system_prompt.strip()]

    if completed_tasks:
        summary_lines = ["---", "## Previously completed tasks (already committed to git):", ""]
        for t in completed_tasks:
            summary_lines.append(f"- **{t.title}**")
        summary_lines.append("")
        summary_lines.append(
            "These are already done. Do not redo them; focus only on the current task below."
        )
        parts.append("\n".join(summary_lines))

    parts.append("---")
    parts.append(f"## Current task: {task.title}")
    parts.append("")
    parts.append(task.body)

    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Tool schemas (JSON format for Ollama)
# ---------------------------------------------------------------------------

_TOOL_SCHEMAS: dict[str, dict] = {
    "run_shell": {
        "type": "function",
        "function": {
            "name": "run_shell",
            "description": (
                "Execute a shell command in the working directory. "
                "Use for running tests, installing packages, git commands, etc."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "Shell command to execute",
                    }
                },
                "required": ["command"],
            },
        },
    },
    "read_file": {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read the content of a file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "File path relative to working directory",
                    }
                },
                "required": ["path"],
            },
        },
    },
    "write_file": {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write content to a file, creating it or overwriting it.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "File path relative to working directory",
                    },
                    "content": {
                        "type": "string",
                        "description": "Content to write to the file",
                    },
                },
                "required": ["path", "content"],
            },
        },
    },
    "list_directory": {
        "type": "function",
        "function": {
            "name": "list_directory",
            "description": "List files and directories at a path.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Directory path relative to working directory (default: '.')",
                    }
                },
                "required": [],
            },
        },
    },
    "task_complete": {
        "type": "function",
        "function": {
            "name": "task_complete",
            "description": (
                "Call this when the task is fully completed and all tests pass. "
                "This signals the runner that the task is done."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "summary": {
                        "type": "string",
                        "description": "Brief description of what was implemented",
                    }
                },
                "required": ["summary"],
            },
        },
    },
}


def build_tool_schemas(config: "WorkflowConfig") -> list[dict]:
    """
    Return the list of tool schemas to pass to Ollama.
    Always includes task_complete regardless of config.enabled_tools.
    """
    schemas = []
    for name in config.enabled_tools:
        if name in _TOOL_SCHEMAS and name != "task_complete":
            schemas.append(_TOOL_SCHEMAS[name])
    schemas.append(_TOOL_SCHEMAS["task_complete"])
    return schemas


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
            if self._should_truncate(messages):
                messages = self._truncate_context(messages)
                self._log.warning("context_truncated iteration=%d", iteration)

            try:
                response = self._client.chat(messages, tools)
            except OllamaConnectionError as exc:
                return False, f"Ollama unreachable: {exc}"
            except Exception as exc:
                return False, f"Ollama error: {exc}"

            msg = self._extract_message(response)
            tool_calls = self._extract_tool_calls(msg)

            self._log.debug(
                "react_iteration=%d tool_calls=%d", iteration, len(tool_calls)
            )

            if not tool_calls:
                content = msg.get("content", "") if isinstance(msg, dict) else str(msg)
                self._log.warning(
                    "react_no_tool_calls iteration=%d — treating as implicit completion",
                    iteration,
                )
                return True, content[:300] if content else "completed (no summary)"

            task_done = False
            task_summary = ""
            for tc_raw in tool_calls:
                tool_call = self._parse_tool_call(tc_raw)

                if tool_call.name == "task_complete":
                    task_summary = tool_call.arguments.get("summary", "")
                    self._log.info("task_complete summary=%r", task_summary)
                    task_done = True
                    messages = self._append_assistant_turn(messages, msg)
                    messages = self._append_tool_result(
                        messages,
                        tool_call.name,
                        ToolResult(tool_name="task_complete", success=True, output="acknowledged"),
                    )
                    break

                result = self._executor.execute(tool_call)
                self._log.debug(
                    "tool_exec name=%s success=%s exit_code=%s",
                    tool_call.name,
                    result.success,
                    result.exit_code,
                )

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
        """Keep: system message (index 0) + first user message (index 1) + last 6 messages."""
        if len(messages) <= 8:
            return messages
        head = messages[:2]
        tail = messages[-6:]
        return head + tail


# ---------------------------------------------------------------------------
# Ollama ReAct loop
# ---------------------------------------------------------------------------

def _get_tool_defs(enabled: list[str]) -> list[dict]:
    """Return Ollama tool-def dicts for the requested tool names."""
    defs = [ALL_TOOL_DEFS[n] for n in enabled if n in ALL_TOOL_DEFS]
    if "task_complete" not in enabled:
        defs.append(ALL_TOOL_DEFS["task_complete"])
    return defs


def run_react_loop(
    task: Task,
    config: "WorkflowConfig",
    prompt: str,
    dry_run: bool = False,
    logger: Optional[logging.Logger] = None,
) -> tuple[bool, str]:
    """
    Run the ReAct loop for a single task via Ollama.
    Returns (success, summary_or_error).
    """
    import ollama as _ollama

    log = logger or logging.getLogger(__name__)

    if dry_run:
        log.info(
            "[dry-run] Would run ReAct loop for task %r on %s",
            task.title, config.ollama_hosts[0],
        )
        return True, "[dry-run] skipped"

    messages: list[dict] = [{"role": "user", "content": prompt}]
    tool_defs = _get_tool_defs(config.enabled_tools)
    client = _ollama.Client(host=config.ollama_hosts[0])

    for iteration in range(1, config.max_iterations + 1):
        log.debug("react iteration=%d", iteration)

        try:
            response = client.chat(
                model=config.ollama_model,
                messages=messages,
                tools=tool_defs,
                options={
                    "temperature": config.ollama_temperature,
                    "num_ctx": config.ollama_num_ctx,
                },
            )
        except Exception as exc:
            return False, f"Ollama error: {exc}"

        msg = response.message
        messages.append({"role": "assistant", "content": msg.content or ""})

        if not msg.tool_calls:
            if iteration == config.max_iterations:
                return False, f"Max iterations ({config.max_iterations}) reached without task_complete"
            continue

        task_done = False
        completion_summary = ""

        for tc in msg.tool_calls:
            tool_name = tc.function.name
            tool_args = dict(tc.function.arguments) if tc.function.arguments else {}

            call = ToolCall(name=tool_name, arguments=tool_args)
            log.info("tool_call=%s args=%s", tool_name, list(tool_args.keys()))

            result = execute_tool(call, config, log)
            log.debug(
                "tool_result=%s success=%s output_len=%d",
                result.tool_name, result.success, len(result.output),
            )

            messages.append({"role": "tool", "content": result.output})

            if tool_name == "task_complete":
                task_done = True
                completion_summary = result.output

        if task_done:
            return True, completion_summary

    return False, f"Max iterations ({config.max_iterations}) reached without task_complete"


# ---------------------------------------------------------------------------
# Git helpers (identical to symphony.py)
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
        subprocess.run(["git", "commit", "-m", message], check=True, capture_output=True)
        logger.info("git: committed — %s", message)
        return True
    except subprocess.CalledProcessError as exc:
        stderr = exc.stderr.decode(errors="replace").strip() if exc.stderr else ""
        logger.error("git commit failed: %s", stderr or exc)
        return False


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

class OllamaSymphonyRunner:
    def __init__(
        self,
        tasks: list[Task],
        config: "WorkflowConfig",
        state: StateStore,
        dry_run: bool = False,
    ):
        self.tasks = tasks
        self.config = config
        self.state = state
        self.dry_run = dry_run
        self.log = logging.getLogger("ollama_symphony")

    def run(self) -> bool:
        """Execute all tasks in order. Returns True if all completed successfully."""
        total = len(self.tasks)
        self.log.info("Ollama Symphony start — %d task(s)", total)
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

        self.log.info("Ollama Symphony complete — all %d task(s) succeeded", total)
        return True

    def _run_task_with_retries(self, task: Task, completed_so_far: list[Task]) -> bool:
        max_attempts = self.config.max_retries + 1
        task_state = self.state.get(task.slug) or TaskState(status="pending")

        for attempt in range(1, max_attempts + 1):
            self.log.info(
                "task_start index=%d title=%r attempt=%d/%d",
                task.index + 1, task.title, attempt, max_attempts,
            )

            prompt = build_prompt(task, self.config, completed_so_far)
            success, output = run_react_loop(
                task, self.config, prompt,
                dry_run=self.dry_run, logger=self.log,
            )

            task_state.attempts = attempt

            if success:
                if self.config.commit_after_each_task:
                    commit_msg = self.config.commit_message_template.format(
                        task_title=task.title,
                        task_index=task.index + 1,
                    )
                    committed = git_commit(commit_msg, self.log) if not self.dry_run else True
                    if not committed:
                        self.log.warning("git commit failed for %r — continuing", task.title)

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
            task_state.error = output[:500]
            self.state.set(task.slug, task_state)
            self.log.warning(
                "task_status=failed index=%d title=%r attempt=%d error=%r",
                task.index + 1, task.title, attempt, output[:200],
            )

            if attempt < max_attempts:
                self.log.info("Retrying in %.0fs…", self.config.retry_delay_s)
                if not self.dry_run:
                    time.sleep(self.config.retry_delay_s)

        return False


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def setup_logging(verbose: bool = False):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        format="%(asctime)s %(levelname)-8s %(message)s",
        datefmt="%H:%M:%S",
        level=level,
        stream=sys.stderr,
    )


# ---------------------------------------------------------------------------
# CLI helpers
# ---------------------------------------------------------------------------

def list_models(config: "WorkflowConfig", logger: logging.Logger) -> None:
    """Print all available models from all configured Ollama hosts."""
    import ollama as _ollama
    for host in config.ollama_hosts:
        try:
            client = _ollama.Client(host=host)
            models = client.list()
            logger.info("Host %s:", host)
            for m in models.models:
                logger.info("  %s", m.model)
        except Exception as exc:
            logger.error("Host %s: error — %s", host, exc)


def check_connectivity(config: "WorkflowConfig", logger: logging.Logger) -> bool:
    """Verify Ollama connectivity for all configured hosts."""
    import ollama as _ollama
    all_ok = True
    for host in config.ollama_hosts:
        try:
            _ollama.Client(host=host).list()
            logger.info("Host %s: OK", host)
        except Exception as exc:
            logger.error("Host %s: FAILED — %s", host, exc)
            all_ok = False
    return all_ok


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Ollama Symphony — run tasks sequentially with a local Ollama model.",
    )
    parser.add_argument("--tasks", default="TASKS.md", metavar="FILE",
                        help="Path to TASKS.md (default: TASKS.md)")
    parser.add_argument("--workflow", default="WORKFLOW.md", metavar="FILE",
                        help="Path to WORKFLOW.md (default: WORKFLOW.md)")
    parser.add_argument("--state", default="TASKS.state.json", metavar="FILE",
                        help="Path to state file (default: TASKS.state.json)")
    parser.add_argument("--reset", action="store_true",
                        help="Ignore saved state, restart from task 1")
    parser.add_argument("--dry-run", action="store_true",
                        help="Parse and log, do not invoke Ollama or git")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Enable debug logging")
    parser.add_argument("--list-models", action="store_true",
                        help="List models available on all configured Ollama hosts")
    parser.add_argument("--check", action="store_true",
                        help="Validate config and Ollama connectivity, then exit")
    args = parser.parse_args()

    setup_logging(args.verbose)
    log = logging.getLogger("ollama_symphony")

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

    if args.list_models:
        list_models(config, log)
        sys.exit(0)

    if args.check:
        ok = check_connectivity(config, log)
        sys.exit(0 if ok else 1)

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
    for t in tasks:
        log.debug("  [%d] %s", t.index + 1, t.title)

    state_path = Path(args.state)
    if args.reset and state_path.exists():
        state_path.unlink()
        log.info("State file reset")

    state = StateStore(state_path)
    state.init_run(str(tasks_path))

    runner = OllamaSymphonyRunner(tasks, config, state, dry_run=args.dry_run)
    success = runner.run()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
