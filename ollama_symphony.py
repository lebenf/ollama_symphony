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
# CLI
# ---------------------------------------------------------------------------

def main():
    setup_logging()
    logging.getLogger("ollama_symphony").info("ollama_symphony placeholder — run tasks 2-4 first")


if __name__ == "__main__":
    main()
