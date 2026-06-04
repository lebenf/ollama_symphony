#!/usr/bin/env python3
"""
Claude Code Symphony — sequential task runner for Claude Code.

Usage:
    python symphony.py [--tasks TASKS.md] [--workflow WORKFLOW.md] [--dry-run]

Each task in TASKS.md is executed by a fresh Claude Code process.
On success, changes are committed to git. On failure, retries up to max_retries.
Progress is saved to TASKS.state.json so the run can be resumed after interruption.
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

    # git settings
    commit_after_each_task: bool = True
    commit_message_template: str = "feat: {task_title}"

    # claude code settings
    claude_command: str = "claude"
    # extra flags passed to every claude invocation
    claude_extra_args: list = field(default_factory=list)

    # system prompt prepended to every task prompt
    system_prompt: str = (
        "You are a development agent.\n"
        "For each task:\n"
        "1. Implement the requested code.\n"
        "2. Write and run the necessary tests.\n"
        "3. Make sure all tests pass before finishing.\n"
        "Do not ask for confirmation; proceed autonomously."
    )


@dataclass
class TaskState:
    status: str           # "pending" | "completed" | "failed"
    attempts: int = 0
    error: Optional[str] = None
    completed_at: Optional[str] = None


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
    """
    content = path.read_text(encoding="utf-8")
    tasks: list[Task] = []

    # Split on level-2 headings
    parts = re.split(r"^##\s+(.+)$", content, flags=re.MULTILINE)
    # parts = [preamble, title1, body1, title2, body2, ...]
    it = iter(parts[1:])  # skip preamble
    for index, (title, body) in enumerate(zip(it, it)):
        tasks.append(Task(index=index, title=title.strip(), body=body.strip()))

    if not tasks:
        raise ValueError(f"No tasks found in {path}. Use ## headings to define tasks.")

    return tasks


def parse_workflow(path: Path) -> tuple[WorkflowConfig, str]:
    """
    Parse WORKFLOW.md with optional YAML front matter.

    Returns (WorkflowConfig, system_prompt_string).
    """
    content = path.read_text(encoding="utf-8")
    config = WorkflowConfig()
    system_prompt = ""

    if content.startswith("---"):
        # Extract front matter
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

            git_cfg = data.get("git", {})
            if "commit_after_each_task" in git_cfg:
                config.commit_after_each_task = bool(git_cfg["commit_after_each_task"])
            if "commit_message_template" in git_cfg:
                config.commit_message_template = str(git_cfg["commit_message_template"])

            claude_cfg = data.get("claude", {})
            if "command" in claude_cfg:
                config.claude_command = str(claude_cfg["command"])
            if "extra_args" in claude_cfg:
                config.claude_extra_args = list(claude_cfg["extra_args"])

            system_prompt = body
        else:
            system_prompt = content
    else:
        system_prompt = content.strip()

    if system_prompt:
        config.system_prompt = system_prompt

    return config, system_prompt


# ---------------------------------------------------------------------------
# State persistence
# ---------------------------------------------------------------------------

class StateStore:
    """
    Persists task progress to a JSON file so runs can be resumed.

    Schema:
    {
      "started_at": "...",
      "tasks_file": "TASKS.md",
      "tasks": {
        "<slug>": {"status": "completed|failed|pending", "attempts": 2, ...}
      }
    }
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
# Prompt assembly
# ---------------------------------------------------------------------------

def build_prompt(
    task: Task,
    config: WorkflowConfig,
    completed_tasks: list[Task],
) -> str:
    """
    Compose the full prompt sent to Claude Code for one task.

    Structure:
      [system prompt]
      [completed tasks summary — if any]
      [current task]
    """
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
# Claude Code invocation
# ---------------------------------------------------------------------------

def run_claude(
    prompt: str,
    config: WorkflowConfig,
    dry_run: bool = False,
    logger: Optional[logging.Logger] = None,
) -> tuple[bool, str]:
    """
    Launch Claude Code with --print and --dangerously-skip-permissions.

    Returns (success: bool, output_or_error: str).
    """
    log = logger or logging.getLogger(__name__)

    cmd = [
        config.claude_command,
        "--print",
        "--dangerously-skip-permissions",
        *config.claude_extra_args,
        "-p",
        prompt,
    ]

    if dry_run:
        log.info("[dry-run] Would execute: %s", " ".join(cmd[:4]) + " -p <prompt>")
        return True, "[dry-run] skipped"

    log.debug("Launching claude (timeout=%ds)", config.turn_timeout_s)

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=config.turn_timeout_s,
        )
    except subprocess.TimeoutExpired:
        return False, f"Claude timed out after {config.turn_timeout_s}s"
    except FileNotFoundError:
        return False, (
            f"'{config.claude_command}' not found. "
            "Install Claude Code: https://docs.anthropic.com/en/docs/claude-code"
        )
    except Exception as exc:
        return False, f"Unexpected error launching claude: {exc}"

    output = result.stdout.strip()
    if result.returncode != 0:
        stderr = result.stderr.strip()
        error_msg = stderr or output or f"exit code {result.returncode}"
        return False, error_msg

    return True, output


# ---------------------------------------------------------------------------
# Git helpers
# ---------------------------------------------------------------------------

def git_commit(message: str, logger: logging.Logger) -> bool:
    """Stage all changes and create a commit. Returns True on success."""
    try:
        # Check if there is anything to commit
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


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

class SymphonyRunner:
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
        self.log = logging.getLogger("symphony")

    def run(self) -> bool:
        """
        Execute all tasks in order. Returns True if all tasks completed successfully.
        """
        total = len(self.tasks)
        self.log.info("Symphony start — %d task(s) to process", total)

        completed_so_far: list[Task] = []

        for task in self.tasks:
            # Resume: skip already-completed tasks
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
        max_attempts = self.config.max_retries + 1  # 1 first try + N retries
        task_state = self.state.get(task.slug) or TaskState(status="pending")

        for attempt in range(1, max_attempts + 1):
            self.log.info(
                "task_start index=%d title=%r attempt=%d/%d",
                task.index + 1, task.title, attempt, max_attempts,
            )

            prompt = build_prompt(task, self.config, completed_so_far)
            success, output = run_claude(
                prompt, self.config, dry_run=self.dry_run, logger=self.log
            )

            task_state.attempts = attempt

            if success:
                # Commit
                if self.config.commit_after_each_task:
                    commit_msg = self.config.commit_message_template.format(
                        task_title=task.title,
                        task_index=task.index + 1,
                    )
                    committed = git_commit(commit_msg, self.log) if not self.dry_run else True
                    if not committed:
                        self.log.warning(
                            "git commit failed for task %r — continuing anyway", task.title
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

            # Failure
            task_state.status = "failed"
            task_state.error = output[:500]  # truncate for state file
            self.state.set(task.slug, task_state)

            self.log.warning(
                "task_status=failed index=%d title=%r attempt=%d error=%r",
                task.index + 1, task.title, attempt, output[:200],
            )

            if attempt < max_attempts:
                self.log.info(
                    "Retrying in %.0fs…", self.config.retry_delay_s
                )
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
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Claude Code Symphony — run tasks sequentially with Claude Code.",
    )
    parser.add_argument(
        "--tasks",
        default="TASKS.md",
        metavar="FILE",
        help="Path to TASKS.md (default: TASKS.md)",
    )
    parser.add_argument(
        "--workflow",
        default="WORKFLOW.md",
        metavar="FILE",
        help="Path to WORKFLOW.md with optional YAML config (default: WORKFLOW.md)",
    )
    parser.add_argument(
        "--state",
        default="TASKS.state.json",
        metavar="FILE",
        help="Path to state file for resume support (default: TASKS.state.json)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse and log everything but do not invoke claude or git",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable debug logging",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Ignore existing state file and restart from the first task",
    )
    args = parser.parse_args()

    setup_logging(args.verbose)
    log = logging.getLogger("symphony")

    # --- Load workflow config ---
    config = WorkflowConfig()
    workflow_path = Path(args.workflow)
    if workflow_path.exists():
        try:
            config, _ = parse_workflow(workflow_path)
            log.info("Loaded workflow config from %s", workflow_path)
        except Exception as exc:
            log.error("Failed to parse %s: %s", workflow_path, exc)
            sys.exit(1)
    else:
        log.info("No %s found — using default config and built-in system prompt", workflow_path)

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
    for t in tasks:
        log.debug("  [%d] %s", t.index + 1, t.title)

    # --- State ---
    state_path = Path(args.state)
    if args.reset and state_path.exists():
        state_path.unlink()
        log.info("State file reset")

    state = StateStore(state_path)
    state.init_run(str(tasks_path))

    # --- Run ---
    runner = SymphonyRunner(tasks, config, state, dry_run=args.dry_run)
    success = runner.run()

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()