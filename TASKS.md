# Task 1 — Setup struttura progetto e domain model

## Obiettivo

Creare la struttura base del progetto `ollama_symphony/` con il domain model, le funzioni di
parsing e la persistenza dello stato. Questo task non richiede Ollama né tool calling: getta le
fondamenta su cui i task successivi costruiranno.

---

## Struttura da creare

```
ollama_symphony/
├── ollama_symphony.py
├── WORKFLOW.md
├── TASKS.md
├── requirements.txt
├── README.md
└── tests/
    ├── __init__.py
    └── test_parsing.py
```

---

## 1. `requirements.txt`

```
ollama>=0.3.0
pyyaml>=6.0
pytest>=8.0
pytest-timeout>=2.3
```

---

## 2. `ollama_symphony.py` — Domain model e parsing

Crea il file con i seguenti contenuti **nell'ordine indicato**.

### 2.1 Header e imports

```python
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
```

### 2.2 Dataclass `Task`

```python
@dataclass
class Task:
    index: int          # 0-based position in file
    title: str          # text of the ## heading
    body: str           # markdown body of the task
    slug: str = ""      # sanitized key used in state file

    def __post_init__(self):
        if not self.slug:
            self.slug = _slugify(self.title)
```

### 2.3 Dataclass `WorkflowConfig`

```python
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
```

### 2.4 Dataclasses `TaskState`, `ToolCall`, `ToolResult`

```python
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
```

### 2.5 Funzioni di parsing

Queste funzioni devono essere **identiche** a quelle di `symphony.py` per garantire compatibilità
con i file esistenti.

```python
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

            # agent section
            agent_cfg = data.get("agent", {})
            if "max_retries" in agent_cfg:
                config.max_retries = int(agent_cfg["max_retries"])
            if "retry_delay_s" in agent_cfg:
                config.retry_delay_s = float(agent_cfg["retry_delay_s"])
            if "turn_timeout_s" in agent_cfg:
                config.turn_timeout_s = int(agent_cfg["turn_timeout_s"])
            if "max_iterations" in agent_cfg:
                config.max_iterations = int(agent_cfg["max_iterations"])

            # git section
            git_cfg = data.get("git", {})
            if "commit_after_each_task" in git_cfg:
                config.commit_after_each_task = bool(git_cfg["commit_after_each_task"])
            if "commit_message_template" in git_cfg:
                config.commit_message_template = str(git_cfg["commit_message_template"])

            # ollama section
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

            # tools section
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
```

### 2.6 `StateStore`

Identica a `symphony.py`:

```python
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
```

### 2.7 Utility

```python
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
```

### 2.8 Placeholder `main()`

Aggiungi in fondo al file un `main()` minimale che verrà completato nel Task 4:

```python
def main():
    setup_logging()
    logging.getLogger("ollama_symphony").info("ollama_symphony placeholder — run tasks 2-4 first")


if __name__ == "__main__":
    main()
```

---

## 3. File di esempio: `WORKFLOW.md`

```yaml
---
agent:
  max_retries: 3
  retry_delay_s: 10
  turn_timeout_s: 600
  max_iterations: 20

git:
  commit_after_each_task: true
  commit_message_template: "feat: {task_title}"

ollama:
  hosts:
    - http://localhost:11434
  model: qwen2.5-coder:7b
  timeout_s: 120
  temperature: 0.2
  context_window: 8192
  num_ctx: 8192

tools:
  enabled:
    - run_shell
    - read_file
    - write_file
    - list_directory
  shell_timeout_s: 30
  working_dir: "."
---
You are an autonomous development agent working on a software project.

For each task assigned to you:
1. Read the task description carefully.
2. Implement the required code changes using the available tools.
3. Write the tests specified in the task.
4. Run the test suite and iterate until all tests pass.
5. Do not ask for confirmation — proceed autonomously.
6. When the task is complete, call the `task_complete` tool with a brief summary.
```

---

## 4. File di esempio: `TASKS.md`

```markdown
# Project Tasks

## Task 1: Setup project structure

Create the basic project structure with the following files:
- `src/__init__.py`
- `src/models.py` with a `User` dataclass (fields: id, name, email)
- `tests/__init__.py`
- `tests/test_models.py` with at least one test that verifies the User dataclass works correctly

Run the tests with `pytest` and make sure they pass.

## Task 2: Add validation logic

In `src/models.py`, add a `validate_email(email: str) -> bool` function.

Add tests covering valid and invalid emails.
Run the tests and make sure they all pass.
```

---

## 5. `tests/__init__.py`

File vuoto.

---

## 6. `tests/test_parsing.py`

```python
"""Tests for parse_tasks, parse_workflow, _slugify, and StateStore."""

import json
import textwrap
from pathlib import Path

import pytest

# Import from the main module
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
    # defaults preserved
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
```

---

## 7. `README.md`

Crea un README con le seguenti sezioni:

```markdown
# Ollama Symphony

A sequential task runner for local LLM models via Ollama.
Reads development tasks from `TASKS.md`, executes them via a ReAct loop
with tool calling, and commits results to git.

Compatible with `symphony.py` task and state file formats.

## Requirements

- Python 3.11+
- [Ollama](https://ollama.ai) running locally or on a remote host
- `pip install -r requirements.txt`
- A git repository (for auto-commit support)

## Quick start

    # 1. Start Ollama and pull a model
    ollama pull qwen2.5-coder:7b

    # 2. Install dependencies
    pip install -r requirements.txt

    # 3. Configure WORKFLOW.md and create TASKS.md

    # 4. Run
    python ollama_symphony.py

## CLI options

    --tasks FILE      Path to TASKS.md         (default: TASKS.md)
    --workflow FILE   Path to WORKFLOW.md       (default: WORKFLOW.md)
    --state FILE      Path to state file        (default: TASKS.state.json)
    --reset           Ignore saved state, restart from task 1
    --dry-run         Parse and log, do not invoke Ollama or git
    --verbose / -v    Enable debug logging
    --list-models     List models available on all configured Ollama hosts
    --check           Validate config and Ollama connectivity, then exit

## File formats

### TASKS.md

Each `##` heading defines one task. Format is identical to `symphony.py`.

### WORKFLOW.md

YAML front matter configures the runner. Markdown body is the system prompt.
See the included `WORKFLOW.md` for all available options.

## Differences from symphony.py

`symphony.py` delegates execution to Claude Code, which has native agentic
capabilities. `ollama_symphony.py` implements a ReAct loop: it sends prompts
to Ollama and handles tool calls (shell, file I/O) locally.
```

---

## 8. Verifica finale

Esegui i test e verifica che passino tutti:

```bash
cd ollama_symphony
pip install -r requirements.txt
pytest tests/test_parsing.py -v
```

Output atteso: tutti i test `PASSED`, nessun errore di import.

Verifica anche che il modulo sia importabile senza errori:

```bash
python -c "from ollama_symphony import Task, WorkflowConfig, TaskState, StateStore, parse_tasks, parse_workflow; print('OK')"
```

---

## Criteri di completamento

- [ ] `ollama_symphony.py` importabile senza errori
- [ ] Tutti i dataclass presenti con i campi corretti
- [ ] `parse_tasks` compatibile con i file `symphony.py` esistenti
- [ ] `parse_workflow` gestisce tutti i campi `ollama` e `tools`
- [ ] `StateStore` legge e scrive correttamente, regge il resume
- [ ] `pytest tests/test_parsing.py -v` → tutti PASSED
- [ ] `WORKFLOW.md`, `TASKS.md`, `README.md` creati
