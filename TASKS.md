# Task 2 — Tool Registry e sicurezza path

### Prerequisiti

Il Task 1 deve essere completato: `ollama_symphony.py` deve già contenere i dataclass
(`Task`, `WorkflowConfig`, `TaskState`, `ToolCall`, `ToolResult`) e le funzioni di parsing.

---

### Obiettivo

Implementare il sistema di tool che il loop ReAct userà per eseguire azioni concrete:
eseguire comandi shell, leggere/scrivere file, listare directory. Include la prevenzione
del path traversal e gli schemi JSON per il tool calling Ollama.

---

### Cosa aggiungere a `ollama_symphony.py`

Aggiungi queste sezioni **dopo** i dataclass e il parsing, **prima** del placeholder `main()`.

---

## 1. Safety — Path traversal prevention

```python
# ---------------------------------------------------------------------------
# Tool safety
# ---------------------------------------------------------------------------

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
```

---

## 2. Tool execution functions

```python
# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------

def execute_run_shell(
    command: str,
    working_dir: Path,
    timeout_s: int,
) -> ToolResult:
    """Execute a shell command in working_dir. Returns ToolResult."""
    try:
        result = subprocess.run(
            command,
            shell=True,
            cwd=working_dir,
            timeout=timeout_s,
            capture_output=True,
            text=True,
        )
        stdout = result.stdout[:4000] if result.stdout else ""
        stderr = result.stderr[:2000] if result.stderr else ""
        output = f"exit_code: {result.returncode}\nstdout: {stdout}\nstderr: {stderr}"
        return ToolResult(
            tool_name="run_shell",
            success=result.returncode == 0,
            output=output,
            exit_code=result.returncode,
        )
    except subprocess.TimeoutExpired:
        return ToolResult(
            tool_name="run_shell",
            success=False,
            output=f"exit_code: -1\nstdout: \nstderr: Command timed out after {timeout_s}s",
            exit_code=-1,
        )
    except Exception as exc:
        return ToolResult(
            tool_name="run_shell",
            success=False,
            output=f"exit_code: -1\nstdout: \nstderr: Error: {exc}",
            exit_code=-1,
        )


def execute_read_file(path: str, working_dir: Path) -> ToolResult:
    """Read a file relative to working_dir. Blocks path traversal."""
    try:
        safe = _safe_path(working_dir, path)
        content = safe.read_text(encoding="utf-8")
        if len(content) > 8000:
            content = content[:8000] + "\n[... truncated ...]"
        return ToolResult(tool_name="read_file", success=True, output=content)
    except ValueError as exc:
        return ToolResult(tool_name="read_file", success=False, output=str(exc))
    except FileNotFoundError:
        return ToolResult(tool_name="read_file", success=False, output=f"File not found: {path}")
    except Exception as exc:
        return ToolResult(tool_name="read_file", success=False, output=f"Error reading file: {exc}")


def execute_write_file(path: str, content: str, working_dir: Path) -> ToolResult:
    """Write content to a file relative to working_dir. Creates parent dirs. Blocks path traversal."""
    try:
        safe = _safe_path(working_dir, path)
        safe.parent.mkdir(parents=True, exist_ok=True)
        safe.write_text(content, encoding="utf-8")
        return ToolResult(
            tool_name="write_file",
            success=True,
            output=f"Written {len(content)} chars to {path}",
        )
    except ValueError as exc:
        return ToolResult(tool_name="write_file", success=False, output=str(exc))
    except Exception as exc:
        return ToolResult(tool_name="write_file", success=False, output=f"Error writing file: {exc}")


def execute_list_directory(path: str, working_dir: Path) -> ToolResult:
    """List files and directories at path relative to working_dir."""
    target_path = path if path else "."
    try:
        safe = _safe_path(working_dir, target_path)
        if not safe.exists():
            return ToolResult(
                tool_name="list_directory",
                success=False,
                output=f"Path not found: {target_path}",
            )
        if not safe.is_dir():
            return ToolResult(
                tool_name="list_directory",
                success=False,
                output=f"Not a directory: {target_path}",
            )
        entries = sorted(safe.iterdir(), key=lambda p: (p.is_file(), p.name))
        lines = []
        for entry in entries[:200]:
            kind = "file" if entry.is_file() else "dir"
            lines.append(f"{kind}: {entry.name}")
        if len(list(safe.iterdir())) > 200:
            lines.append("[... truncated at 200 entries ...]")
        output = "\n".join(lines) if lines else "(empty directory)"
        return ToolResult(tool_name="list_directory", success=True, output=output)
    except ValueError as exc:
        return ToolResult(tool_name="list_directory", success=False, output=str(exc))
    except Exception as exc:
        return ToolResult(
            tool_name="list_directory", success=False, output=f"Error listing directory: {exc}"
        )
```

---

## 3. Tool schemas (JSON format for Ollama)

```python
# ---------------------------------------------------------------------------
# Tool schemas
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


def build_tool_schemas(config: WorkflowConfig) -> list[dict]:
    """
    Return the list of tool schemas to pass to Ollama.
    Always includes task_complete regardless of config.enabled_tools.
    """
    schemas = []
    for name in config.enabled_tools:
        if name in _TOOL_SCHEMAS and name != "task_complete":
            schemas.append(_TOOL_SCHEMAS[name])
    # task_complete is always present
    schemas.append(_TOOL_SCHEMAS["task_complete"])
    return schemas
```

---

## 4. `ToolExecutor` class

```python
# ---------------------------------------------------------------------------
# Tool executor
# ---------------------------------------------------------------------------

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
                # task_complete is handled by the ReAct loop, not executed here
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
```

---

## 5. `tests/test_tools.py`

```python
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
```

---

## 6. Verifica finale

```bash
pytest tests/test_tools.py -v
```

Output atteso: tutti i test `PASSED`.

Verifica anche che il modulo sia ancora importabile senza errori dopo le aggiunte:

```bash
python -c "
from ollama_symphony import (
    _safe_path, execute_run_shell, execute_read_file,
    execute_write_file, execute_list_directory,
    build_tool_schemas, ToolExecutor
)
print('OK')
"
```

---

## Criteri di completamento

- [ ] `_safe_path` blocca il path traversal con `ValueError`
- [ ] Tutti e quattro i tool (`run_shell`, `read_file`, `write_file`, `list_directory`) implementati
- [ ] `build_tool_schemas` include sempre `task_complete`
- [ ] `ToolExecutor` gestisce tool sconosciuti senza crash
- [ ] `pytest tests/test_tools.py -v` → tutti PASSED
- [ ] `pytest tests/test_parsing.py -v` → ancora tutti PASSED (nessuna regressione)
