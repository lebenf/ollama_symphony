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

Key differences:

| Feature | symphony.py | ollama_symphony.py |
|---|---|---|
| Model | Claude (Anthropic API) | Any Ollama model |
| Tool execution | Claude Code built-ins | Local Python handlers |
| Cost | API credits | Local compute |
| State format | TASKS.state.json | Same — compatible |

## Multi-host Ollama

Configure multiple Ollama hosts in `WORKFLOW.md` front matter:

```yaml
ollama_hosts:
  - http://gpu1.local:11434
  - http://gpu2.local:11434
  - http://localhost:11434
```

The runner uses round-robin across hosts and automatically falls back to the
next host if a request fails. If all hosts fail, the task is retried up to
`max_retries` times.

## Modelli consigliati

| Model | Size | Notes |
|---|---|---|
| `qwen2.5-coder:7b` | 4 GB | Default. Good balance of speed and quality |
| `qwen2.5-coder:14b` | 8 GB | Better reasoning, slower |
| `deepseek-coder-v2:16b` | 9 GB | Strong at code generation |
| `llama3.1:8b` | 5 GB | General purpose, good instruction following |
| `codellama:13b` | 8 GB | Specialized for code, older architecture |

Set the model in `WORKFLOW.md`:

```yaml
ollama_model: qwen2.5-coder:7b
```

## Tool disponibili

| Tool | Enabled by default | Description |
|---|---|---|
| `run_shell` | yes | Execute a shell command in the working directory |
| `read_file` | yes | Read a file relative to the working directory |
| `write_file` | yes | Write or overwrite a file |
| `list_directory` | yes | List entries in a directory |
| `task_complete` | always | Signal task completion with a summary (always active) |

Enable or disable tools in `WORKFLOW.md`:

```yaml
tools:
  enabled:
    - run_shell
    - read_file
    - write_file
    - list_directory
  shell_timeout_s: 60
```
