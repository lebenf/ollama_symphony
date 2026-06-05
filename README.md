# Ollama Symphony

A sequential task runner for local LLM models via Ollama.
Reads development tasks from `TASKS.md`, executes them via a ReAct loop
with tool calling, and commits results to git.

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

    # 3. Create config.yml and TASKS.md (see SYMPHONY_SPEC.md)

    # 4. Run
    python ollama_symphony.py

## CLI options

    --tasks FILE      Path to TASKS.md             (default: TASKS.md)
    --config FILE     Path to config.yml            (default: config.yml)
    --workflow FILE   Path to WORKFLOW.md (legacy)  (default: WORKFLOW.md)
    --state FILE      Path to state file            (default: TASKS.state.json)
    --reset           Ignore saved state, restart from task 1
    --dry-run         Parse and log, do not invoke Ollama or git
    --verbose / -v    Enable debug logging
    --list-models     List models available on all configured Ollama hosts
    --check           Validate config and Ollama connectivity, then exit

## Configuration

The runner loads configuration in this order: `config.yml` → `WORKFLOW.md` → built-in defaults.

### config.yml (recommended)

```yaml
agent:
  max_retries: 3
  max_iterations: 20

ollama:
  hosts:
    - http://localhost:11434
  model: qwen2.5-coder:7b
  num_ctx: 16384
  context_window: 16384
  temperature: 0.2

tools:
  shell_timeout_s: 30
  working_dir: "."

system_prompt: |
  You are an autonomous development agent...
```

### WORKFLOW.md (legacy format)

YAML front matter configures the runner; the Markdown body becomes the system prompt.

```
---
ollama:
  model: qwen2.5-coder:14b
  num_ctx: 16384
  context_window: 16384
---
You are an autonomous development agent...
```

See `SYMPHONY_SPEC.md` for the full reference on all parameters and task authoring guidelines.

## Multi-host Ollama

Configure multiple Ollama hosts for round-robin load balancing:

```yaml
ollama:
  hosts:
    - http://gpu1.local:11434
    - http://gpu2.local:11434
    - http://localhost:11434
```

The runner automatically falls back to the next host on failure. If all hosts fail,
the task is retried up to `max_retries` times.

## Recommended models

| Model | Size | Notes |
|---|---|---|
| `qwen2.5-coder:7b` | 4 GB | Default. Good balance of speed and quality |
| `qwen2.5-coder:14b` | 8 GB | Better reasoning, slower |
| `deepseek-coder-v2:16b` | 9 GB | Strong at code generation |
| `llama3.1:8b` | 5 GB | General purpose, good instruction following |

Set the model in `config.yml`:

```yaml
ollama:
  model: qwen2.5-coder:7b
```

## Available tools

| Tool | Enabled by default | Description |
|---|---|---|
| `run_shell` | yes | Execute a shell command in the working directory |
| `read_file` | yes | Read a file relative to the working directory |
| `write_file` | yes | Write or overwrite a file |
| `list_directory` | yes | List entries in a directory |
| `task_complete` | always | Signal task completion (always active) |

Enable or disable tools in `config.yml`:

```yaml
tools:
  enabled:
    - run_shell
    - read_file
    - write_file
    - list_directory
  shell_timeout_s: 60
```
