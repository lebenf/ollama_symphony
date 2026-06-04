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
