# Ollama Symphony — Guide to writing TASKS.md and WORKFLOW.md

This document specifies how to write `TASKS.md` and `WORKFLOW.md` (or `config.yml`)
for the **Ollama Symphony** runner (`ollama_symphony.py`).

Attach it to a project when asking an agent or collaborator to plan development work:
whoever writes the tasks must follow these rules to ensure the runner executes them correctly.

---

## How the runner works (brief overview)

`ollama_symphony.py` reads `TASKS.md` and executes each task in order via a
**ReAct loop** (Reason + Act) with a local Ollama model. For each task the runner:

1. Builds a prompt from the system prompt, a summary of already-completed tasks,
   and the body of the current task.
2. Sends the prompt to the Ollama model, which responds with tool calls (`tool_calls`).
3. Executes the requested tools (`run_shell`, `read_file`, `write_file`, `list_directory`)
   and feeds the results back to the model.
4. Repeats until the model calls `task_complete` or `max_iterations` is reached.
5. On success, creates a git commit.

**Differences from Claude Code Symphony:**

- The model has no direct filesystem or shell access: it can only act via the four
  available tools.
- Context is limited by `num_ctx` (typically 8 192 – 32 768 tokens). When the
  conversation exceeds 85 % of the context window, intermediate messages are truncated:
  the model loses the history of steps already taken.
- Local models (7B–14B) are less capable than Claude: ambiguous or oversized tasks
  produce infinite loops or incorrect results.
- The model must explicitly call `task_complete` to signal completion; without that
  call the task fails with `max_iterations`.

Practical implications for task authors:

- Each task must be **self-contained and small**: the model must be able to complete
  it in a few tool cycles without losing context.
- Do not assume the model "remembers" decisions from previous tasks unless they are
  already in the committed code.
- Explicit verification instructions (e.g. `pytest`) are required: the model does not
  infer on its own when the work is done.

---

## TASKS.md

### Format

```markdown
# Optional project title

<!-- free text ignored by the runner -->

## <Task 1 title>

<task body>

## <Task 2 title>

<task body>
```

- The file is standard Markdown.
- Each `##` heading (level 2) opens a new task.
- The task body is all text until the next `##` or end of file.
- Headings of other levels (`#`, `###`, etc.) and text before the first `##` are
  ignored by the runner and can be used freely as comments or documentation.

### Task title

The title is the text after `## `. It is used:

- as the git commit message (`feat: <title>`, configurable)
- in runner logs
- in the completed-tasks summary injected into the prompt

**Rules:**

- Use a short, descriptive title that is understandable out of context.
- Prefer the form `Verb + object`: *Implement authentication*, *Add POST /users endpoint*.
- Avoid generic titles like *Fix*, *Update*, *Misc changes*.
- Keep it under ~80 characters.

**Good examples:**

```
## Create User model with email validation
## Add REST endpoint GET /users/{id}
## Write integration tests for AuthService
## Refactoring: extract LoggerFactory into separate module
```

**Examples to avoid:**

```
## Fix
## Task 1
## Changes to backend and frontend and tests and configuration
```

### Task body

The body is the prompt sent to the Ollama model. It must contain instructions
sufficient to complete the work without human interaction, and precise enough
that the model does not need to make architectural decisions on its own.

**Rules:**

1. **Describe the expected result**, not just the intent.
   Instead of *"improve error handling"*, write
   *"in `src/api.py`, every unhandled exception must be caught and returned
   as a JSON response `{error: <message>}` with status 500"*.

2. **Specify the files involved** when you know them.
   *"Modify `src/models.py` and add tests in `tests/test_models.py`"*
   is more precise than *"add the tests"*.

3. **Always include test instructions and a completion criterion.**
   The runner waits for a `task_complete` call: the model only makes that call
   when tests pass and it considers the work done. Write explicitly which command
   to run and when to consider the task complete.
   Example: *"Run `pytest tests/test_foo.py` and make sure all tests pass."*

4. **One responsibility per task.**
   Each task must do one thing only. If a task covers two distinct areas
   (e.g. implement the functions *and* write mock tests), split it into two tasks.
   With 7B–14B models, separating implementation from tests reduces the risk of loops.

5. **Avoid implicit dependencies on future tasks.**
   If task A produces an interface that task B will consume, describe that interface
   in task A (or in an architecture document already in the repository).

6. **Do not include credentials, tokens, or secrets.**
   Reference environment variables or configuration files already present in the repo.

7. **Markdown is welcome** for readability (lists, inline code, code blocks).

### Tools available to the model

The model can only use these tools during each task:

| Tool | Purpose |
|---|---|
| `run_shell` | Execute a shell command (timeout set by `shell_timeout_s`) |
| `read_file` | Read a file (truncated at 8 000 characters) |
| `write_file` | Write or overwrite a file |
| `list_directory` | List the contents of a directory |
| `task_complete` | Signal that the task completed successfully |

The model has no advanced search, diff, or partial-edit tools: `write_file` overwrites
the entire file. Keep this in mind when deciding how many files a task should touch.

### Size and granularity

| Signal | Suggestion |
|---|---|
| Body is fewer than 2 lines | Probably too vague: add details or acceptance criteria |
| Body exceeds ~25 lines | Risk of context loop: consider splitting |
| Task touches more than 2–3 unrelated files | Split into smaller tasks |
| Task includes both complex implementation and mock tests | Split into two tasks |
| Task requires architectural decisions not yet made | Add a preliminary design task, or document decisions in the repo before running |

The ~25-line limit is stricter than Claude Code Symphony because each ReAct iteration
consumes context, and with `num_ctx: 8192` the model can do roughly 10–15 useful cycles
before context is truncated. With `num_ctx: 16384` or higher the margin increases,
but the rule still applies.

### Task order

Order tasks so that each one can be executed with the code produced by previous tasks
already in the repository. Practical ordering:

1. Project setup and structure
2. Domain models and types
3. Business logic (functions, services)
4. Persistence or I/O layer
5. Unit tests for already-implemented modules
6. APIs / external interfaces
7. End-to-end integration tests
8. Refactoring, optimisation, documentation

### Complete example

```markdown
# User management service

## Create project structure and dependencies

Initialise the Python project:
- Create `src/__init__.py`, `src/models.py`, `src/services.py`, `src/api.py`
- Create `tests/__init__.py`
- Create `requirements.txt` with: `fastapi`, `pydantic`, `pytest`, `httpx`

Run `pytest tests/` (no tests yet; it must simply produce no import errors).

## Define the User model

In `src/models.py`, define a Pydantic `User` model with fields:
- `id: UUID` (auto-generated)
- `name: str` (non-empty)
- `email: str` (must contain `@`)

Verify the module is importable: `python -c "from src.models import User; print('ok')"`.

## Write tests for the User model

Read `src/models.py`, then in `tests/test_models.py` add:
- a test that verifies correct creation of a valid User
- a test that verifies a missing `@` in email raises `ValidationError`

Run `pytest tests/test_models.py` and make sure they pass.

## Implement UserService with in-memory store

In `src/services.py`, implement `UserService` with methods:
- `create(name, email) -> User` — create and store a user
- `get(user_id) -> User | None` — return the user or None
- `list() -> list[User]` — return all users

The store is an in-memory dictionary (`dict[UUID, User]`).
Verify the module is importable: `python -c "from src.services import UserService; print('ok')"`.

## Write tests for UserService

Read `src/services.py`, then in `tests/test_services.py` add tests for each method.
Run `pytest tests/test_services.py` and make sure they pass.
```

---

## Configuration: config.yml and WORKFLOW.md

The runner looks for configuration in this order:

1. `config.yml` (plain YAML, takes priority)
2. `WORKFLOW.md` (YAML front matter + system prompt in Markdown)
3. Built-in defaults if neither file is present

### config.yml (recommended format)

```yaml
agent:
  max_retries: 3          # retries per task on failure (default: 3)
  retry_delay_s: 10       # seconds to wait between retries (default: 10)
  turn_timeout_s: 600     # timeout for a single ReAct iteration (default: 600)
  max_iterations: 20      # max ReAct iterations per task (default: 20)

git:
  commit_after_each_task: true
  commit_message_template: "feat: {task_title}"   # placeholders: {task_title}, {task_index}

ollama:
  hosts:
    - http://localhost:11434  # multiple hosts supported with round-robin
  model: qwen2.5-coder:7b
  timeout_s: 120          # HTTP timeout for a single model call
  temperature: 0.2
  context_window: 8192    # used internally for truncation estimation
  num_ctx: 8192           # effective context window passed to the model

tools:
  enabled:
    - run_shell
    - read_file
    - write_file
    - list_directory
  shell_timeout_s: 30     # timeout for each run_shell command
  working_dir: "."        # working directory for all tools

system_prompt: |
  You are an autonomous development agent...
```

**Notes on `num_ctx`:**

- `8192` is the practical minimum; with medium-complexity tasks context gets truncated
  around iteration 12–15.
- `16384` is a good compromise for standard tasks.
- Higher values require more VRAM; verify the model supports the chosen size.
- Keep `context_window` and `num_ctx` in sync.

**Notes on `max_iterations`:**

- `20` is the default; tasks with many test-fix-retest cycles may need 30–40.
- Increase `max_iterations` together with `num_ctx`: extra iterations are useless if
  context is truncated and the model loses track of what it was doing.

### WORKFLOW.md (legacy compatible format)

```
---
agent:
  max_retries: 2
  max_iterations: 30

git:
  commit_message_template: "feat({task_index}): {task_title}"

ollama:
  model: qwen2.5-coder:14b
  num_ctx: 16384
  context_window: 16384

tools:
  shell_timeout_s: 60
---
You are an autonomous development agent working on a Python 3.11 project.

Technical constraints:
- Use `pytest` as the test runner; always run tests before completing a task.
- Production code goes in `src/`, tests in `tests/`.
- Use type hints everywhere.
```

The Markdown body after `---` becomes the system prompt. The YAML section supports
the same parameters as `config.yml` (except `system_prompt` as a key, which in
WORKFLOW.md is the file body).

### Interpreter and tool paths

The runner executes `run_shell` commands by inheriting the environment of the
process that launched it. If the project has its own virtualenv (`.venv/`) and
the runner is started from a different virtualenv, `python`, `pip`, and `pytest`
will resolve to the wrong interpreter — causing `ModuleNotFoundError` for
project-specific packages.

**Always specify explicit paths in the system prompt** when the project has its
own virtualenv:

```
- Use `.venv/bin/python` (not `python`) for any direct Python invocation.
- Use `.venv/bin/pip` (not `pip`) to install packages.
- Use `.venv/bin/pytest` (not `pytest`) to run tests.
```

If the project has no dedicated virtualenv and relies on the system or runner
environment, plain `python` / `pip` / `pytest` are fine — but document that
assumption explicitly so it is not lost when the project is moved.

### Guidelines for the system prompt

The system prompt precedes every task and is the right place for:

- **Technical constraints** that apply to the whole project
  (e.g. language, Python version, framework, code style)
- **Interpreter and tool paths** — always specify `.venv/bin/python`,
  `.venv/bin/pip`, `.venv/bin/pytest` when the project has its own virtualenv
  (see section above)
- **Conventions** the model must follow in every task
  (e.g. where to put tests, how to name files, how to handle errors)
- **Project context** not derivable from the code
  (e.g. relevant table schemas, external filesystem structure)
- **Behavioural instructions**
  (e.g. "do not modify files outside `src/` and `tests/`")

**Do not put in the system prompt:**

- Instructions specific to a single task (put them in the task body in `TASKS.md`)
- Credentials or secrets
- Instructions that assume tools not available to the model
  (e.g. "search the codebase with ripgrep" — the model has no native `grep`,
  it must use `run_shell`)

---

## Checklist before delivery

Before delivering `TASKS.md` (and `config.yml` / `WORKFLOW.md`) verify:

- [ ] Every task has a clear, descriptive title
- [ ] Every task specifies the files to create or modify (when known)
- [ ] Every task includes explicit instructions on how to run tests and when to consider them passed
- [ ] Any task combining complex implementation with mock tests is split into two tasks
- [ ] No task body exceeds ~25 lines (with `num_ctx: 8192`) or ~40 lines (with `num_ctx: 16384`)
- [ ] Tasks are ordered so each can run with the code from previous tasks already committed
- [ ] No task is ambiguous enough to require an undocumented architectural decision
- [ ] `num_ctx` and `context_window` are in sync in the config
- [ ] `max_iterations` is sized for task complexity (≥ 25 for tasks with test-fix cycles)
- [ ] The system prompt specifies explicit interpreter paths (`.venv/bin/python`, `.venv/bin/pip`,
  `.venv/bin/pytest`) if the project has its own virtualenv
- [ ] The system prompt describes constraints valid for the whole project, not a single task
- [ ] No secrets or credentials are present in any file
