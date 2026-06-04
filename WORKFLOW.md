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
