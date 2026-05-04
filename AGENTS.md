# cecli Agent Guide

cecli (package `cecli-dev`) is an AI pair-programming CLI, originally forked from Aider. It uses `asyncio`, `litellm`, and a custom conversation-management system.

## Entry Points

- `cecli.main:main` — main entry (registered as `cecli`, `aider-ce`, `ce.cli` console scripts)
- `python -m cecli` — alternative launch

## Dev Environment

- **Python**: >=3.10 (CI tests 3.10-3.14)
- **Package manager**: `uv` (used in CI; no Makefile)
- **Install deps**: `uv pip install -r requirements/requirements.in -r requirements/requirements-dev.in`
- **Run tests**: `pytest` (or `pytest tests/basic/test_something.py`)
- **Pre-commit**: `pre-commit run --all-files`
- **Lint/format**: black (100 chars, --preview), isort (--profile black), flake8 --show-source, codespell

## Architecture

- `cecli/coders/base_coder.py` — `Coder` class, core async loop (`run()`, `generate()`, `send_message()`)
- `cecli/commands/` — CLI commands; each extends `BaseCommand` with `NORM_NAME`, `DESCRIPTION`, and async `execute()`
- `cecli/helpers/conversation/` — `ConversationManager` (via `ConversationService`) is the single source of truth for message history. Messages carry tags: `CUR`, `DONE`, `SYSTEM`, `DIFFS`, etc.
- `cecli/models.py` — Model config and litellm integration

## Adding a Command

1. Create a class in `cecli/commands/<name>.py` extending `BaseCommand`
2. Define `NORM_NAME` (kebab-case) and `DESCRIPTION`
3. Implement async `classmethod execute(cls, io, coder, args, **kwargs)`
4. Register it in `cecli/commands/__init__.py` (`CommandRegistry.register(...)`)

**Important**: Class names MUST end with `Command` (enforced by `CommandMeta` metaclass). `NORM_NAME`, `DESCRIPTION`, and `execute` are all required.

## Conversation System

- Never mutate `coder.cur_messages` or `coder.done_messages` directly (they are properties backed by `ConversationManager`)
- Use `ConversationService.get_manager(coder).add_message(...)` / `clear_tag(...)` / `get_messages_dict(...)`
- Message tags: `MessageTag.CUR`, `MessageTag.DONE`, `MessageTag.SYSTEM`, etc.

## Testing

- `pytest` with `asyncio_mode = auto` (configured in `pytest.ini`)
- Test paths: `tests/basic`, `tests/tools`, `tests/coders`, `tests/conversations`, `tests/helpers`, `tests/hooks`, `tests/mcp`, `tests/help`, `tests/browser`, `tests/scrape`
- `CECLI_TUI=false` is set automatically via `pytest.ini`
- Many tests need a git repo and git config:
  ```
  git config --global user.name "Test User"
  git config --global user.email "test@example.com"
  git config --global init.defaultBranch main
  ```

## Code Style
- Black: 100 char line length, `--preview` enabled
- isort: `--profile black`
- flake8: `--show-source`
- Custom pre-commit hook filters `cecli/resources/model-metadata.json` to chat-mode only

## Sessions

- Auto-save is enabled via `--auto-save` and triggered after every user-LLM turn.
- Session files are stored in `.cecli/sessions/` as JSON.
- **Naming**: After the first turn, the weak model generates a 5-10 word summary in the background. The session filename becomes `{timestamp}_{summary}.json` (e.g. `20260504_132340_fix-login-bug.json`). The name is computed **once** and never changes.
- **Before summary is ready**: The fallback filename is just the session start timestamp (e.g. `20260504_132340.json`).
- Key files: `cecli/sessions.py` (save/load), `cecli/coders/base_coder.py` (auto-save trigger and naming logic), `cecli/history.py` (`ChatSummary` for weak-model calls).

## Custom Commands

Users can load external commands via config (`custom.command-paths`). The loader scans `.py` files for a class named exactly `CustomCommand`.
