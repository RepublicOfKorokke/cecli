# cecli Agent Guide

cecli (package `cecli-dev`) is an AI pair-programming CLI, originally forked from Aider. It uses `asyncio`, `litellm`, and a custom conversation-management system.

## Entry Points

- `cecli.main:main` — main entry (registered as `cecli`, `aider-ce`, `ce.cli` console scripts)
- `python -m cecli` — alternative launch

## Dev Environment

- **Python**: >=3.10
- **Package manager**: `uv` (used in CI; no Makefile)
- **Install deps**: `uv pip install -r requirements/requirements.in -r requirements/requirements-dev.in`
- **Run tests**: `pytest` (or `pytest tests/basic/test_something.py`)
- **Pre-commit**: `pre-commit run --all-files`
- **Lint/format**: black (100 chars, --preview), isort (--profile black), flake8, codespell

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

## Conversation System

- Never mutate `coder.cur_messages` or `coder.done_messages` directly (they are properties backed by `ConversationManager`)
- Use `ConversationService.get_manager(coder).add_message(...)` / `clear_tag(...)` / `get_messages_dict(...)`
- Message tags: `MessageTag.CUR`, `MessageTag.DONE`, `MessageTag.SYSTEM`, etc.

## Testing

- `pytest` with `asyncio_mode = auto` (configured in `pytest.ini`)
- Tests live under `tests/` with subdirs: `basic`, `coders`, `conversations`, `helpers`, `hooks`, `mcp`, `tools`, etc.
- `CECLI_TUI=false` is set automatically via `pytest.ini`
- Many tests need a git repo and git config (`user.name`, `user.email`, `init.defaultBranch main`)

## Code Style
- Black: 100 char line length, `--preview` enabled
- isort: `--profile black`
- flake8: `--show-source`
- Custom pre-commit hook filters `cecli/resources/model-metadata.json` to chat-mode only

## Custom Commands

Users can load external commands via config (`custom.command-paths`). The loader scans `.py` files for a class named exactly `CustomCommand`.
