from typing import List

from cecli.commands.utils.base_command import BaseCommand
from cecli.commands.utils.helpers import format_command_result
from cecli.repo import ANY_GIT_ERROR


class CommitStagedCommand(BaseCommand):
    NORM_NAME = "commit-staged"
    DESCRIPTION = "Commit only explicitly staged changes (commit message optional)"

    @classmethod
    async def execute(cls, io, coder, args, **kwargs):
        """Execute the commit-staged command with given parameters."""
        try:
            return await cls._raw_cmd_commit_staged(io, coder, args)
        except ANY_GIT_ERROR as err:
            io.tool_error(f"Unable to complete commit-staged: {err}")
            return format_command_result(io, "commit-staged", f"Unable to complete commit-staged: {err}", err)

    @classmethod
    async def _raw_cmd_commit_staged(cls, io, coder, args):
        """Raw commit-staged implementation without error handling."""
        if not coder.repo:
            io.tool_error("No git repository found.")
            return format_command_result(io, "commit-staged", "No git repository found")

        if not coder.repo.has_staged_changes():
            io.tool_warning("No staged changes to commit.")
            return format_command_result(io, "commit-staged", "No staged changes to commit")

        commit_message = args.strip() if args else None
        await coder.repo.commit(message=commit_message, coder=coder, staged_only=True)
        return format_command_result(io, "commit-staged", "Staged changes committed successfully")

    @classmethod
    def get_completions(cls, io, coder, args) -> List[str]:
        """Get completion options for commit-staged command."""
        return []

    @classmethod
    def get_help(cls) -> str:
        """Get help text for commit-staged command."""
        help_text = super().get_help()
        help_text += "\nUsage:\n"
        help_text += "  /commit-staged              # Commit staged changes with auto-generated message\n"
        help_text += "  /commit-staged <message>    # Commit staged changes with specific message\n"
        help_text += "\nThis command commits only the changes that have been explicitly staged\n"
        help_text += "in the git index (e.g., via `git add`). Unstaged changes are left untouched.\n"
        help_text += "\nNote: This only commits changes made outside the chat session.\n"
        help_text += "Changes made by cecli during the chat are automatically committed.\n"
        return help_text
