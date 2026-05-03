from typing import List

from cecli.commands.utils.base_command import BaseCommand
from cecli.commands.utils.helpers import format_command_result
from cecli.helpers.conversation import ConversationService, MessageTag


class RegenerateCommand(BaseCommand):
    NORM_NAME = "regenerate"
    DESCRIPTION = "Regenerate the last assistant response"

    @classmethod
    async def execute(cls, io, coder, args, **kwargs):
        manager = ConversationService.get_manager(coder)
        cur_messages = manager.get_tag_messages(MessageTag.CUR)

        # Find the last user message by scanning from the end
        last_user_index = None
        for i, msg in enumerate(reversed(cur_messages)):
            if msg.message_dict.get("role") == "user":
                last_user_index = len(cur_messages) - 1 - i
                break

        if last_user_index is None:
            io.tool_error(
                "No user message found in the current conversation to regenerate from."
            )
            return format_command_result(
                io, "regenerate", "No user message found to regenerate from"
            )

        # Remove all messages after the last user message (trailing assistants)
        messages_to_remove = cur_messages[last_user_index + 1 :]
        for msg in messages_to_remove:
            manager.remove_message(msg)

        # Show a preview of the user message being regenerated
        last_user_msg = cur_messages[last_user_index]
        content = last_user_msg.message_dict.get("content", "")
        preview = content[:200] + "..." if len(content) > 200 else content
        io.tool_output(f"Regenerating response to: {preview}")

        coder._regenerate_next = True
        return format_command_result(io, "regenerate", "Regenerating last response")

    @classmethod
    def get_completions(cls, io, coder, args) -> List[str]:
        return []

    @classmethod
    def get_help(cls) -> str:
        help_text = super().get_help()
        help_text += "\nUsage:\n"
        help_text += "  /regenerate  # Remove the last assistant response and regenerate it\n"
        help_text += (
            "\nThis command removes the most recent assistant response from the current\n"
        )
        help_text += (
            "conversation and resends the prompt to the LLM to generate a fresh response.\n"
        )
        help_text += (
            "Note: This does not revert any file changes that may have been made.\n"
        )
        return help_text
