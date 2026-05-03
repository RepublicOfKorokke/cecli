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

        last_assistant_msg = None
        for msg in reversed(cur_messages):
            if msg.message_dict.get("role") == "assistant":
                last_assistant_msg = msg
                break

        if last_assistant_msg is None:
            io.tool_error("No assistant response found in the current conversation to regenerate.")
            return format_command_result(
                io, "regenerate", "No assistant response found to regenerate"
            )

        removed = manager.remove_message(last_assistant_msg)
        if not removed:
            io.tool_error("Failed to remove the last assistant response.")
            return format_command_result(
                io, "regenerate", "Failed to remove assistant response"
            )

        # Show a preview of the user message being regenerated
        cur_messages = manager.get_tag_messages(MessageTag.CUR)
        last_user_msg = None
        for msg in reversed(cur_messages):
            if msg.message_dict.get("role") == "user":
                last_user_msg = msg
                break

        if last_user_msg:
            content = last_user_msg.message_dict.get("content", "")
            preview = content[:200] + "..." if len(content) > 200 else content
            io.tool_output(f"Regenerating response to: {preview}")
        else:
            io.tool_output("Last assistant response removed. Regenerating...")

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
