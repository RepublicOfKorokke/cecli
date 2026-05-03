import json
from typing import Any, Dict, List

from cecli.commands.utils.base_command import BaseCommand
from cecli.commands.utils.helpers import format_command_result
from cecli.editor import pipe_editor
from cecli.helpers.conversation import ConversationService, MessageTag


class EditHistoryCommand(BaseCommand):
    NORM_NAME = "edit-history"
    DESCRIPTION = "Edit the conversation history in an external editor"

    @classmethod
    async def execute(cls, io, coder, args, **kwargs):
        editor = kwargs.get("editor") or getattr(coder, "editor", None)

        history_json = cls._get_history_json(coder)
        edited_content = pipe_editor(history_json, suffix="json", editor=editor)

        if edited_content.strip() == history_json.strip():
            io.tool_output("No changes made to history.")
            return format_command_result(io, "edit-history", "No changes made")

        return cls._apply_history_edits(io, coder, edited_content)

    @classmethod
    def _get_history_json(cls, coder) -> str:
        manager = ConversationService.get_manager(coder)

        cur_messages = manager.get_tag_messages(MessageTag.CUR)
        done_messages = manager.get_tag_messages(MessageTag.DONE)

        conversational_messages = []
        for msg in cur_messages + done_messages:
            role = msg.message_dict.get("role")
            if role in ("user", "assistant"):
                conversational_messages.append(msg)

        conversational_messages.sort(key=lambda m: m.timestamp)

        history = [
            {
                "role": msg.message_dict["role"],
                "content": msg.message_dict.get("content") or "",
            }
            for msg in conversational_messages
        ]

        return json.dumps(history, indent=2, ensure_ascii=False)

    @classmethod
    def _apply_history_edits(cls, io, coder, edited_content: str):
        try:
            edited_history = json.loads(edited_content)
        except json.JSONDecodeError as e:
            io.tool_error(f"Invalid JSON: {e}")
            return format_command_result(io, "edit-history", f"Invalid JSON: {e}")

        if not isinstance(edited_history, list):
            io.tool_error("History must be a JSON array of message objects.")
            return format_command_result(io, "edit-history", "History must be a JSON array")

        validation_error = cls._validate_history(edited_history)
        if validation_error:
            io.tool_error(f"Validation error: {validation_error}")
            return format_command_result(
                io, "edit-history", f"Validation error: {validation_error}"
            )

        manager = ConversationService.get_manager(coder)
        manager.clear_tag(MessageTag.CUR)
        manager.clear_tag(MessageTag.DONE)

        for msg in edited_history:
            manager.add_message(
                message_dict={"role": msg["role"], "content": msg["content"]},
                tag=MessageTag.CUR,
            )

        io.tool_output(f"History updated with {len(edited_history)} messages.")
        return format_command_result(
            io, "edit-history", f"History updated with {len(edited_history)} messages"
        )

    @classmethod
    def _validate_history(cls, history: List[Dict[str, Any]]) -> str:
        for i, msg in enumerate(history):
            if not isinstance(msg, dict):
                return f"Message at index {i} is not an object."

            if "role" not in msg:
                return f"Message at index {i} is missing 'role'."
            if "content" not in msg:
                return f"Message at index {i} is missing 'content'."

            if msg["role"] not in ("user", "assistant"):
                return (
                    f"Message at index {i} has invalid role '{msg['role']}'. "
                    "Must be 'user' or 'assistant'."
                )

            if not isinstance(msg["content"], str):
                return f"Message at index {i} has 'content' that is not a string."

        return ""

    @classmethod
    def get_completions(cls, io, coder, args) -> List[str]:
        return []

    @classmethod
    def get_help(cls) -> str:
        help_text = super().get_help()
        help_text += "\nUsage:\n"
        help_text += "  /edit-history  # Open the conversation history in an editor\n"
        help_text += (
            "\nThis command opens the session's conversation history in your system's default\n"
        )
        help_text += (
            "text editor (or the editor specified by the EDITOR environment variable) as a\n"
        )
        help_text += (
            "JSON file. You can review, modify, and save the history. The session will then\n"
        )
        help_text += "continue from the edited state.\n"
        help_text += "\nThe JSON format is an array of objects with 'role' and 'content' fields:\n"
        help_text += (
            '  [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]\n'
        )
        return help_text
