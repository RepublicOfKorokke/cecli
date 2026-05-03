import json
from unittest import mock

import pytest

from cecli.coders import Coder
from cecli.commands.edit_history import EditHistoryCommand
from cecli.helpers.conversation import ConversationService, MessageTag
from cecli.io import InputOutput
from cecli.models import Model


GPT35 = Model("gpt-3.5-turbo")


async def _create_coder():
    io = InputOutput(pretty=False, fancy_input=False, yes=True)
    coder = await Coder.create(GPT35, None, io)
    return coder, io


@pytest.mark.asyncio
async def test_get_history_json():
    coder, io = await _create_coder()
    manager = ConversationService.get_manager(coder)

    manager.add_message(
        {"role": "assistant", "content": "Old response"}, tag=MessageTag.DONE, timestamp=100
    )
    manager.add_message({"role": "user", "content": "Previous"}, tag=MessageTag.DONE, timestamp=200)
    manager.add_message({"role": "system", "content": "You are helpful"}, tag=MessageTag.CUR, timestamp=300)
    manager.add_message({"role": "user", "content": "Hello"}, tag=MessageTag.CUR, timestamp=400)
    manager.add_message(
        {"role": "assistant", "content": "Hi there"}, tag=MessageTag.CUR, timestamp=500
    )

    history_json = EditHistoryCommand._get_history_json(coder)
    history = json.loads(history_json)

    assert isinstance(history, list)
    assert len(history) == 4

    roles = [msg["role"] for msg in history]
    assert roles == ["assistant", "user", "user", "assistant"]

    contents = [msg["content"] for msg in history]
    assert contents == ["Old response", "Previous", "Hello", "Hi there"]


@pytest.mark.asyncio
async def test_get_history_json_empty():
    coder, io = await _create_coder()
    history_json = EditHistoryCommand._get_history_json(coder)
    history = json.loads(history_json)
    assert history == []


@pytest.mark.asyncio
async def test_apply_history_edits_success():
    coder, io = await _create_coder()
    manager = ConversationService.get_manager(coder)

    manager.add_message({"role": "user", "content": "Hello"}, tag=MessageTag.CUR)
    manager.add_message(
        {"role": "assistant", "content": "Hi there"}, tag=MessageTag.CUR
    )

    edited = json.dumps(
        [{"role": "user", "content": "New prompt"}],
        indent=2,
        ensure_ascii=False,
    )

    with mock.patch.object(io, "tool_output"):
        result = EditHistoryCommand._apply_history_edits(io, coder, edited)

    cur = manager.get_messages_dict(MessageTag.CUR)
    done = manager.get_messages_dict(MessageTag.DONE)

    assert len(cur) == 1
    assert len(done) == 0
    assert cur[0]["role"] == "user"
    assert cur[0]["content"] == "New prompt"


@pytest.mark.asyncio
async def test_apply_history_edits_invalid_json():
    coder, io = await _create_coder()

    with mock.patch.object(io, "tool_error") as mock_tool_error:
        result = EditHistoryCommand._apply_history_edits(io, coder, "not json")
        mock_tool_error.assert_called_once()
        assert "Invalid JSON" in mock_tool_error.call_args[0][0]


@pytest.mark.asyncio
async def test_apply_history_edits_not_a_list():
    coder, io = await _create_coder()

    with mock.patch.object(io, "tool_error") as mock_tool_error:
        result = EditHistoryCommand._apply_history_edits(io, coder, '{"role": "user"}')
        mock_tool_error.assert_called_once_with(
            "History must be a JSON array of message objects."
        )


@pytest.mark.asyncio
async def test_validate_history_missing_role():
    error = EditHistoryCommand._validate_history([{"content": "hello"}])
    assert "missing 'role'" in error


@pytest.mark.asyncio
async def test_validate_history_missing_content():
    error = EditHistoryCommand._validate_history([{"role": "user"}])
    assert "missing 'content'" in error


@pytest.mark.asyncio
async def test_validate_history_invalid_role():
    error = EditHistoryCommand._validate_history([{"role": "system", "content": "hello"}])
    assert "invalid role" in error


@pytest.mark.asyncio
async def test_validate_history_non_string_content():
    error = EditHistoryCommand._validate_history([{"role": "user", "content": 123}])
    assert "not a string" in error


@pytest.mark.asyncio
async def test_validate_history_valid():
    error = EditHistoryCommand._validate_history(
        [{"role": "user", "content": "hello"}, {"role": "assistant", "content": "hi"}]
    )
    assert error == ""


@pytest.mark.asyncio
async def test_execute_no_changes():
    coder, io = await _create_coder()

    history_json = EditHistoryCommand._get_history_json(coder)

    with mock.patch(
        "cecli.commands.edit_history.pipe_editor", return_value=history_json
    ), mock.patch.object(io, "tool_output") as mock_tool_output:
        result = await EditHistoryCommand.execute(io, coder, "")
        mock_tool_output.assert_any_call("No changes made to history.")
