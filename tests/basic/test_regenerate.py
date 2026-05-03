import asyncio
from pathlib import Path
from unittest import mock

import pytest

from cecli.coders import Coder
from cecli.commands import Commands
from cecli.helpers.conversation import ConversationService, MessageTag
from cecli.io import InputOutput
from cecli.models import Model


GPT35 = Model("gpt-3.5-turbo")


async def _create_coder():
    io = InputOutput(pretty=False, fancy_input=False, yes=True)
    coder = await Coder.create(GPT35, None, io)
    return coder, io


@pytest.mark.asyncio
async def test_regenerate_removes_last_assistant():
    coder, io = await _create_coder()
    commands = Commands(io, coder)
    manager = ConversationService.get_manager(coder)

    manager.add_message({"role": "user", "content": "Hello"}, tag=MessageTag.CUR)
    manager.add_message(
        {"role": "assistant", "content": "Hi there"}, tag=MessageTag.CUR
    )
    manager.add_message({"role": "user", "content": "How are you?"}, tag=MessageTag.CUR)
    manager.add_message(
        {"role": "assistant", "content": "I am fine"}, tag=MessageTag.CUR
    )

    with mock.patch.object(io, "tool_output"):
        await commands.execute("regenerate", "")

    cur = manager.get_messages_dict(MessageTag.CUR)
    assert len(cur) == 3
    assert cur[-1]["role"] == "user"
    assert cur[-1]["content"] == "How are you?"
    assert getattr(coder, "_regenerate_next", False) is True


@pytest.mark.asyncio
async def test_regenerate_no_assistant_error():
    coder, io = await _create_coder()
    commands = Commands(io, coder)
    manager = ConversationService.get_manager(coder)

    manager.add_message({"role": "user", "content": "Hello"}, tag=MessageTag.CUR)

    with mock.patch.object(io, "tool_error") as mock_tool_error:
        await commands.execute("regenerate", "")
        mock_tool_error.assert_called_once_with(
            "No assistant response found in the current conversation to regenerate."
        )

    assert getattr(coder, "_regenerate_next", False) is False


@pytest.mark.asyncio
async def test_regenerate_ignores_done_messages():
    coder, io = await _create_coder()
    commands = Commands(io, coder)
    manager = ConversationService.get_manager(coder)

    manager.add_message(
        {"role": "assistant", "content": "Old response"}, tag=MessageTag.DONE
    )
    manager.add_message({"role": "user", "content": "Hello"}, tag=MessageTag.CUR)

    with mock.patch.object(io, "tool_error") as mock_tool_error:
        await commands.execute("regenerate", "")
        mock_tool_error.assert_called_once_with(
            "No assistant response found in the current conversation to regenerate."
        )


@pytest.mark.asyncio
async def test_regenerate_run_one_continuation():
    coder, io = await _create_coder()
    manager = ConversationService.get_manager(coder)

    manager.add_message({"role": "user", "content": "Hello"}, tag=MessageTag.CUR)
    manager.add_message(
        {"role": "assistant", "content": "Hi there"}, tag=MessageTag.CUR
    )

    coder._regenerate_next = True

    with mock.patch.object(coder, "send_message") as mock_send:
        async def mock_async_generator(*args, **kwargs):
            return
            yield

        mock_send.return_value = mock_async_generator()
        await coder.run_one("/regenerate", preproc=True)

    assert getattr(coder, "_regenerate_next", False) is False
    mock_send.assert_called_once()
    call_args = mock_send.call_args
    assert call_args[0][0] is None
