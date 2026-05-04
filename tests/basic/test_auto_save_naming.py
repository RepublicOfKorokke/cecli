import re
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from cecli.coders import Coder
from cecli.helpers.conversation import ConversationService, MessageTag
from cecli.io import InputOutput
from cecli.models import Model
from cecli.utils import GitTemporaryDirectory


@pytest.fixture
def mock_args():
    return SimpleNamespace(
        auto_save=True,
        auto_save_session_name="auto-save",
    )


async def test_auto_save_name_fallback_before_first_turn(mock_args):
    """Before any assistant reply, the timestamp fallback name is used."""
    with GitTemporaryDirectory():
        io = InputOutput(pretty=False, fancy_input=False, yes=True)
        model = Model("gpt-3.5-turbo")
        coder = await Coder.create(model, None, io, args=mock_args)
        name = coder._get_auto_save_session_name()
        assert re.match(r"^\d{8}_\d{6}$", name), f"Unexpected fallback name: {name}"


async def test_auto_save_name_computed_after_first_turn(mock_args):
    """After the first assistant reply, the name contains a timestamp and summary."""
    with GitTemporaryDirectory():
        io = InputOutput(pretty=False, fancy_input=False, yes=True)
        model = Model("gpt-3.5-turbo")
        coder = await Coder.create(model, None, io, args=mock_args)

        # Mock summarizer to return a known summary
        coder.summarizer.summarize_all_as_text = AsyncMock(return_value="fix login bug")

        manager = ConversationService.get_manager(coder)
        manager.add_message(
            {"role": "user", "content": "Help me fix the login bug"},
            tag=MessageTag.CUR,
        )
        manager.add_message(
            {"role": "assistant", "content": "Sure, let's fix it."},
            tag=MessageTag.CUR,
        )

        await coder._compute_auto_save_session_name()

        name = coder._get_auto_save_session_name()
        assert re.match(r"^\d{8}_\d{6}_[a-z0-9-]+$", name), f"Unexpected name: {name}"
        assert name.endswith("_fix-login-bug")


async def test_auto_save_name_caches_after_first_call(mock_args):
    """The summarizer is only called once; subsequent calls reuse the cached name."""
    with GitTemporaryDirectory():
        io = InputOutput(pretty=False, fancy_input=False, yes=True)
        model = Model("gpt-3.5-turbo")
        coder = await Coder.create(model, None, io, args=mock_args)

        coder.summarizer.summarize_all_as_text = AsyncMock(return_value="refactor auth")

        manager = ConversationService.get_manager(coder)
        manager.add_message({"role": "user", "content": "Refactor auth"}, tag=MessageTag.CUR)
        manager.add_message({"role": "assistant", "content": "Done."}, tag=MessageTag.CUR)

        await coder._compute_auto_save_session_name()
        first_name = coder._get_auto_save_session_name()

        # Second call should not invoke the summarizer again
        await coder._compute_auto_save_session_name()
        second_name = coder._get_auto_save_session_name()

        assert first_name == second_name
        coder.summarizer.summarize_all_as_text.assert_awaited_once()


async def test_auto_save_name_stable_across_turns(mock_args):
    """Adding more messages after the first turn does not change the saved name."""
    with GitTemporaryDirectory():
        io = InputOutput(pretty=False, fancy_input=False, yes=True)
        model = Model("gpt-3.5-turbo")
        coder = await Coder.create(model, None, io, args=mock_args)

        coder.summarizer.summarize_all_as_text = AsyncMock(return_value="initial task")

        manager = ConversationService.get_manager(coder)
        manager.add_message({"role": "user", "content": "First task"}, tag=MessageTag.CUR)
        manager.add_message({"role": "assistant", "content": "Completed."}, tag=MessageTag.CUR)

        await coder._compute_auto_save_session_name()
        first_name = coder._get_auto_save_session_name()

        # Simulate a second turn
        manager.add_message({"role": "user", "content": "Second task"}, tag=MessageTag.CUR)
        manager.add_message({"role": "assistant", "content": "Also done."}, tag=MessageTag.CUR)

        await coder._compute_auto_save_session_name()
        second_name = coder._get_auto_save_session_name()

        assert first_name == second_name
        coder.summarizer.summarize_all_as_text.assert_awaited_once()


async def test_auto_save_name_sanitizes_summary(mock_args):
    """Special characters in the summary are sanitized to safe filename components."""
    with GitTemporaryDirectory():
        io = InputOutput(pretty=False, fancy_input=False, yes=True)
        model = Model("gpt-3.5-turbo")
        coder = await Coder.create(model, None, io, args=mock_args)

        coder.summarizer.summarize_all_as_text = AsyncMock(return_value="Fix: the Bug! (now)")

        manager = ConversationService.get_manager(coder)
        manager.add_message({"role": "user", "content": "Fix bug"}, tag=MessageTag.CUR)
        manager.add_message({"role": "assistant", "content": "OK."}, tag=MessageTag.CUR)

        await coder._compute_auto_save_session_name()
        name = coder._get_auto_save_session_name()
        assert re.match(r"^\d{8}_\d{6}_[a-z0-9-]+$", name), f"Unexpected name: {name}"
        assert name.endswith("_fix-the-bug-now")


async def test_auto_save_name_empty_summary_fallback(mock_args):
    """If the summarizer returns nothing, the summary component falls back to 'chat'."""
    with GitTemporaryDirectory():
        io = InputOutput(pretty=False, fancy_input=False, yes=True)
        model = Model("gpt-3.5-turbo")
        coder = await Coder.create(model, None, io, args=mock_args)

        coder.summarizer.summarize_all_as_text = AsyncMock(return_value="")

        manager = ConversationService.get_manager(coder)
        manager.add_message({"role": "user", "content": "Hi"}, tag=MessageTag.CUR)
        manager.add_message({"role": "assistant", "content": "Hello."}, tag=MessageTag.CUR)

        await coder._compute_auto_save_session_name()
        name = coder._get_auto_save_session_name()
        assert re.match(r"^\d{8}_\d{6}_[a-z0-9-]+$", name), f"Unexpected name: {name}"
        assert name.endswith("_chat")


async def test_get_session_file_path_rejects_empty_name(mock_args):
    """get_session_file_path raises ValueError for empty session names."""
    with GitTemporaryDirectory():
        io = InputOutput(pretty=False, fancy_input=False, yes=True)
        model = Model("gpt-3.5-turbo")
        coder = await Coder.create(model, None, io, args=mock_args)
        from cecli.sessions import SessionManager

        sm = SessionManager(coder, io)
        with pytest.raises(ValueError, match="Session name cannot be empty"):
            sm.get_session_file_path("")


async def test_fallback_file_removed_after_computed_save(mock_args):
    """After a successful computed-name save, the timestamp fallback is deleted."""
    with GitTemporaryDirectory():
        io = InputOutput(pretty=False, fancy_input=False, yes=True)
        model = Model("gpt-3.5-turbo")
        coder = await Coder.create(model, None, io, args=mock_args)

        # Compute the name first
        coder.summarizer.summarize_all_as_text = AsyncMock(return_value="task done")
        manager = ConversationService.get_manager(coder)
        manager.add_message({"role": "user", "content": "Do task"}, tag=MessageTag.CUR)
        manager.add_message({"role": "assistant", "content": "Done."}, tag=MessageTag.CUR)
        await coder._compute_auto_save_session_name()

        computed_name = coder._get_auto_save_session_name()
        ts = coder._session_start_time.strftime("%Y%m%d_%H%M%S")

        from cecli.sessions import SessionManager

        sm = SessionManager(coder, io)
        fallback = sm.get_session_file_path(ts)
        computed = sm.get_session_file_path(computed_name)

        # Create the fallback file on disk
        fallback.parent.mkdir(parents=True, exist_ok=True)
        fallback.write_text("{}", encoding="utf-8")
        assert fallback.exists()

        # Trigger the save-and-cleanup
        coder._do_save_and_cleanup(computed_name)

        assert computed.exists(), "Computed-name session file should exist"
        assert not fallback.exists(), "Fallback file should be deleted after computed save"
