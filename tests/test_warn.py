"""The library anchors its warnings at user code, not at itself."""

import warnings
from pathlib import Path

import pytest
from _warnpkg import _warn_fixture

from langgraph_events import _warn


@pytest.fixture
def _fixture_is_library(monkeypatch):
    """Treat the fixture module as if it were library code."""
    monkeypatch.setattr(_warn, "_PACKAGE_ROOT", Path(_warn_fixture.__file__).parent)


def describe_warn_user():

    def when_emitted_one_frame_inside_the_library():

        def it_anchors_at_the_caller(_fixture_is_library):
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                _warn_fixture.shallow(_warn.warn_user)

            assert caught[0].filename == __file__

    def when_emitted_several_frames_inside_the_library():

        # The bug this prevents: a hand-counted stacklevel silently goes
        # stale the moment the emitting code moves into a helper.
        def it_anchors_at_the_same_caller(_fixture_is_library):
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                _warn_fixture.deep(_warn.warn_user)

            assert caught[0].filename == __file__

    def when_given_a_category():

        def it_uses_it(_fixture_is_library):
            with pytest.warns(DeprecationWarning):
                _warn_fixture.shallow(lambda m: _warn.warn_user(m, DeprecationWarning))
