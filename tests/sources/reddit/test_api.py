import json
from types import SimpleNamespace
from typing import Any

import pandas as pd
import pytest

from naturalv2.sources.reddit import api


class DummyLimiter:
    """Async context manager that records usage without doing any work."""

    def __init__(self) -> None:
        self.entered = 0
        self.exited = 0

    async def __aenter__(self) -> "DummyLimiter":  # noqa: D401
        self.entered += 1
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:  # noqa: D401
        self.exited += 1


class _AsyncIter:
    """Simple async iterator helper for predictable iteration in tests."""

    def __init__(self, items: list[Any]) -> None:
        self._items = items
        self._idx = 0

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self._idx >= len(self._items):
            raise StopAsyncIteration
        item = self._items[self._idx]
        self._idx += 1
        return item


def _make_response_exception(status: int) -> BaseException:
    """Build a minimal asyncprawcore-style ResponseException instance."""
    asyncprawcore = pytest.importorskip("asyncprawcore")

    class FakeResponseException(asyncprawcore.exceptions.ResponseException):
        def __init__(self, status_code: int):
            self.response = SimpleNamespace(
                status=status_code, reason="reason", headers={}
            )

    return FakeResponseException(status)


def test_is_retryable_error_handles_expected_types():
    # HTTPError is also a URLError subclass, so any HTTPError will currently be
    # treated as retryable by the helper logic.
    http_error_retry = api.error.HTTPError(
        "https://example.com", 500, "kaboom", {}, None
    )
    response_retry = _make_response_exception(429)
    assert api.is_retryable_error(http_error_retry)
    assert api.is_retryable_error(response_retry)
    assert api.is_retryable_error(api.ssl.SSLError())
    assert api.is_retryable_error(api.RemoteDisconnected("disconnect"))

    # Non-retryable asyncprawcore response
    non_retryable = _make_response_exception(400)
    assert api.is_retryable_error(non_retryable) is False


@pytest.mark.asyncio
async def test_search_subreddits_returns_display_names(monkeypatch):
    limiter = DummyLimiter()
    fake_subs = [SimpleNamespace(display_name="a"), SimpleNamespace(display_name="b")]

    class FakeSubreddits:
        def search(self, keyword: str):
            assert keyword == "migraine"
            return _AsyncIter(fake_subs)

    fake_client = SimpleNamespace(subreddits=FakeSubreddits())

    result = await api.search_subreddits.__wrapped__(  # type: ignore[attr-defined]
        "migraine", fake_client, limiter
    )

    assert result == ["a", "b"]
    assert limiter.entered == 1 and limiter.exited == 1


@pytest.mark.asyncio
async def test_search_posts_in_subreddit_formats_snippets(monkeypatch):
    limiter = DummyLimiter()
    fake_posts = [
        SimpleNamespace(title="Title 1", selftext="Body 1"),
        SimpleNamespace(title="Title 2", selftext=None),
    ]

    class FakeSubreddit:
        def search(self, keyword: str, limit: int):
            assert keyword == "pain"
            assert limit == 2
            return _AsyncIter(fake_posts)

    class FakeClient:
        async def subreddit(self, name: str):
            assert name == "migraine"
            return FakeSubreddit()

    snippets = await api.search_posts_in_subreddit.__wrapped__(  # type: ignore[attr-defined]
        "migraine", "pain", FakeClient(), limiter, limit=2, char_limit=10
    )

    assert "**Title**: Title 1" in snippets[0]
    assert "**Post content**: Body 1" in snippets[0]
    assert snippets[1].endswith("**Post content**: ")
    assert limiter.entered == 1 and limiter.exited == 1


@pytest.mark.asyncio
async def test_fetch_sub_about_writes_json_and_handles_nsfw(tmp_path):
    limiter = DummyLimiter()

    class FakeSubreddit:
        def __init__(self, name: str):
            self.display_name = name.title()

        over18 = True
        description = "desc"
        public_description = "public"

    class FakeClient:
        async def subreddit(self, name: str, fetch: bool = False):
            return FakeSubreddit(name)

    result = await api._fetch_sub_about.__wrapped__(  # type: ignore[attr-defined]
        "testsub", FakeClient(), tmp_path.as_posix(), limiter
    )

    about_path = tmp_path / "testsub_about.json"
    assert about_path.exists()
    stored = json.loads(about_path.read_text())
    assert stored["description"] == "This subreddit is NSFW (not safe for work)."
    assert result["description"] == "This subreddit is NSFW (not safe for work)."


@pytest.mark.asyncio
async def test_fetch_sub_about_records_api_errors(tmp_path):
    limiter = DummyLimiter()
    response_exc = _make_response_exception(400)

    class FakeClient:
        async def subreddit(self, name: str, fetch: bool = False):
            raise response_exc

    result = await api._fetch_sub_about.__wrapped__(  # type: ignore[attr-defined]
        "oops", FakeClient(), tmp_path.as_posix(), limiter
    )

    saved = json.loads((tmp_path / "oops_about.json").read_text())
    assert result["subreddit"] == "error"
    assert saved["description"] == "reason"


@pytest.mark.asyncio
async def test_get_sub_about_info_reads_existing_csv(tmp_path):
    csv_path = tmp_path / "subs_about.csv"
    df = pd.DataFrame(
        [{"subreddit": "data", "description": "d", "public_description": "p"}]
    )
    df.to_csv(csv_path)

    result = await api.get_sub_about_info(tmp_path.as_posix())

    assert list(result["subreddit"]) == ["data"]


@pytest.mark.asyncio
async def test_get_sub_about_info_requires_env_vars(tmp_path, monkeypatch):
    subs_list = tmp_path / "subs_list.txt"
    subs_list.write_text("xsub\n")

    monkeypatch.setattr(
        "naturalv2.sources.reddit.pushshift_archive.download_subs_list",
        lambda data_path: subs_list.as_posix(),
    )
    # Ensure env vars are absent
    for var in [
        "PRAW_CLIENT_ID",
        "PRAW_CLIENT_SECRET",
        "PRAW_PWD",
        "PRAW_USERNAME",
        "PRAW_AGENT",
    ]:
        monkeypatch.delenv(var, raising=False)

    with pytest.raises(ValueError):
        await api.get_sub_about_info(tmp_path.as_posix())


@pytest.mark.asyncio
async def test_get_sub_about_info_fetches_missing_subs(tmp_path, monkeypatch):
    subs_list = tmp_path / "subs_list.txt"
    subs_list.write_text("a\nb\n")
    about_dir = tmp_path / "subs_about"
    about_dir.mkdir()
    # Pre-existing about JSON for one subreddit
    (about_dir / "a_about.json").write_text(
        json.dumps({"subreddit": "a", "description": "ad", "public_description": "ap"})
    )

    async def fake_fetch(sub_name, client, save_dir, rate_limiter):
        return {
            "subreddit": sub_name,
            "description": f"{sub_name} desc",
            "public_description": f"{sub_name} pub",
        }

    class FakeReddit:
        def __init__(self, *args, **kwargs):
            self.args = args

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

    monkeypatch.setattr(
        "naturalv2.sources.reddit.pushshift_archive.download_subs_list",
        lambda data_path: subs_list.as_posix(),
    )
    monkeypatch.setattr(api, "_fetch_sub_about", fake_fetch)
    monkeypatch.setattr(api.asyncpraw, "Reddit", FakeReddit)

    required_env = {
        "PRAW_CLIENT_ID": "x",
        "PRAW_CLIENT_SECRET": "x",
        "PRAW_PWD": "x",
        "PRAW_USERNAME": "x",
        "PRAW_AGENT": "x",
    }
    monkeypatch.setenv("PRAW_CLIENT_ID", required_env["PRAW_CLIENT_ID"])
    monkeypatch.setenv("PRAW_CLIENT_SECRET", required_env["PRAW_CLIENT_SECRET"])
    monkeypatch.setenv("PRAW_PWD", required_env["PRAW_PWD"])
    monkeypatch.setenv("PRAW_USERNAME", required_env["PRAW_USERNAME"])
    monkeypatch.setenv("PRAW_AGENT", required_env["PRAW_AGENT"])

    result = await api.get_sub_about_info(tmp_path.as_posix())

    assert set(result["subreddit"]) == {"a", "b"}
    assert (tmp_path / "subs_about.csv").exists()
