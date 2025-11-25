import ssl
from pathlib import Path
from typing import Iterable

import pyarrow as pa
import pyarrow.compute as pc
import zstandard as zstd

from naturalv2.sources.reddit import pushshift_archive as pa_mod
from naturalv2.sources.reddit.processing import _utils as utils


def test_with_tls_fallback_retries_with_insecure_context():
    calls: list[ssl.SSLContext | None] = []

    def fake_fetch(url: str, context: ssl.SSLContext | None) -> str:
        calls.append(context)
        if len(calls) == 1:
            raise pa_mod.error.URLError(ssl.SSLCertVerificationError())
        return "ok"

    result = pa_mod._with_tls_fallback(
        "https://example.com", fake_fetch, description="testing"
    )

    assert result == "ok"
    assert calls[0] is None and isinstance(calls[1], ssl.SSLContext)


def test_download_subs_list_parses_html(tmp_path, monkeypatch):
    html = """
    <a href="files/foo_submissions.zst">foo</a>
    <a href="files/bar_comments.zst">bar</a>
    """
    fetch_calls = 0

    def fake_with_tls_fallback(url: str, fetch, *, description: str):
        nonlocal fetch_calls
        fetch_calls += 1
        return html

    monkeypatch.setattr(pa_mod, "_with_tls_fallback", fake_with_tls_fallback)

    path = Path(pa_mod.download_subs_list(tmp_path.as_posix()))
    assert path == tmp_path / "subs_list.txt"
    assert sorted(path.read_text().splitlines()) == ["bar", "foo"]

    # Second invocation should reuse the file without fetching again
    pa_mod.download_subs_list(tmp_path.as_posix())
    assert fetch_calls == 1


def test_download_sub_data_creates_file(tmp_path, monkeypatch):
    created: list[str] = []

    def fake_with_tls_fallback(url: str, fetch, *, description: str):
        file_path = tmp_path / "mysub_submissions.zst"
        file_path.write_text("payload")
        created.append(file_path.as_posix())
        return file_path.as_posix()

    monkeypatch.setattr(pa_mod, "_with_tls_fallback", fake_with_tls_fallback)

    result = Path(
        pa_mod.download_sub_data.__wrapped__(  # type: ignore[attr-defined]
            "mysub", "submissions", tmp_path.as_posix()
        )
    )

    assert result == tmp_path / "mysub_submissions.zst"
    assert created


def _compress_ndjson(path, lines: Iterable[str]) -> None:
    compressor = zstd.ZstdCompressor(level=1)
    with open(path, "wb") as f, compressor.stream_writer(f) as writer:
        for line in lines:
            writer.write(line.encode("utf-8"))


def test_iter_zst_ndjson_blocks_emits_complete_lines(tmp_path):
    lines = ['{"id": "1"}\n', '{"id": "2"}\n', '{"id": "3"}\n']
    zst_path = tmp_path / "sample.zst"
    _compress_ndjson(zst_path, lines)

    chunks = list(pa_mod.iter_zst_ndjson_blocks(zst_path.as_posix(), chunk_size=8))

    combined = b"".join(chunks).decode("utf-8").splitlines()
    assert combined == [line.strip() for line in lines]


def test_parse_ndjson_bytes_to_table_dequotes_and_derives_columns(monkeypatch):
    raw_lines = [
        '{"id":"1","created_utc":"123","score":"1","subreddit":"TestA","selftext":"hi"}\n',
        '{"id":"2","created_utc":"456","score":"2.5","subreddit":"TestB","body":"yo"}\n',
    ]
    chunk = "".join(raw_lines).encode("utf-8")

    monkeypatch.setattr(
        pa_mod,
        "apply_rule_based_filter",
        lambda table, field: pc.is_valid(table[field]),
    )

    table = pa_mod._parse_ndjson_bytes_to_table(chunk, "dummy.zst")

    assert table is not None
    assert table.column("content_type").to_pylist() == ["submissions", "comments"]
    expected_buckets = utils.bucket_from_subreddit(
        pa.array(["TestA", "TestB"], type=pa.string())
    ).to_pylist()
    assert table.column("bucket").to_pylist() == expected_buckets
    assert table.column("created_utc").to_pylist() == [123, 456]


def test_iter_bucketed_batches_sorts_by_bucket(monkeypatch, tmp_path):
    dummy_zst = tmp_path / "dummy.zst"
    dummy_zst.write_bytes(b"not-used")

    def fake_iter_blocks(path, chunk_size=0, tqdm_pbar=None):
        yield b"irrelevant"

    sample_table = pa.table(
        {
            "subreddit": ["bsub", "asub"],
            "content_type": ["c", "c"],
            "bucket": ["b", "a"],
        }
    )

    def fake_parse(chunk, zst_path, use_threads=True):
        return sample_table

    monkeypatch.setattr(pa_mod, "iter_zst_ndjson_blocks", fake_iter_blocks)
    monkeypatch.setattr(pa_mod, "_parse_ndjson_bytes_to_table", fake_parse)

    batches = list(
        pa_mod.iter_bucketed_batches(dummy_zst.as_posix(), progress_enabled=False)
    )
    assert batches
    first_batch = batches[0]
    assert first_batch.column("bucket").to_pylist() == ["a", "b"]
