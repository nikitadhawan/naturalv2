from datetime import datetime

import polars as pl
import pyarrow as pa
import pytest

from naturalv2.sources.reddit.processing import _utils as utils
from naturalv2.sources.reddit.processing import contextualize as ctx
from naturalv2.sources.reddit.processing import filter as pfilter


def test_apply_rule_based_filter_flags_common_cases():
    table = pa.table(
        {
            "body": pa.array(
                [
                    "Fish &amp; chips with sauce",  # valid text, HTML unescapes
                    "[deleted]",  # sentinel
                    "Real content",  # bot author blocked
                    "www.example.com",  # URL-only
                    None,  # null text
                ],
                type=pa.string(),
            ),
            "author": pa.array(
                ["user", "user", "bot_123", "user", "AutoModerator"], type=pa.string()
            ),
        }
    )

    mask = pfilter.apply_rule_based_filter(table, "body").combine_chunks()

    values = mask.to_pylist()

    assert values[0] is True
    assert values[1] is False  # sentinel
    assert values[2] is True  # bot-like author
    assert not bool(values[3])  # URL-only text should be filtered
    assert values[4] is False  # AutoModerator blocked


def test_scan_reddit_chunks_filters_and_limits_columns(tmp_path):
    parquet_dir = tmp_path / "content_type=submissions" / "bucket=001"
    parquet_dir.mkdir(parents=True)
    file_path = parquet_dir / "sample.parquet"
    pl.DataFrame(
        {
            "subreddit": ["TestSub", "Other"],
            "title": ["keep", "drop"],
            "report_text": ["body", "ignored"],
            "score": [1, 2],
        }
    ).write_parquet(file_path)

    batches = list(
        pfilter.scan_reddit_chunks(
            [file_path.as_posix()],
            columns=["subreddit", "title", "report_text", "score", "missing"],
            target_subreddits=["TestSub"],
            batch_size=1,
        )
    )

    assert len(batches) == 1
    batch = batches[0]
    assert batch.shape == (1, 4)
    assert batch["subreddit"].to_list() == ["TestSub"]
    assert batch["title"].to_list() == ["keep"]
    assert set(batch.columns) == {"subreddit", "title", "report_text", "score"}


def test_scan_reddit_chunks_skips_files_without_text_columns(tmp_path):
    file_path = tmp_path / "bucket=000" / "part.parquet"
    file_path.parent.mkdir(parents=True)
    pl.DataFrame(
        {
            "subreddit": ["TestSub"],
            "value": [10],
        }
    ).write_parquet(file_path)

    # Request only non-text columns -> function should skip this file entirely.
    batches = list(
        pfilter.scan_reddit_chunks(
            [file_path.as_posix()],
            columns=["subreddit", "value"],
            target_subreddits=None,
            batch_size=16,
        )
    )

    assert batches == []


def test_write_to_parquet_partitions_creates_hive_layout(tmp_path):
    schema = ctx.CONTEXTUALIZED_RECORD_SCHEMA
    bucket = utils.bucket_from_subreddit(pa.array(["testsub"])).to_pylist()[0]
    batch = pa.RecordBatch.from_arrays(
        [
            pa.array(["testsub"]),
            pa.array(["title"]),
            pa.array(["body"]),
            pa.array(["report"]),
            pa.array(["submission"]),
            pa.array([1], type=pa.int64()),
            pa.array(["2024-01-01T00:00:00Z"]),
            pa.array([""]),
            pa.array([["reply"]], type=pa.list_(pa.string())),
            pa.array(["submissions"]),
            pa.array([bucket]),
        ],
        names=[field.name for field in schema],
    )

    written = ctx.write_to_parquet_partitions(
        data_stream=[batch],
        output_dir=tmp_path.as_posix(),
        schema=schema,
        run_tag="unit",
        max_partitions=8,
        min_rows_per_group=1,
        max_rows_per_group=2,
        max_open_files=8,
    )

    assert len(written) == 1
    assert "content_type=submissions" in written[0]
    assert f"bucket={bucket}" in written[0]
    assert tmp_path.joinpath("content_type=submissions", f"bucket={bucket}").exists()
    assert tmp_path.joinpath(
        "content_type=submissions", f"bucket={bucket}", "unit-part-0.parquet"
    ).exists()


def test_write_to_parquet_partitions_validates_args(tmp_path):
    schema = ctx.CONTEXTUALIZED_RECORD_SCHEMA
    with pytest.raises(ValueError):
        ctx.write_to_parquet_partitions(
            data_stream=[],
            output_dir=tmp_path.as_posix(),
            schema=schema,
            parquet_compression_level=0,
        )


def test_prepare_comment_batch_filters_and_aligns_records():
    batch = pa.record_batch(
        {
            "link_id": pa.array(["t3_abc", "t3_def", None]),
            "author": pa.array(["c1", "c2", "c3"]),
            "body": pa.array(["b1", "b2", "b3"]),
            "created_utc": pa.array([1, 2, 3], type=pa.int64()),
        }
    )
    lookup_table = pa.table(
        {"post_id": pa.array(["abc"]), "author": pa.array(["poster"])}
    )

    prepared = ctx._prepare_comment_batch(
        batch=batch, lookup_fn=lambda post_ids: lookup_table
    )

    assert prepared is not None
    assert prepared.aligned.num_rows == 1
    assert prepared.matched_post_ids.to_pylist() == ["abc"]
    # Index 0 in lookup_table corresponds to "abc"
    assert prepared.indices.to_pylist() == [0]

    # No matching posts -> None
    assert (
        ctx._prepare_comment_batch(
            batch=batch,
            lookup_fn=lambda post_ids: pa.table(
                {
                    "post_id": pa.array([], type=pa.string()),
                    "author": pa.array([], type=pa.string()),
                }
            ),
        )
        is None
    )


def test_utils_helpers_format_and_mask():
    link_ids = pa.array(["t3_XyZ", "  ", None])
    assert utils.comment_post_id_array(link_ids).to_pylist() == ["xyz", "", ""]

    mask = utils.non_empty_mask(pa.array(["a", ""]))
    assert mask.to_pylist() == [True, False]
    assert utils.mask_has_true(mask) is True
    assert utils.mask_has_true(pa.array([], type=pa.bool_())) is False

    report = utils.build_report_text_array(
        base_text=pa.array(["Base", "NoReplies"]),
        post_ids=pa.array(["p1", "p2"]),
        reply_lookup={"p1": ["one", "two"]},
    )
    assert report[0].as_py().startswith("Base\n\nThe original poster also replied")
    assert report[1].as_py() == "NoReplies"


def test_utils_builds_submission_permalinks_when_missing():
    result = utils.build_submission_permalink_array(
        existing=pa.array(["", "https://existing"], type=pa.string()),
        subreddits=pa.array(["Science", ""], type=pa.string()),
        post_ids=pa.array(["abc", "def"], type=pa.string()),
    )

    assert result.to_pylist() == [
        "https://www.reddit.com/r/Science/comments/abc/",
        "https://existing",
    ]


def test_utils_builds_comment_permalinks_when_missing():
    result = utils.build_comment_permalink_array(
        existing=pa.array(["", "https://already"], type=pa.string()),
        post_ids=pa.array(["abc", "xyz"], type=pa.string()),
        comment_ids=pa.array(["c1", "c2"], type=pa.string()),
    )

    assert result.to_pylist() == [
        "https://www.reddit.com/comments/abc/_/c1",
        "https://already",
    ]


def test_utils_unique_and_cast_helpers_cover_chunked_data():
    arr = pa.chunked_array(
        [
            pa.array(["A", "b", None], type=pa.string()),
            pa.array(["b", ""], type=pa.string()),
        ]
    )
    assert utils.unique_strings(arr) == ["A", "b"]

    str_arr = utils.ensure_string_array(pa.array([1, None], type=pa.int32()), default="x")
    assert str_arr.to_pylist() == ["1", "x"]

    int_arr = utils.ensure_int64_array(pa.array([1.9, None], type=pa.float64()))
    assert int_arr.to_pylist() == [1, 0]


def test_utils_timestamp_and_filter_helpers():
    ts = utils.ensure_timestamp_array(pa.array([None, 1], type=pa.int64()))
    assert ts.to_pylist() == [
        datetime.utcfromtimestamp(0),
        datetime.utcfromtimestamp(1),
    ]

    formatted = utils.format_timestamp_array(pa.array([None, 1], type=pa.int64()))
    assert formatted.to_pylist() == ["", "1970-01-01T00:00:01Z"]

    values = pa.chunked_array([pa.array([1, 2]), pa.array([3])])
    mask = pa.array([True, False, True])
    filtered = utils.filter_array(values, mask)
    assert filtered.to_pylist() == [1, 3]
    assert isinstance(filtered, pa.Array)


def test_utils_author_reply_and_list_helpers():
    replies = utils.author_replies_column(
        pa.array(["p1", "p2"], type=pa.string()),
        reply_lookup={"p1": ["a"], "p2": []},
    )
    assert replies.to_pylist() == [["a"], []]

    empty = utils.empty_list_array(2)
    assert empty.type == pa.list_(pa.string())
    assert empty.to_pylist() == [[], []]

    constant = utils.constant_string_array("fill", 3)
    assert constant.to_pylist() == ["fill", "fill", "fill"]
