import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import pytest
from ahocorasick import Automaton

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


def test_get_study_relevant_posts_matches_and_respects_cutoff():
    automaton = Automaton()
    automaton.add_word("drug", "drug")
    automaton.make_automaton()

    df = pd.DataFrame(
        [
            {
                "report_text": "This drug works!",
                "date_created": pd.Timestamp("2022-01-01"),
            },
            {
                "report_text": "No keywords here",
                "date_created": pd.Timestamp("2022-01-01"),
            },
        ]
    )

    matched = pfilter.get_study_relevant_posts(
        df, automaton, cutoff_dt=None, date_column="date_created"
    )
    assert matched["treatments_mentioned"].explode().tolist() == ["drug"]

    cutoff = pd.Timestamp("2021-12-31")
    assert pfilter.get_study_relevant_posts(df, automaton, cutoff).empty


def test_scan_subreddit_filters_by_partition(tmp_path):
    table = pa.table(
        {
            "subreddit": ["TestSub", "Other"],
            "content_type": ["submissions", "submissions"],
            "bucket": ["t", "o"],
            "value": [1, 2],
        }
    )
    partitioning = ds.partitioning(
        pa.schema([("content_type", pa.string()), ("bucket", pa.string())]),
        flavor="hive",
    )
    ds.write_dataset(table, tmp_path, format="parquet", partitioning=partitioning)

    try:
        batches = list(
            pfilter.scan_subreddit(
                tmp_path.as_posix(),
                subreddits="TestSub",
                content_type="submissions",
                columns=["value"],
                batch_size=8,
            )
        )
    except pa.ArrowNotImplementedError as exc:  # pragma: no cover
        pytest.skip(f"pyarrow missing kernel support: {exc}")
    assert len(batches) == 1
    assert list(batches[0]["value"]) == [1]


def test_write_to_parquet_partitions_creates_hive_layout(tmp_path):
    schema = ctx.CONTEXTUALIZED_RECORD_SCHEMA
    batch = pa.RecordBatch.from_arrays(
        [
            pa.array(["testsub"]),
            pa.array(["title"]),
            pa.array(["body"]),
            pa.array(["report"]),
            pa.array(["submission"]),
            pa.array([1], type=pa.int64()),
            pa.array(["2024-01-01"]),
            pa.array([""]),
            pa.array([["reply"]], type=pa.list_(pa.string())),
            pa.array(["submissions"]),
            pa.array(["t"]),
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
    assert "bucket=t" in written[0]
    assert tmp_path.joinpath("content_type=submissions", "bucket=t").exists()
    assert tmp_path.joinpath(
        "content_type=submissions", "bucket=t", "unit-part-0.parquet"
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
    assert utils._comment_post_id_array(link_ids).to_pylist() == ["xyz", "", ""]

    mask = utils._non_empty_mask(pa.array(["a", ""]))
    assert mask.to_pylist() == [True, False]
    assert utils._mask_has_true(mask) is True
    assert utils._mask_has_true(pa.array([], type=pa.bool_())) is False

    report = utils._build_report_text_array(
        base_text=pa.array(["Base", "NoReplies"]),
        post_ids=pa.array(["p1", "p2"]),
        reply_lookup={"p1": ["one", "two"]},
    )
    assert report[0].as_py().startswith("Base\n\nThe original poster also replied")
    assert report[1].as_py() == "NoReplies"

    sub_series = pd.Series(["sub"], dtype="string")
    post_series = pd.Series(["123"], dtype="string")
    permalink_series = utils._build_submission_permalink_series(sub_series, post_series)
    assert permalink_series.iloc[0] == "/r/sub/comments/123/"

    comment_permalink = utils._build_comment_permalink_series(
        pd.Series(["123"]), pd.Series(["c1"])
    )
    assert comment_permalink.iloc[0] == "https://www.reddit.com/comments/123/_/c1"
