import polars as pl
import pyarrow as pa
import pytest

from naturalv2.sources.reddit.processing import _utils as utils
from naturalv2.sources.reddit.processing import contextualize as ctx
from naturalv2.sources.reddit.processing import filter as pfilter
from naturalv2.sources.reddit.processing._utils import bucket_from_subreddit
from naturalv2.sources.reddit.stages.curate import RedditCurateStage


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
    test_subreddit = "TestSub"
    bucket = bucket_from_subreddit(pa.array([test_subreddit])).to_pylist()[0]
    parquet_dir = tmp_path / "content_type=submissions" / f"bucket={bucket}"
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
        pfilter.scan_reddit_dataset(
            [file_path.as_posix()],
            columns=["subreddit", "title", "report_text", "score", "missing"],
            subreddit=[test_subreddit],
            batch_size=1,
        )
    )

    assert len(batches) == 1
    batch = batches[0]
    assert batch.shape == (1, 4)
    assert batch["subreddit"].to_list() == ["TestSub"]
    assert batch["title"].to_list() == ["keep"]
    assert set(batch.columns) == {"subreddit", "title", "report_text", "score"}


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
            pa.array(["author-key"]),
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


def test_author_key_expression_pseudonymizes_authors():
    keys = pl.DataFrame(
        {"author": ["Commenter", " commenter ", "Other", "[deleted]", None]}
    ).select(ctx._author_key_expr())["author_key"]

    assert keys[0] == keys[1]
    assert keys[0] != keys[2]
    assert keys[0] != "Commenter"
    assert keys[3:].to_list() == [None, None]


def test_author_key_is_available_to_curation():
    assert ctx.CONTEXTUALIZED_RECORD_SCHEMA.field("author_key").type == pa.string()
    assert "author_key" in RedditCurateStage(num_workers=1)._curation_columns


# -- OP-reply attribution goes through author_key -----------------------------


def _write_bucket_inputs(tmp_path):
    """Two threads: one whose OP account was deleted, one whose OP is active."""
    submissions = pl.DataFrame(
        {
            "id": ["p1", "p2"],
            "created_utc": [1_700_000_000, 1_700_000_100],
            "subreddit": ["testsub", "testsub"],
            "title": ["deleted op", "active op"],
            "selftext": ["I tried the drug.", "Started the drug last week."],
            "author": ["[deleted]", "Poster"],
            "score": [1.0, 1.0],
        },
        schema={
            "id": pl.String,
            "created_utc": pl.Int64,
            "subreddit": pl.String,
            "title": pl.String,
            "selftext": pl.String,
            "author": pl.String,
            "score": pl.Float64,
        },
    )
    comments = pl.DataFrame(
        {
            "id": ["c1", "c2", "c3", "c4"],
            "link_id": ["t3_p1", "t3_p1", "t3_p2", "t3_p2"],
            "created_utc": [
                1_700_000_010,
                1_700_000_020,
                1_700_000_110,
                1_700_000_120,
            ],
            "subreddit": ["testsub"] * 4,
            "body": [
                "Stranger whose account is gone.",
                "Another reader.",
                "OP follow-up: it helped.",
                "Reader on the active post.",
            ],
            # c1 is a *different* deleted account; c3 is the OP under a case
            # variant of their username (Reddit usernames are case-insensitive).
            "author": ["[deleted]", "Reader", "poster", "Reader"],
            "score": [1.0] * 4,
        },
        schema={
            "id": pl.String,
            "link_id": pl.String,
            "created_utc": pl.Int64,
            "subreddit": pl.String,
            "body": pl.String,
            "author": pl.String,
            "score": pl.Float64,
        },
    )
    sub_dir = tmp_path / "in" / "content_type=submissions" / "bucket=b"
    com_dir = tmp_path / "in" / "content_type=comments" / "bucket=b"
    sub_dir.mkdir(parents=True)
    com_dir.mkdir(parents=True)
    submissions.write_parquet(sub_dir / "s.parquet")
    comments.write_parquet(com_dir / "c.parquet")
    return {
        "submissions": [str(sub_dir / "s.parquet")],
        "comments": [str(com_dir / "c.parquet")],
    }


def test_process_bucket_attributes_op_replies_by_author_key(tmp_path):
    sub_out, com_out = ctx._process_bucket(
        "b", _write_bucket_inputs(tmp_path), tmp_path / "out", "unit"
    )
    submissions = {row["title"]: row for row in pl.read_parquet(sub_out).to_dicts()}
    comments = pl.read_parquet(com_out)

    # A deleted OP has no key, so a *different* deleted account's comment is not
    # spliced into the post as an "original poster" reply...
    deleted_op = submissions["deleted op"]
    assert deleted_op["author_key"] is None
    assert deleted_op["author_replies"] == []
    assert "original poster also replied" not in deleted_op["report_text"]

    # ...while the active OP's own comment is, even under a case variant.
    active_op = submissions["active op"]
    assert active_op["author_key"] is not None
    assert active_op["author_replies"] == ["OP follow-up: it helped."]
    assert "OP follow-up: it helped." in active_op["report_text"]

    # Comment reports: the OP's own comment is folded into the post and not
    # emitted separately; everyone else's -- including the deleted stranger --
    # survives as its own report.
    assert set(comments["report_text"].to_list()) == {
        "Stranger whose account is gone.",
        "Another reader.",
        "Reader on the active post.",
    }
    assert comments.filter(pl.col("report_text").str.starts_with("Stranger"))[
        "author_key"
    ].to_list() == [None]
