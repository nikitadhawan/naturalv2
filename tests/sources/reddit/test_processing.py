import polars as pl
import pyarrow as pa
import pytest

from naturalv2.sources.reddit.processing import _utils as utils
from naturalv2.sources.reddit.processing import contextualize as ctx
from naturalv2.sources.reddit.processing import filter as pfilter
from naturalv2.sources.reddit.processing._utils import bucket_from_subreddit


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
