from pathlib import Path

import polars as pl

from naturalv2.sources.reddit.stages.curate import (
    RedditCurateStage,
    _aggregate_reports_by_author,
    _build_report_expr,
)


def _record(author, date, permalink, report):
    return {
        "author_key": author,
        "date_created": date,
        "permalink": permalink,
        "subreddit": "health",
        "report": report,
    }


def test_report_distinguishes_author_text_from_context():
    records = pl.DataFrame(
        {
            "subreddit": ["health", "health"],
            "title": ["My treatment", "Another person's question"],
            "initial_post": ["", "Context author's experience"],
            "report_text": ["My submission body", "My comment"],
            "report_type": ["submission", "comment"],
            "date_created": ["January 01, 2024", "January 02, 2024"],
            "permalink": ["/post", "/post/_/comment"],
        }
    )
    submission, comment = records.with_columns(
        _build_report_expr(records.columns).alias("report")
    )["report"]

    assert "**Author's own text**\nTitle: My treatment" in submission
    assert "Context written by another Reddit user" not in submission
    assert "/post/_/comment" in comment
    assert all(
        text in comment
        for text in [
            "**Context written by another Reddit user**",
            "Another person's question",
            "Context author's experience",
            "**Author's own text**\nComment:\nMy comment",
        ]
    )


def test_aggregate_reports_by_author_orders_deduplicates_and_keeps_unkeyed():
    records = pl.DataFrame(
        [
            _record("author-a", "January 02, 2024", "/later", "later"),
            _record("author-a", "January 01, 2024", "/earlier", "earlier"),
            _record("author-a", "January 01, 2024", "/earlier", "duplicate"),
            _record(None, "January 03, 2024", "/one", "unkeyed one"),
            _record(None, "January 04, 2024", "/two", "unkeyed two"),
        ]
    )

    aggregated = _aggregate_reports_by_author(records)
    author = aggregated.filter(author_key="author-a").row(0, named=True)

    assert len(aggregated) == 3
    assert author["source_record_count"] == 2
    assert "Combined Reddit records from one pseudonymous author" in author["report"]
    assert author["report"].index("earlier") < author["report"].index("later")
    assert "duplicate" not in author["report"]
    assert aggregated.filter(pl.col("author_key").is_null())[
        "source_record_count"
    ].to_list() == [1, 1]


def test_consolidation_combines_records_across_worker_files(tmp_path: Path):
    nct_id = "NCT00000001"
    temp_root = tmp_path / "temp"
    for worker, record in [
        ("a", _record("author-a", "January 02, 2024", "/later", "later")),
        ("b", _record("author-a", "January 01, 2024", "/earlier", "earlier")),
    ]:
        worker_path = temp_root / worker
        worker_path.mkdir(parents=True)
        pl.DataFrame([record]).write_parquet(worker_path / f"{nct_id}_{worker}.parquet")

    target = tmp_path / "curated"
    target.mkdir()
    (target / "stale.parquet").write_bytes(b"stale")
    count = RedditCurateStage._consolidate_parquet_chunks(
        str(temp_root), nct_id, str(target)
    )
    consolidated = pl.read_parquet(target)

    assert count == len(consolidated) == 1
    assert consolidated["report"].item().index("earlier") < consolidated[
        "report"
    ].item().index("later")
    assert not (target / "stale.parquet").exists()
