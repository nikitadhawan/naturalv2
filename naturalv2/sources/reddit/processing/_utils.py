from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc


def _comment_post_id_array(link_ids: pa.Array) -> pa.Array:
    """Normalize comment link_ids to bare post ids (lowercased, trim ``t3_``)."""
    arr = _ensure_string_array(link_ids)
    arr = pc.utf8_lower(arr)
    arr = pc.replace_substring_regex(arr, "^t3_", "")
    return pc.utf8_trim_whitespace(arr)


def _submission_post_id_array(ids: pa.Array) -> pa.Array:
    """Return lowercased submission ids as a string array."""
    arr = _ensure_string_array(ids)
    return pc.utf8_lower(arr)


def _unique_strings(arr: pa.Array) -> list[str]:
    """Collect sorted unique, non-empty string values from an Arrow array."""
    if isinstance(arr, pa.ChunkedArray):
        arr = arr.combine_chunks()
    return sorted({value for value in arr.to_pylist() if value})


def _non_empty_mask(arr: pa.Array) -> pa.Array:
    """Boolean mask where entries have non-zero length and are not null."""
    mask = pc.greater(pc.utf8_length(arr), 0)
    return pc.fill_null(mask, False)


def _mask_has_true(mask: pa.Array | pa.ChunkedArray) -> bool:
    """Return True when a boolean mask contains at least one truthy value."""
    array = mask.combine_chunks() if isinstance(mask, pa.ChunkedArray) else mask
    if len(array) == 0:
        return False
    total = pc.sum(pc.cast(array, pa.int64()))
    return bool(total.as_py())


def _ensure_string_array(
    arr: pa.Array | pa.ChunkedArray, default: str = ""
) -> pa.Array:
    """Cast to string array, replacing nulls with ``default``."""
    array = arr.combine_chunks() if isinstance(arr, pa.ChunkedArray) else arr
    if not pa.types.is_string(array.type):
        array = pc.cast(array, pa.string(), safe=False)
    return pc.fill_null(array, pa.scalar(default, type=pa.string()))


def _ensure_int64_array(arr: pa.Array | pa.ChunkedArray) -> pa.Array:
    """Cast to int64 array, replacing nulls with zeros."""
    array = arr.combine_chunks() if isinstance(arr, pa.ChunkedArray) else arr
    array = pc.cast(array, pa.int64(), safe=False)
    return pc.fill_null(array, pa.scalar(0, type=pa.int64()))


def _format_timestamp_array(arr: pa.Array | pa.ChunkedArray) -> pa.Array:
    """Format timestamps (seconds) as ``Month DD, YYYY`` strings with empty nulls."""
    array = arr.combine_chunks() if isinstance(arr, pa.ChunkedArray) else arr
    timestamp = pc.cast(array, pa.timestamp("s"), safe=False)
    formatted = pc.strftime(timestamp, format="%B %d, %Y")
    return pc.fill_null(formatted, pa.scalar("", type=pa.string()))


def _filter_array(values: pa.Array | pa.ChunkedArray, mask: pa.Array) -> pa.Array:
    """Filter an Arrow array by mask and combine chunks if needed."""
    filtered = pc.filter(values, mask)
    return (
        filtered.combine_chunks() if isinstance(filtered, pa.ChunkedArray) else filtered
    )


def _submission_permalink_array(
    *,
    existing: pa.Array,
    subreddits: pa.Array,
    post_ids: pa.Array,
) -> pa.Array:
    """Return per-row submission permalinks, filling missing values when needed."""
    permalink = existing
    length_mask = pc.greater(pc.utf8_length(permalink), 0)
    if _mask_has_true(pc.invert(length_mask)):
        fallback = []
        for subreddit, post_id in zip(subreddits.to_pylist(), post_ids.to_pylist()):
            if subreddit:
                fallback.append(
                    f"https://www.reddit.com/r/{subreddit}/comments/{post_id}/"
                )
            else:
                fallback.append(f"https://www.reddit.com/comments/{post_id}/")
        merged = permalink.to_pylist()
        bool_mask = pc.invert(length_mask).to_pylist()
        for idx, missing in enumerate(bool_mask):
            if missing:
                merged[idx] = fallback[idx]
        permalink = pa.array(merged, type=pa.string())
    return permalink


def _comment_permalink_array(
    *,
    existing: pa.Array,
    post_ids: pa.Array,
    comment_ids: pa.Array,
) -> pa.Array:
    """Return per-row comment permalinks, filling from ids when missing."""
    permalink = existing
    length_mask = pc.greater(pc.utf8_length(permalink), 0)
    if _mask_has_true(pc.invert(length_mask)):
        fallback = [
            f"https://www.reddit.com/comments/{post_id}/_/{comment_id}"
            for post_id, comment_id in zip(
                post_ids.to_pylist(), comment_ids.to_pylist()
            )
        ]
        merged = permalink.to_pylist()
        bool_mask = pc.invert(length_mask).to_pylist()
        for idx, missing in enumerate(bool_mask):
            if missing:
                merged[idx] = fallback[idx]
        permalink = pa.array(merged, type=pa.string())
    return permalink


def _bucket_from_subreddit(subreddit: pa.Array) -> pa.Array:
    """Bucket key derived from first character of subreddit (``_`` for empty)."""
    lowered = pc.utf8_lower(subreddit)
    first_char = pc.utf8_slice_codeunits(lowered, start=0, stop=1)
    empty_mask = pc.equal(pc.utf8_length(first_char), 0)
    bucket = pc.if_else(empty_mask, pa.scalar("_", type=pa.string()), first_char)
    return pc.fill_null(bucket, pa.scalar("_", type=pa.string()))


def _post_bucket_array(post_ids: pa.Array) -> pa.Array:
    """Bucket key derived from first character of post_id (``_`` for empty)."""
    lowered = pc.utf8_lower(post_ids)
    first_char = pc.utf8_slice_codeunits(lowered, start=0, stop=1)
    empty_mask = pc.equal(pc.utf8_length(first_char), 0)
    bucket = pc.if_else(empty_mask, pa.scalar("_", pa.string()), first_char)
    return pc.fill_null(bucket, pa.scalar("_", pa.string()))


def _build_report_text_array(
    *,
    base_text: pa.Array,
    post_ids: pa.Array,
    reply_lookup: Mapping[str, list[str]],
) -> pa.Array:
    """Append formatted author replies (if any) to base text per post id."""
    result: list[str] = []
    for text, post_id in zip(base_text.to_pylist(), post_ids.to_pylist()):
        replies = reply_lookup.get(post_id) if reply_lookup else None
        if replies:
            suffix = _format_reply_suffix(replies)
            result.append(f"{text or ''}{suffix}")
        else:
            result.append(text or "")
    return pa.array(result, type=pa.string())


def _format_reply_suffix(replies: Sequence[str]) -> str:
    """Render replies as quoted lines appended after a prefix sentence."""
    if not replies:
        return ""
    quoted = "".join(f"\n> {reply}" for reply in replies if reply)
    return (
        "\n\nThe original poster also replied with the following comments in the thread:"
        + quoted
    )


def _author_replies_column(
    post_ids: pa.Array, reply_lookup: Mapping[str, list[str]]
) -> pa.Array:
    """Build list-of-string arrays aligning stored author replies to post ids."""
    data: list[list[str]] = []
    for post_id in post_ids.to_pylist():
        replies = reply_lookup.get(post_id, []) if reply_lookup else []
        data.append([str(reply) for reply in replies])
    return pa.array(data, type=pa.list_(pa.string()))


def _empty_list_array(length: int) -> pa.Array:
    """Return a list<string> array of empty lists of given length."""
    return pa.array([[] for _ in range(length)], type=pa.list_(pa.string()))


def _constant_string_array(value: str, length: int) -> pa.Array:
    """Return a string array filled with ``value``."""
    return pa.array([value] * length, type=pa.string())


def _build_submission_permalink_series(
    subreddit: pd.Series, post_id: pd.Series
) -> pd.Series:
    """Vectorized construction of submission permalinks from subreddit/post ids."""
    sub = subreddit.astype("string").fillna("")
    pid = post_id.astype("string")
    with_sub = "/r/" + sub + "/comments/" + pid + "/"
    without_sub = "/comments/" + pid + "/"
    return pd.Series(
        np.where(sub.ne(""), with_sub, without_sub), index=post_id.index, dtype="string"
    )


def _build_comment_permalink_series(
    post_id: pd.Series, comment_id: pd.Series
) -> pd.Series:
    """Vectorized construction of full comment URLs from post/comment ids."""
    return (
        "https://www.reddit.com/comments/"
        + post_id.astype("string")
        + "/_/"
        + comment_id.astype("string")
    ).astype("string")
