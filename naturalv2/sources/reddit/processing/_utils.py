"""Helpers for processing Reddit data."""

import hashlib
from collections.abc import Mapping, Sequence

import pyarrow as pa
import pyarrow.compute as pc


BUCKET_COUNT = 128
BUCKET_PAD_WIDTH = len(str(BUCKET_COUNT - 1))


def comment_post_id_array(link_ids: pa.Array) -> pa.Array:
    """Normalize comment link_ids to bare post ids (lowercased, trim ``t3_``)."""
    arr = ensure_string_array(link_ids)
    arr = pc.utf8_lower(arr)
    arr = pc.replace_substring_regex(arr, "^t3_", "")
    return pc.utf8_trim_whitespace(arr)


def submission_post_id_array(ids: pa.Array) -> pa.Array:
    """Return lowercased submission ids as a string array."""
    arr = ensure_string_array(ids)
    return pc.utf8_lower(arr)


def unique_strings(arr: pa.Array) -> list[str]:
    """Collect sorted unique, non-empty string values from an Arrow array."""
    if isinstance(arr, pa.ChunkedArray):
        arr = arr.combine_chunks()
    return sorted({value for value in arr.to_pylist() if value})


def non_empty_mask(arr: pa.Array) -> pa.Array:
    """Boolean mask where entries have non-zero length and are not null."""
    mask = pc.greater(pc.utf8_length(arr), 0)
    return pc.fill_null(mask, False)


def mask_has_true(mask: pa.Array | pa.ChunkedArray) -> bool:
    """Return True when a boolean mask contains at least one truthy value."""
    array = mask.combine_chunks() if isinstance(mask, pa.ChunkedArray) else mask
    if len(array) == 0:
        return False
    total = pc.sum(pc.cast(array, pa.int64()))
    return bool(total.as_py())


def ensure_string_array(arr: pa.Array | pa.ChunkedArray, default: str = "") -> pa.Array:
    """Cast to string array, replacing nulls with ``default``."""
    array = arr.combine_chunks() if isinstance(arr, pa.ChunkedArray) else arr
    if not pa.types.is_string(array.type):
        array = pc.cast(array, pa.string(), safe=False)
    return pc.fill_null(array, pa.scalar(default, type=pa.string()))


def ensure_int64_array(arr: pa.Array | pa.ChunkedArray) -> pa.Array:
    """Cast to int64 array, replacing nulls with zeros."""
    array = arr.combine_chunks() if isinstance(arr, pa.ChunkedArray) else arr
    array = pc.cast(array, pa.int64(), safe=False)
    return pc.fill_null(array, pa.scalar(0, type=pa.int64()))


def ensure_timestamp_array(arr: pa.Array | pa.ChunkedArray) -> pa.Array:
    """Cast to UTC seconds timestamp array, replacing nulls with epoch."""
    array = arr.combine_chunks() if isinstance(arr, pa.ChunkedArray) else arr
    ts = pc.cast(array, pa.timestamp("s"), safe=False)
    return pc.fill_null(ts, pa.scalar(0, type=pa.timestamp("s")))


def format_timestamp_array(arr: pa.Array | pa.ChunkedArray) -> pa.Array:
    """Format timestamps (seconds) as ISO-8601 UTC strings with empty nulls."""
    array = arr.combine_chunks() if isinstance(arr, pa.ChunkedArray) else arr
    timestamp = pc.cast(array, pa.timestamp("s"), safe=False)
    formatted = pc.strftime(timestamp, format="%Y-%m-%dT%H:%M:%SZ")
    return pc.fill_null(formatted, pa.scalar("", type=pa.string()))


def filter_array(values: pa.Array | pa.ChunkedArray, mask: pa.Array) -> pa.Array:
    """Filter an Arrow array by mask and combine chunks if needed."""
    filtered = pc.filter(values, mask)
    return (
        filtered.combine_chunks() if isinstance(filtered, pa.ChunkedArray) else filtered
    )


def build_submission_permalink_array(
    *,
    existing: pa.Array,
    subreddits: pa.Array,
    post_ids: pa.Array,
) -> pa.Array:
    """Return per-row submission permalinks, filling missing values when needed."""
    permalink = existing
    length_mask = pc.greater(pc.utf8_length(permalink), 0)
    if mask_has_true(pc.invert(length_mask)):
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


def build_comment_permalink_array(
    *,
    existing: pa.Array,
    post_ids: pa.Array,
    comment_ids: pa.Array,
) -> pa.Array:
    """Return per-row comment permalinks, filling from ids when missing."""
    permalink = existing
    length_mask = pc.greater(pc.utf8_length(permalink), 0)
    if mask_has_true(pc.invert(length_mask)):
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


def _stable_hash_int(arr: pa.Array) -> pa.Array:
    """Return a deterministic int64 hash using BLAKE2b."""
    arr = ensure_string_array(arr)
    hashes: list[int] = []
    for value in arr.to_pylist():
        if value is None:
            hashes.append(0)
            continue
        digest = hashlib.blake2b(value.encode("utf-8"), digest_size=8).digest()
        hashes.append(int.from_bytes(digest, byteorder="big", signed=True))
    return pa.array(hashes, type=pa.int64())


def _mod_int(arr: pa.Array, modulus: int) -> pa.Array:
    """Modulo for int arrays."""
    mask = pa.scalar(modulus - 1, pa.int64())
    return pc.bit_wise_and(arr, mask)


def bucket_from_subreddit(subreddit: pa.Array) -> pa.Array:
    """Stable hash bucket of subreddit name for partitioning."""
    lowered = pc.utf8_lower(subreddit)
    hashed = _stable_hash_int(lowered)
    # Use a fixed modulus to bound bucket count and pad for sortable labels.
    bucket_idx = _mod_int(pc.abs(hashed), BUCKET_COUNT)

    # Turn the bucket_idx into a string and left-pad it with zeros to a fixed width
    # e.g. bucket_idx = 7 + bucket_pad_width=3 -> 007
    bucket_str = pc.utf8_lpad(
        pc.cast(bucket_idx, pa.string()), BUCKET_PAD_WIDTH, padding="0"
    )
    return pc.fill_null(
        bucket_str, pa.scalar("0".zfill(BUCKET_PAD_WIDTH), type=pa.string())
    )


def post_bucket_array(post_ids: pa.Array) -> pa.Array:
    """Stable hash bucket of post_id for grouping replies."""
    lowered = pc.utf8_lower(post_ids)
    hashed = _stable_hash_int(lowered)
    bucket_idx = _mod_int(pc.abs(hashed), BUCKET_COUNT)
    bucket_str = pc.utf8_lpad(
        pc.cast(bucket_idx, pa.string()), BUCKET_PAD_WIDTH, padding="0"
    )
    return pc.fill_null(
        bucket_str, pa.scalar("0".zfill(BUCKET_PAD_WIDTH), type=pa.string())
    )


def build_report_text_array(
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


def author_replies_column(
    post_ids: pa.Array, reply_lookup: Mapping[str, list[str]]
) -> pa.Array:
    """Build list-of-string arrays aligning stored author replies to post ids."""
    data: list[list[str]] = []
    for post_id in post_ids.to_pylist():
        replies = reply_lookup.get(post_id, []) if reply_lookup else []
        data.append([str(reply) for reply in replies])
    return pa.array(data, type=pa.list_(pa.string()))


def empty_list_array(length: int) -> pa.Array:
    """Return a list<string> array of empty lists of given length."""
    return pa.array([[] for _ in range(length)], type=pa.list_(pa.string()))


def constant_string_array(value: str, length: int) -> pa.Array:
    """Return a string array filled with ``value``."""
    return pa.array([value] * length, type=pa.string())
