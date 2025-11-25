"""Helpers for processing Reddit data."""

import hashlib

import pyarrow as pa
import pyarrow.compute as pc


BUCKET_COUNT = 1024
BUCKET_PAD_WIDTH = len(str(BUCKET_COUNT - 1))


def ensure_string_array(arr: pa.Array | pa.ChunkedArray, default: str = "") -> pa.Array:
    """Cast to string array, replacing nulls with ``default``."""
    array = arr.combine_chunks() if isinstance(arr, pa.ChunkedArray) else arr
    if not pa.types.is_string(array.type):
        array = pc.cast(array, pa.string(), safe=False)
    return pc.fill_null(array, pa.scalar(default, type=pa.string()))


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
