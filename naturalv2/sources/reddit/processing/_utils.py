"""Helpers for processing Reddit data."""

import gc
import hashlib
import multiprocessing as mp

import psutil
import pyarrow as pa
import pyarrow.compute as pc


BUCKET_COUNT = 160
BUCKET_PAD_WIDTH = len(str(BUCKET_COUNT - 1))


def worker_initializer(num_threads: int) -> None:
    """
    Initialize a worker process with thread configuration for PyArrow.

    This function is called once per worker process in a multiprocessing pool.
    It configures PyArrow's CPU and I/O thread counts to prevent thread
    oversubscription, which can cause CPU thrashing when multiple workers
    are running in parallel.

    Parameters
    ----------
    num_threads : int
        Number of threads to allocate to this worker for PyArrow operations.
        Typically set to total_cpu_count / num_workers to distribute resources
        evenly across workers.

    Returns
    -------
    None

    Notes
    -----
    This should be passed as the `initializer` parameter to multiprocessing.Pool
    or similar parallel processing tools.

    Examples
    --------
    >>> import multiprocessing as mp
    >>> pool = mp.Pool(processes=4, initializer=worker_initializer, initargs=(2,))

    """

    # Minimize CPU thrashing by setting the number of threads per worker for pyarrow
    pa.set_cpu_count(num_threads)
    pa.set_io_thread_count(num_threads)


def get_default_num_workers(
    mem_gb_per_worker: int | None = None, threads_per_worker: int | None = None
) -> int:
    """
    Calculate the optimal number of worker processes based on system resources.

    This function determines how many parallel workers can run simultaneously
    without overwhelming the system's CPU or memory resources. It considers:
    - Available CPU cores
    - Available memory (if memory constraint is specified)
    - Thread requirements per worker (if threading constraint is specified)

    The function balances these constraints to prevent resource exhaustion.

    Parameters
    ----------
    mem_gb_per_worker : int, optional, default=None
        Expected memory usage per worker in gigabytes. If provided, the number
        of workers will be limited to ensure total memory usage doesn't exceed
        available system memory. If ``None``, memory is not considered as a constraint.
    threads_per_worker : int, optional, default=None
        Number of threads each worker will use. If provided, ensures total
        thread count (workers × threads_per_worker) doesn't exceed CPU count.
        If ``None``, thread count is not considered as a constraint.

    Returns
    -------
    int
        Recommended number of worker processes. Always returns at least 1.
        Maximum is capped at 32 or limited by resource constraints.

    Examples
    --------
    >>> # Get workers with no constraints (CPU-based only)
    >>> num_workers = get_default_num_workers()

    >>> # Limit by memory: each worker needs ~4GB
    >>> num_workers = get_default_num_workers(mem_gb_per_worker=4)

    >>> # Limit by both memory and threads
    >>> num_workers = get_default_num_workers(mem_gb_per_worker=4, threads_per_worker=2)

    Notes
    -----
    The default CPU-based calculation uses `min(32, cpu_count + 4)` to allow
    some oversubscription for I/O-bound tasks while capping at a reasonable
    maximum. When both memory and thread constraints are provided, the function
    returns the minimum of the two limits.

    """
    cpu_count: int = psutil.cpu_count(logical=True) or 1

    max_workers_by_cpu: int = min(32, cpu_count + 4)  # Default max if no constraint

    if threads_per_worker is not None:
        max_workers_by_cpu = int(cpu_count // min(threads_per_worker, cpu_count))

    if mem_gb_per_worker is not None:
        available_mem_gb: float = psutil.virtual_memory().available / (1024**3)
        max_workers_by_mem = int(available_mem_gb // mem_gb_per_worker)

        return max(1, min(max_workers_by_cpu, max_workers_by_mem))

    return max(1, max_workers_by_cpu)


def release_memory() -> None:
    """Aggressively release memory used by PyArrow and Python.

    This function performs multiple memory cleanup operations:
    1. Releases PyArrow's internal memory pool
    2. Runs Python's garbage collector to free cyclic references
    3. On Linux, calls malloc_trim to return memory to the OS

    This is particularly useful in multiprocessing scenarios where worker
    processes can accumulate memory over time, even after data is no longer
    actively used.

    Returns
    -------
    None

    Notes
    -----
    The malloc_trim operation (step 3) is Linux-specific and will be silently
    skipped on Windows and macOS. This is intentional to maintain cross-platform
    compatibility.

    Examples
    --------
    >>> # Release memory after processing a large batch
    >>> for batch in data_batches:
    ...     process_batch(batch)
    ...     del batch
    ...     release_memory()

    See Also
    --------
    gc.collect : Python's garbage collector

    """
    #  Release PyArrow's Internal Pool
    pa.default_memory_pool().release_unused()

    # Force Python GC to release cyclic references
    gc.collect()

    # Force C-level memory release (Linux only)
    # This is meant for the OS to shrink memory held by idle workers
    try:
        import ctypes  # noqa: PLC0415

        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass  # Ignore on Windows/MacOS


def get_tqdm_position() -> int:
    """
    Get the vertical position for a tqdm progress bar in a multiprocessing context.

    When multiple workers are running in parallel and each displays a progress bar,
    this function ensures each worker's progress bar appears at a different vertical
    position on the terminal, preventing overlapping displays.

    Returns
    -------
    int
        The position index for the tqdm progress bar. Returns 1 if called outside
        a multiprocessing context (e.g., in the main process), or worker_id + 1
        if called from within a worker process.

    Examples
    --------
    >>> from tqdm import tqdm
    >>> position = get_tqdm_position()
    >>> for item in tqdm(items, position=position, leave=False):
    ...     process(item)

    Notes
    -----
    The position is calculated as worker_id + 1 to reserve position 0 for a
    potential main progress bar. If the function cannot determine the worker ID
    (e.g., not in a multiprocessing context), it defaults to position 1.

    """
    try:
        # _identity returns a tuple like (1,) for the 1st worker, (2,) for 2nd...
        worker_id = mp.current_process()._identity[0]
        tqdm_position = worker_id + 1  # +1 to skip main bar
    except (AttributeError, IndexError):
        tqdm_position = 1  # Fallback

    return tqdm_position


def ensure_string_array(arr: pa.Array | pa.ChunkedArray, default: str = "") -> pa.Array:
    """
    Convert a PyArrow array to a string array with null handling.

    This utility function ensures the input array is in string format and
    replaces any null values with a default string. If the input is a
    ChunkedArray, it will be combined into a single contiguous array.

    Parameters
    ----------
    arr : pa.Array or pa.ChunkedArray
        Input PyArrow array of any type. Can be a single Array or ChunkedArray.
    default : str, default=""
        The string value to use for replacing null entries. By default, nulls
        are replaced with empty strings.

    Returns
    -------
    pa.Array
        A PyArrow Array of string type with all nulls replaced by `default`.

    Examples
    --------
    >>> import pyarrow as pa
    >>> arr = pa.array([1, 2, None, 4])
    >>> ensure_string_array(arr)
    <pyarrow.lib.StringArray object at 0x...>
    ['1', '2', '', '4']

    >>> arr = pa.array(["hello", None, "world"])
    >>> ensure_string_array(arr, default="N/A")
    <pyarrow.lib.StringArray object at 0x...>
    ['hello', 'N/A', 'world']

    """
    array = arr.combine_chunks() if isinstance(arr, pa.ChunkedArray) else arr
    if not pa.types.is_string(array.type):
        array = pc.cast(array, pa.string(), safe=False)
    return pc.fill_null(array, pa.scalar(default, type=pa.string()))


def _stable_hash_int(arr: pa.Array) -> pa.Array:
    """
    Compute deterministic 64-bit integer hashes for string values.

    This function uses the BLAKE2b cryptographic hash algorithm to generate
    stable, reproducible hash values from strings. The same input will always
    produce the same hash value across different runs and machines.

    Parameters
    ----------
    arr : pa.Array
        Input PyArrow array. Will be converted to string type if necessary.
        Null values are hashed to 0.

    Returns
    -------
    pa.Array
        An int64 array where each element is the hash of the corresponding
        input element. Null inputs produce 0 as output.

    Notes
    -----
    This function uses BLAKE2b with an 8-byte (64-bit) digest size, which is
    then converted to a signed int64. The hash is stable and deterministic,
    making it suitable for partitioning and bucketing operations that need
    to be reproducible across different runs.

    Examples
    --------
    >>> import pyarrow as pa
    >>> arr = pa.array(["apple", "banana", None, "apple"])
    >>> hashes = _stable_hash_int(arr)
    >>> # Same values produce same hashes
    >>> hashes[0] == hashes[3]
    True

    """
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
    """
    Apply modulo operation to an integer array.

    This function computes the remainder when each element in the array is
    divided by the modulus value. The operation is performed efficiently
    using NumPy's vectorized operations.

    Parameters
    ----------
    arr : pa.Array
        Input PyArrow array of integer values.
    modulus : int
        The divisor for the modulo operation. Must be a positive integer.

    Returns
    -------
    pa.Array
        An int64 array where each element is (input_element % modulus).

    Examples
    --------
    >>> import pyarrow as pa
    >>> arr = pa.array([10, 25, 33, 47, 50])
    >>> _mod_int(arr, 10)
    <pyarrow.lib.Int64Array object at 0x...>
    [0, 5, 3, 7, 0]

    Notes
    -----
    The function converts the PyArrow array to NumPy for the modulo operation,
    then converts back to PyArrow. This approach is efficient for numerical
    operations.

    """
    # Convert to numpy, perform modulo, then wrap back to PyArrow
    # This keeps the data as int64 the entire time.
    numpy_view = arr.to_numpy(zero_copy_only=False)
    remainder = numpy_view % modulus
    return pa.array(remainder, type=pa.int64())


def bucket_from_subreddit(subreddit: pa.Array) -> pa.Array:
    """
    Generate zero-padded bucket strings from subreddit names for partitioning.

    This function creates stable, uniformly distributed partition keys from
    subreddit names. It:
    1. Converts subreddit names to lowercase for case-insensitive bucketing
    2. Computes a stable hash of each name
    3. Takes the modulo to limit bucket count to ``BUCKET_COUNT``
    4. Zero-pads the result to create sortable string labels (e.g., "007", "042")

    This ensures subreddits are distributed across a fixed number of buckets,
    which is useful for organizing data into manageable partitions.

    Parameters
    ----------
    subreddit : pa.Array
        PyArrow array of subreddit names (strings). Case is ignored during
        bucketing. Null values are assigned to bucket "000".

    Returns
    -------
    pa.Array
        String array of zero-padded bucket identifiers. Each bucket is a
        string like "000", "001", ..., "159", padded to uniform width.

    Examples
    --------
    >>> import pyarrow as pa
    >>> subreddits = pa.array(["AskReddit", "science", "askreddit", None])
    >>> buckets = bucket_from_subreddit(subreddits)
    >>> buckets
    <pyarrow.lib.StringArray object at 0x...>
    ['042', '103', '042', '000'] # NOTE: actual bucket numbers may vary
    >>> # Note: "AskReddit" and "askreddit" get the same bucket (case-insensitive)

    Notes
    -----
    The bucket count is controlled by the module-level constant ``BUCKET_COUNT``.
    The padding width is automatically determined to accommodate all bucket numbers.

    See Also
    --------
    _stable_hash_int : The underlying hash function used
    _mod_int : The modulo operation to limit bucket range

    """
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
