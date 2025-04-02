import asyncio
import heapq
import logging
import re
import time
from dataclasses import dataclass
from typing import Any, Optional, Union

from naturalv2.models.rate_limiter._bucket import Bucket


@dataclass
class RateLimiterAcquisitionHandle:
    estimated_tokens: int


class RateLimiter:
    """Rate limiter with support for multiple token buckets and request buckets.

    This rate limiter is designed to be used in an asynchronous context, such as
    with asyncio. It supports multiple token buckets and request buckets, each
    with their own refill rates and burst capacities. The rate limiter can be used
    to acquire tokens and request tokens, and will block until the requested
    capacity is available.

    Parameters
    ----------
    requests_per_minute : int
        The number of requests allowed per minute.
    tokens_per_minute : int, optional, default=None
        The number of tokens allowed per minute. If ``None``, the token per minute
        limit is not considered or enforced.
    requests_per_day : int, optional, default=None
        The number of requests allowed per day. If ``None``, the request per day
        limit is not considered or enforced.
    tokens_per_day : int, optional, default=None
        The number of tokens allowed per day. If ``None``, the token per day
        limit is not considered or enforced.
    max_request_burst : int, optional, default=None
        The maximum burst capacity for the request buckets. This is only applied
        to the ``requests_per_minute`` bucket. If ``None``, the burst capacity is
        set to the same value as ``requests_per_minute``.
    max_token_burst : int, optional, default=None
        The maximum burst capacity for the token buckets. This is only applied to
        the ``tokens_per_minute`` bucket. If ``None``, the burst capacity is set
        to the same value as ``tokens_per_minute``.

    Raises
    ------
    ValueError
        If ``requests_per_minute``, tokens_per_minute``, ``tokens_per_day`` or
        ``requests_per_day`` are non-positive.

    Examples
    --------
    >>> rate_limiter = RateLimiter(requests_per_minute=60, tokens_per_minute=120)
    >>> estimated_tokens = ...  # estimate the number of tokens needed (input + output)
    >>> handle = await rate_limiter.acquire(
    ...     estimated_tokens
    ... )  # block until tokens are available
    >>> response = ...  # make the request
    >>> actual_tokens = ...  # get the actual number of tokens consumed
    >>> await rate_limiter.adjust(
    ...     handle, actual_tokens, response_headers=response._response_headers
    ... )  # adjust the rate limiter state based on the actual tokens consumed

    """

    def __init__(
        self,
        requests_per_minute: int,
        tokens_per_minute: Optional[int] = None,
        requests_per_day: Optional[int] = None,
        tokens_per_day: Optional[int] = None,
        max_request_burst: Optional[int] = None,
        max_token_burst: Optional[int] = None,
    ) -> None:
        self.requests_per_minute = requests_per_minute
        self.tokens_per_minute = tokens_per_minute
        self.requests_per_day = requests_per_day
        self.tokens_per_day = tokens_per_day
        self.max_request_burst = max_request_burst
        self.max_token_burst = max_token_burst

        # buckets for tracking requests
        self._request_buckets: dict[str, Bucket] = {
            "rpm": Bucket(
                requests_per_minute, capacity=max_request_burst, time_period=60
            )
        }
        if requests_per_day is not None:
            self._request_buckets["rpd"] = Bucket(
                requests_per_day, time_period=24 * 60 * 60
            )

        # buckets for tracking (LLM) tokens
        self._token_buckets: dict[str, Bucket] = {}
        if tokens_per_minute is not None:
            self._token_buckets["tpm"] = Bucket(
                tokens_per_minute, capacity=max_token_burst, time_period=60
            )
        if tokens_per_day is not None:
            self._token_buckets["tpd"] = Bucket(
                tokens_per_day, time_period=24 * 60 * 60
            )

        # global lock for thread safety
        self._lock = asyncio.Lock()

        # min-heap of waiters (tokens_needed, time_added, future)
        self._waiters: list[tuple[int, float, asyncio.Future]] = []

        # timer handle for next refill
        self._timer_handle: Optional[asyncio.TimerHandle] = None

    async def acquire(self, estimated_tokens: int) -> Optional[asyncio.Future]:
        """Acquire tokens from the rate limiter.

        This is the main method for acquiring tokens from the rate limiter. It will
        block until the requested number of tokens are available across all buckets.
        The method will return a future that will be resolved when the tokens are
        acquired.

        It acquires the global lock to ensure thread safety and atomicity of the
        operation. It will prioritize waiting tasks and notify waiters when tokens
        are available.

        Parameters
        ----------
        estimated_tokens : int
            The number of tokens to acquire.

        Returns
        -------
        asyncio.Future, optional
            A future that will be resolved when the tokens are acquired, or ``None``
            if the acquisition failed.

        Raises
        ------
        ValueError
            If the requested number of tokens exceeds the capacity of any token bucket.

        """
        if any(
            estimated_tokens > bucket.capacity
            for bucket in self._token_buckets.values()
        ):
            raise ValueError(
                f"Requested number of tokens {estimated_tokens} exceeds the capacity "
                "of at least one token bucket. Please reduce the number of tokens "
                "or increase ``max_token_burst``."
            )

        wait_future: Optional[asyncio.Future] = None  # future to return to caller

        while True:
            async with self._lock:  # acquire global lock for thread safety/atmoicity
                current_time = time.monotonic()

                self._notify_waiters()  # prioritize waiting tasks

                bucket_token_pairs = self._get_bucket_token_pairs(estimated_tokens)

                # check if all buckets have enough tokens for the request
                # if so, consume the tokens and return the handle
                if self._has_capacity(bucket_token_pairs, current_time):
                    self._consume_all(bucket_token_pairs, current_time)
                    logging.debug(
                        f"Acquired {estimated_tokens} tokens without waiting."
                    )

                    self._notify_waiters()  # wake waiters in case there is more capacity
                    return RateLimiterAcquisitionHandle(estimated_tokens)

                # not enough tokens, add to priority queue
                if wait_future is None:
                    try:
                        loop = asyncio.get_running_loop()
                        wait_future = loop.create_future()
                        heapq.heappush(
                            self._waiters,
                            (estimated_tokens, current_time, wait_future),
                        )
                        logging.debug(
                            f"Added {estimated_tokens} tokens to the wait queue "
                            f"at {current_time:.2f}s."
                        )

                        self._notify_waiters()  # in case there is more capacity
                    except RuntimeError as e:
                        logging.error(
                            f"Failed to get loop to create waiter future: {e}"
                        )
                        return None
                else:  # already waiting, process after lock release
                    pass

            if wait_future:
                try:
                    await wait_future

                    return RateLimiterAcquisitionHandle(estimated_tokens)
                except asyncio.CancelledError:
                    logging.warning(
                        f"Waiting for ~{estimated_tokens} tokens was cancelled."
                    )
                    async with self._lock:
                        self._notify_waiters()  # in case there is more capacity
                    return None
                except Exception:
                    return None

    async def adjust(
        self,
        handle: RateLimiterAcquisitionHandle,
        actual_tokens: int,
        response_headers: Optional[dict[str, Any]] = None,
    ) -> None:
        """Adjust the rate limiter state based on the actual tokens consumed.

        This method should be called after a request has been made and the actual
        number of tokens consumed is known. It will adjust the token buckets and
        request buckets based on the difference between the estimated tokens and
        the actual tokens consumed. It will also attempt to get the rate limit
        information from the response headers to update the rate limiter state.

        Parameters
        ----------
        handle : RateLimiterAcquisitionHandle
            The acquisition handle returned by the ``acquire`` method. This handle
            contains the estimated number of tokens consumed.
        actual_tokens : int
            The actual number of tokens consumed by the request.
        response_headers : dict[str, Any], optional, default=None
            The response headers from the request. If provided, the rate limit
            information will be parsed and used to update the rate limiter state.

        Raises
        ------
        ValueError
            If ``actual_tokens`` and ``handle.estimated_tokens`` are negative.

        """
        # input validation
        if actual_tokens < 0:
            raise ValueError("Actual tokens consumed must be non-negative.")
        if handle.estimated_tokens < 0:
            raise ValueError("Estimated tokens must be non-negative.")

        estimated_tokens = handle.estimated_tokens
        delta = estimated_tokens - actual_tokens
        is_failure_release = actual_tokens == 0  # request failed (e.g., 429 error)

        log_adjustment = (
            delta != 0 or response_headers is not None or is_failure_release
        )
        if log_adjustment:
            logging.debug(
                f"Adjusting usage: Est={estimated_tokens}, Act={actual_tokens}, "
                f"Delta={delta if not is_failure_release else 'N/A (Failure Release)'}, "
                f"Headers={'Yes' if response_headers else 'No'}"
            )
        elif not is_failure_release:
            return

        async with self._lock:
            current_time = time.monotonic()
            needs_notify = False

            if is_failure_release:
                for key, bucket in self._request_buckets.items():
                    bucket.adjust_bucket_level(1, current_time)
                    logging.debug(
                        f"Returned 1 request token to bucket '{key}' due to failure."
                    )
                needs_notify = True

            if delta != 0:
                for bucket in self._token_buckets.values():
                    bucket.adjust_bucket_level(delta, current_time)

                if delta > 0:  # tokens were returned
                    logging.debug(f"Returned {delta} tokens to token bucket(s).")
                    needs_notify = True

            # try to get rate limit information from the response headers
            # this will server as 'ground truth' for the rate limit state
            response_info = _parse_rate_limit_info_from_response_headers(
                response_headers, current_time
            )
            if response_info:

                def _sync_bucket_with_limits(
                    bucket: Optional[Bucket],
                    limit: Optional[int],
                    remaining: Optional[int],
                    reset_time: Optional[float],
                ) -> bool:
                    if bucket and reset_time is not None and remaining is not None:
                        bucket.sync_from_response_header(
                            server_token_limit=limit,
                            server_tokens_remaining=remaining,
                            server_reset_time=reset_time,
                            current_time=current_time,
                        )

                        return True
                    return False

                # sync client-server request-/token-per-day buckets
                needs_notify = (
                    _sync_bucket_with_limits(
                        self._request_buckets.get("rpd", None),
                        response_info["requests_limit"],
                        response_info["requests_remaining"],
                        response_info["requests_reset"],
                    )
                    or needs_notify
                )
                needs_notify = (
                    _sync_bucket_with_limits(
                        self._token_buckets.get("tpd", None),
                        response_info["tokens_limit"],
                        response_info["tokens_remaining"],
                        response_info["tokens_reset"],
                    )
                    or needs_notify
                )

            if needs_notify:
                logging.debug("State changed after adjustment, notifying waiters.")
                self._notify_waiters()

    def _get_bucket_token_pairs(
        self, estimated_tokens: int
    ) -> list[tuple[Bucket, int]]:
        """Get a list of (bucket, tokens) pairs for the given number of tokens."""
        bucket_token_pairs = []
        for bucket in self._token_buckets.values():
            bucket_token_pairs.append((bucket, estimated_tokens))
        for bucket in self._request_buckets.values():
            bucket_token_pairs.append((bucket, 1))  # each request costs 1 token
        return bucket_token_pairs

    def _has_capacity(
        self, bucket_token_pairs: list[tuple[Bucket, int]], current_time: float
    ) -> bool:
        """Check if all buckets have enough tokens for the request."""
        return all(
            bucket.get_bucket_level(current_time) >= tokens
            for bucket, tokens in bucket_token_pairs
        )

    def _consume_all(
        self, bucket_token_pairs: list[tuple[Bucket, int]], current_time: float
    ) -> None:
        """Consume tokens from all buckets."""
        for bucket, tokens in bucket_token_pairs:
            is_tokens_consumed = bucket.consume_tokens(tokens, current_time)

            if not is_tokens_consumed:
                logging.warning(
                    f"Failed to consume {tokens} tokens from bucket '{bucket}'."
                )

    def _notify_waiters(self) -> None:
        """Notify waiters that tokens are available."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            logging.error("Could not get running event loop to notify waiters.")
            return

        if self._timer_handle:  # reset the timer
            self._timer_handle.cancel()
            self._timer_handle = None

        current_time = time.monotonic()

        while self._waiters:
            tokens_needed, _, fut = self._waiters[0]

            if fut.done():  # remove cancelled or completed futures
                heapq.heappop(self._waiters)
                continue

            bucket_token_pairs = self._get_bucket_token_pairs(tokens_needed)
            if self._has_capacity(bucket_token_pairs, current_time):
                logging.debug(
                    f"Waking up waiter for ~{tokens_needed} tokens with available capacity."
                )
                heapq.heappop(self._waiters)
                self._consume_all(bucket_token_pairs, current_time)
                loop.call_soon(fut.set_result, True)
            else:
                # calculate the time to wait for the next token refill
                delay = 0.0
                for bucket, tokens_needed in bucket_token_pairs:
                    wait_needed = bucket.estimate_next_refill_time(
                        tokens_needed, current_time
                    )
                    delay = max(delay, wait_needed)

                logging.debug(
                    f"Waiter for ~{tokens_needed} tokens still waiting, "
                    f"next refill in {delay:.4f}s."
                )
                self._schedule_timer(delay)
                break

    def _schedule_timer(self, delay: float) -> None:
        """Schedule a timer to wake up the next waiter."""
        if self._timer_handle:
            self._timer_handle.cancel()
            self._timer_handle = None

        if delay > 0 and delay != float("inf"):
            try:
                loop = asyncio.get_running_loop()
                self._timer_handle = loop.call_later(delay, self._wake_next_timer)
                logging.debug(f"Scheduled timer for {delay:.4f}s")
            except RuntimeError:
                logging.error("Cannot schedule timer: No running event loop.")

    def _wake_next_timer(self) -> None:
        """Wake up the next waiter when the timer fires."""
        self._timer_handle = None

        async def do_notify():  # for acquiring lock in async context
            async with self._lock:
                logging.debug("Timer fired, notifying waiters.")
                self._notify_waiters()

        try:
            asyncio.create_task(do_notify())
        except RuntimeError:
            logging.warning(
                "Failed to create task for timer callback - loop likely closing."
            )


def _parse_rate_limit_info_from_response_headers(
    response_headers: Optional[dict[str, Any]], current_time: float
) -> Optional[dict[str, Optional[Union[int, float]]]]:
    """Parse rate limit information from response headers."""
    if response_headers is None:
        return None

    requests_limit: Optional[int] = None
    tokens_limit: Optional[int] = None
    requests_remaining: Optional[int] = None
    tokens_remaining: Optional[int] = None
    requests_reset: Optional[float] = None
    tokens_reset: Optional[float] = None

    for header, value in response_headers.items():
        key = header.lower()
        try:
            if "x-ratelimit-limit-requests" in key:
                requests_limit = int(value)
            elif "x-ratelimit-limit-tokens" in key:
                tokens_limit = int(value)
            elif "x-ratelimit-remaining-requests" in key:
                requests_remaining = int(value)
            elif "x-ratelimit-remaining-tokens" in key:
                tokens_remaining = int(value)
            elif "x-ratelimit-reset-requests" in key:
                requests_reset = _parse_reset_time(value, current_time)
            elif "x-ratelimit-reset-tokens" in key:
                tokens_reset = _parse_reset_time(value, current_time)
        except ValueError:
            logging.error(f"Failed to parse rate limit info: {header}={value}")
            continue

    return {
        "requests_limit": requests_limit,
        "tokens_limit": tokens_limit,
        "requests_remaining": requests_remaining,
        "tokens_remaining": tokens_remaining,
        "requests_reset": requests_reset,
        "tokens_reset": tokens_reset,
    }


def _parse_reset_time(header_value: str, current_time: float) -> Optional[float]:  # noqa: PLR0911, PLR0912
    """Parse x-ratelimit-reset-* header value into a future monotonic timestamp.

    Handles:
    - Seconds delta (e.g., "8.64", "15.0s")
    - Integer seconds delta (e.g., "15")
    - Unix timestamp (integer or float, assumed > typical small deltas)
    - Simple HH:MM:SS:MMM duration strings (e.g., "23:45:44.756")

    Returns
    -------
    float:
        Absolute monotonic timestamp for reset, or ``None`` if unparseable.
    """
    if not isinstance(header_value, str):
        return None

    header_value = header_value.strip()
    if not header_value:  # whitespace or empty string
        return None

    try:  # direct float conversion (seconds delta or Unix timestamp)
        value = float(header_value)
        if value > time.time() - (5 * 365 * 86400):
            # Heuristic: value is large, assume it's a future Unix timestamp
            # and convert to monotonic delay
            delay = max(0.0, value - time.time())  # delay from 'now' in wallclock
            return current_time + delay  # monotonic time

        return current_time + max(0.0, value)
    except ValueError:  # not a simple float, continue parsing
        pass

    # try parsing duration strings like '8.64s', '23h45m44.756s'
    total_seconds = 0.0

    pattern = re.compile(r"(\d+(?:\.\d+)?)(ms|h|m|s)", re.IGNORECASE)
    last_match_end = 0
    found_match = False

    for match in pattern.finditer(header_value):
        start, end = match.span()
        num_str, unit = match.groups()
        found_match = True

        # Ensure no non-whitespace characters exist *between* valid matches
        if start != last_match_end:
            logging.debug(
                f"Invalid duration format: Unexpected characters "
                f"'{header_value[last_match_end:start]}' before '{match.group()}'"
            )
            return None

        try:
            val = float(num_str)
        except ValueError:
            # Should not happen with the regex, but safeguard
            logging.debug(f"Invalid number '{num_str}' in duration string.")
            return None

        unit_lower = unit.lower()
        if unit_lower == "h":
            total_seconds += val * 3600
        elif unit_lower == "m":
            total_seconds += val * 60
        elif unit_lower == "ms":
            total_seconds += val / 1000.0
        elif unit_lower == "s":
            total_seconds += val
        # No 'else' needed as regex only matches valid units

        last_match_end = end

    # Ensure the entire string was consumed by the matches.
    # If last_match_end doesn't reach the end of the string, there's trailing text.
    if not found_match or last_match_end != len(header_value):
        if found_match:  # Only log trailing chars if we found at least one match
            logging.debug(
                f"Invalid duration format: Trailing characters "
                f"'{header_value[last_match_end:]}'"
            )
        return None  # No matches found or trailing characters exist

    # If we reach here, the entire string was parsed successfully as duration components
    return current_time + max(0.0, total_seconds)
