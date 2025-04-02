import logging
import math
import time
from typing import Optional


class Bucket:
    """A fixed-capacity bucket that refills at a constant rate.

    This class implements the token bucket algorithm [1], which is a rate limiting
    algorithm that allows for bursts of traffic up to a certain limit. The
    bucket has a fixed capacity and refills at a constant rate. Tokens can be
    consumed from the bucket, and the bucket will refill over time.

    Parameters
    ----------
    tokens_per_period : int
        The number of tokens that are added to the bucket every ``time_period``
        seconds.
    capacity : int, optional, default=None
        The maximum number of tokens that the bucket can hold. If not provided,
        the capacity will be set to ``tokens_per_period``.
    time_period : int, optional, default=60
        The time period in seconds over which the tokens are added to the bucket.

    Raises
    ------
    ValueError
        If ``tokens_per_period``, ``capacity``, or ``time_period`` are non-positive.

    Warnings
    --------
    This class is not thread-safe and is therefore not intended to be used on its
    own. If the bucket is being accessed by multiple threads or processes, the
    caller should ensure that the bucket is locked before calling any of the public
    methods.

    References
    ----------
    [1] https://en.wikipedia.org/wiki/Token_bucket

    """

    SYNC_RATE_VALIDITY_DURATION = 60.0

    def __init__(
        self,
        tokens_per_period: int,
        capacity: Optional[int] = None,
        time_period: int = 60,
    ) -> None:
        if not isinstance(tokens_per_period, int) or tokens_per_period <= 0:
            raise ValueError(
                "Expected ``tokens_per_period`` to be a positive integer, "
                f"but got {tokens_per_period}"
            )
        if capacity is not None and (not isinstance(capacity, int) or capacity <= 0):
            raise ValueError(
                f"Expected ``capacity`` to be a positive integer, but got {capacity}"
            )
        if not isinstance(time_period, int) or time_period <= 0:
            raise ValueError(
                f"Expected ``time_period`` to be a positive integer, but got {time_period}"
            )

        self._tokens_per_period = tokens_per_period
        self._capacity = capacity or tokens_per_period
        self._time_period = time_period

        self._rate_per_second = tokens_per_period / time_period
        self._last_refill_time = time.monotonic()
        self._tokens = float(self.capacity)

        self._last_sync_time: float = 0.0
        self._server_reset_time: Optional[float] = None
        self._last_sync_rate: Optional[float] = None

    @property
    def capacity(self) -> int:
        """Return the maximum number of tokens that the bucket can hold."""
        return self._capacity

    @property
    def tokens_per_period(self) -> int:
        """Return the number of tokens that are added to the bucket every time period."""
        return self._tokens_per_period

    @property
    def time_period(self) -> int:
        """Return the time period in seconds over which the tokens are added to the bucket."""
        return self._time_period

    def get_bucket_level(self, current_time: Optional[float] = None) -> int:
        """Return the current number of tokens in the bucket.

        Parameters
        ----------
        current_time : float, optional, default=None
            The current time in seconds since the epoch. If not provided, the
            current time will be determined using ``time.monotonic()``.

        Returns
        -------
        int
            The current number of tokens in the bucket.

        Warnings
        --------
        This method assumes that the caller has acquired the global lock on the
        bucket. If the bucket is being accessed by multiple threads or processes,
        the caller should ensure that the bucket is locked before calling this
        method.
        """
        current_time = current_time or time.monotonic()
        self._refill(current_time)
        return math.floor(self._tokens)

    def consume_tokens(
        self, num_tokens_needed: int, current_time: Optional[float] = None
    ) -> bool:
        """Consume tokens from the bucket.

        Parameters
        ----------
        num_tokens_needed : int
            The number of tokens to consume from the bucket.
        current_time : float, optional, default=None
            The current time in seconds since the epoch. If not provided, the
            current time will be determined using ``time.monotonic()``.

        Returns
        -------
        bool
            ``True`` if the tokens were successfully consumed from the bucket,
            otherwise ``False``.

        Warnings
        --------
        This method assumes that the caller has acquired the global lock on the
        bucket. If the bucket is being accessed by multiple threads or processes,
        the caller should ensure that the bucket is locked before calling this
        method.
        """
        if num_tokens_needed <= 0:
            return True

        current_time = current_time or time.monotonic()

        if self.get_bucket_level(current_time) >= num_tokens_needed:
            self._tokens -= float(num_tokens_needed)

            # reset sync state since we've consumed tokens
            self._last_sync_rate = None
            self._server_reset_time = None
            return True
        return False

    def adjust_bucket_level(
        self, delta: int, current_time: Optional[float] = None
    ) -> None:
        """Adjust the number of tokens in the bucket.

        This method is useful for adjusting the number of tokens in the bucket
        using information from the server response headers.

        Parameters
        ----------
        delta : int
            The number of tokens to add or remove from the bucket.
        current_time : float, optional, default=None
            The current time in seconds since the epoch. If not provided, the
            current time will be determined using ``time.monotonic()``.

        Warnings
        --------
        This method assumes that the caller has acquired the global lock on the
        bucket. If the bucket is being accessed by multiple threads or processes,
        the caller should ensure that the bucket is locked before calling this
        method.
        """
        if delta == 0:
            return

        current_time = current_time or time.monotonic()
        self._refill(current_time)

        new_tokens = self._tokens + delta
        self._tokens = max(0.0, min(self.capacity, new_tokens))

        self._last_sync_rate = None
        self._server_reset_time = None

    def estimate_next_refill_time(
        self, num_tokens_needed: int, current_time: Optional[float] = None
    ) -> float:
        """Estimate the time until the bucket has enough tokens.

        This method is useful for determining how long to wait before the bucket
        has enough tokens to satisfy a request. It can use the server refill time
        if it is known, or estimate the refill rate based on the last known rate.

        Parameters
        ----------
        num_tokens_needed : int
            The number of tokens needed to satisfy the request.
        current_time : float, optional, default=None
            The current time in seconds since the epoch. If not provided, the
            current time will be determined using ``time.monotonic()``.

        Returns
        -------
        float
            The estimated time in seconds until the bucket has enough tokens.

        Warnings
        --------
        This method assumes that the caller has acquired the global lock on the
        bucket. If the bucket is being accessed by multiple threads or processes,
        the caller should ensure that the bucket is locked before calling this
        method.
        """
        current_time = current_time or time.monotonic()

        avilable_tokens = self.get_bucket_level(current_time)
        shortfall = num_tokens_needed - avilable_tokens

        if shortfall <= 0:  # no wait needed
            return 0.0

        effective_rate = self._get_effective_rate(current_time)
        if effective_rate <= 0:
            if (
                self._server_reset_time is not None
                and self._server_reset_time > current_time
            ):
                logging.debug(
                    f"Rate is zero, but server reset time known: {self._server_reset_time:.2f}"
                )
                wait_needed = self._server_reset_time - current_time
                wait_time = max(0.0, wait_needed) + 1e-9
                logging.debug(
                    f"Calculated wait based on server reset time: {wait_time:.2f}s"
                )
                return wait_time

            logging.warning(
                "Bucket refill rate is zero and no future reset time known, "
                "will wait indefinitely."
            )
            return float("inf")

        time_needed = (shortfall / effective_rate) + 1e-9
        logging.debug(
            f"Calculating wait based on effective rate {effective_rate:.2f}: {time_needed:.2f}s"
        )

        if (
            self._server_reset_time is not None
            and self._server_reset_time > current_time
        ):  # server reset time is in the future, don't wake before it
            time_until_reset = self._server_reset_time - current_time
            if time_needed < time_until_reset and avilable_tokens <= 0:
                logging.debug(
                    f"Adjusting wait time to server reset time: {time_until_reset:.2f}s"
                )
                return max(0.0, time_until_reset) + 1e-9

        return time_needed

    def sync_from_response_header(
        self,
        server_tokens_remaining: int,
        server_reset_time: float,
        server_token_limit: Optional[int],
        current_time: Optional[float] = None,
    ) -> None:
        """Directly sets token count and resets refill timer.

        This method uses information from the server response headers to directly
        set the number of tokens in the bucket and reset the refill timer. It also
        estimates the server refill rate based on the response headers.

        This typically only gets triggered for daily rate limits, per OpenAI's API
        response headers.

        Parameters
        ----------
        server_tokens_remaining : int
            The number of tokens remaining in the server bucket.
        server_reset_time : float
            The time at which the server will reset the token count.
         server_token_limit : int, optional
            The maximum number of tokens that the server can provide in the time period
            (typically daily).
        current_time : float, optional, default=None
            The current time in seconds since the epoch. If not provided, the
            current time will be determined using ``time.monotonic()``.

        Raises
        ------
        ValueError
            If ``server_tokens_remaining`` is negative, ``server_reset_time`` is
            negative, or ``server_token_limit`` is non-positive.

        """
        # input validation
        if not isinstance(server_tokens_remaining, int) or server_tokens_remaining < 0:
            raise ValueError(
                f"Expected ``server_tokens_remaining`` to be a non-negative integer, "
                f"but got {server_tokens_remaining}"
            )
        if not isinstance(server_reset_time, float) or server_reset_time < 0:
            raise ValueError(
                f"Expected ``server_reset_time`` to be a non-negative float, "
                f"but got {server_reset_time}"
            )
        if server_token_limit is not None and server_token_limit <= 0:
            raise ValueError(
                f"Expected ``server_token_limit`` to be a positive integer, "
                f"but got {server_token_limit}"
            )
        if current_time is not None and current_time < 0:
            raise ValueError(
                f"Expected ``current_time`` to be a non-negative float, "
                f"but got {current_time}"
            )

        current_time = current_time or time.monotonic()

        self._tokens = max(0.0, min(self.capacity, float(server_tokens_remaining)))
        self._last_sync_time = current_time
        self._server_reset_time = server_reset_time

        if (
            server_token_limit is not None
            and server_token_limit > 0
            and self.time_period > 0
        ):
            self._last_sync_rate = float(server_token_limit) / self.time_period
            logging.debug(
                f"Estimated server refill rate at: {self._last_sync_rate:.2f}/s "
                "based on response headers."
            )
        else:
            self._last_sync_rate = None
            logging.debug("Could not estimate server rate from sync headers.")

    def _get_effective_rate(self, current_time: float) -> float:
        """Return the effective refill rate of the bucket.

        The effective rate is the server refill rate if known, or the local rate
        if the server rate is not known.
        """
        if (
            self._last_sync_rate is not None
            and (current_time - self._last_sync_time) < self.SYNC_RATE_VALIDITY_DURATION
        ):
            logging.debug("Using estimated server refill rate.")
            return self._last_sync_rate

        if self._last_sync_rate is not None:
            logging.debug("Sync rate expired, reverting to default, local rate.")
            self._last_sync_rate = None

        return self._rate_per_second

    def _refill(self, current_time: float) -> None:
        """Refill the bucket with tokens.

        This method refills the bucket with tokens based on the current time and
        the refill rate. The bucket will not be refilled if the current time is
        before the last refill time. It will be refilled to the maximum capacity
        if the server reset time has passed.
        """
        if (
            self._server_reset_time is not None
            and current_time >= self._server_reset_time
        ):
            logging.debug(
                f"Server reset time passed at {current_time:.2f}. Resetting bucket to full."
            )
            self._tokens = float(self.capacity)
            self._last_sync_time = current_time
            self._server_reset_time = None
            self._last_sync_rate = None
            return

        elapsed_time = current_time - self._last_refill_time

        if elapsed_time < 0:
            self._last_refill_time = current_time
            return

        tokens_to_add = elapsed_time * self._get_effective_rate(current_time)

        if tokens_to_add > 0:
            self._tokens = min(self.capacity, self._tokens + tokens_to_add)
            self._last_refill_time = current_time
