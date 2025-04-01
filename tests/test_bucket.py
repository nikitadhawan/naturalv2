import math
import time

import pytest

from naturalv2.models.rate_limiter._bucket import Bucket


@pytest.fixture
def simple_bucket():
    return Bucket(tokens_per_period=10)


@pytest.mark.parametrize(
    "tokens_per_period, capacity, time_period",
    [
        (0, 10, 1),
        (10, 0, 1),
        (10, 10, 0),
        (10, 10, -1),
    ],
)
def test_failed_initialization(tokens_per_period, capacity, time_period):
    with pytest.raises(ValueError, match="to be a positive integer"):
        Bucket(
            tokens_per_period=tokens_per_period,
            capacity=capacity,
            time_period=time_period,
        )


@pytest.mark.parametrize(
    "tokens_per_period, capacity, time_period",
    [
        (10, 10, 60),
        (10, 10, 1),
        (10, 20, 5),
    ],
)
def test_bucket_initialization(tokens_per_period, capacity, time_period):
    bucket = Bucket(
        tokens_per_period=tokens_per_period,
        capacity=capacity,
        time_period=time_period,
    )
    assert bucket.tokens_per_period == tokens_per_period
    assert bucket.capacity == capacity
    assert bucket.time_period == time_period
    assert bucket._tokens == float(capacity)
    assert bucket._rate_per_second == tokens_per_period / time_period
    assert math.isclose(bucket._last_refill_time, time.monotonic(), rel_tol=1e-6)
    assert bucket._last_sync_time == 0.0
    assert bucket._server_reset_time is None
    assert bucket._last_sync_rate is None


def test_consume_tokens(simple_bucket: Bucket):
    # test successful consumption of non-positive amount of tokens
    assert simple_bucket.consume_tokens(0) is True
    assert simple_bucket.consume_tokens(-1) is True

    # successful consumption of available tokens
    now = time.monotonic()
    success = simple_bucket.consume_tokens(5, now)
    assert success is True
    assert simple_bucket._tokens == float(5)
    assert simple_bucket._last_refill_time == now

    # unsuccessful consumption of more tokens than available
    success = simple_bucket.consume_tokens(6, now)
    assert success is False
    assert simple_bucket._tokens == float(5)
    assert simple_bucket._last_refill_time == now

    # check backwards time: no refill, but consumption should still work
    success = simple_bucket.consume_tokens(5, now - 1)
    assert success is True
    assert simple_bucket._tokens == 0
    assert simple_bucket._last_refill_time == now - 1

    # server reset time is set and it's in the past: full refill expected
    simple_bucket._server_reset_time = now - 1
    success = simple_bucket.consume_tokens(7, now)
    assert success is True
    assert simple_bucket._tokens == float(3)
    assert simple_bucket._last_sync_time == now
    assert simple_bucket._server_reset_time is None
    assert simple_bucket._last_sync_rate is None


def test_get_bucket_level(simple_bucket: Bucket):
    now = time.monotonic()
    assert simple_bucket.get_bucket_level(now) == 10

    simple_bucket.consume_tokens(5, now)
    assert simple_bucket.get_bucket_level(now) == 5

    simple_bucket.consume_tokens(5, now)
    assert simple_bucket.get_bucket_level(now) == 0

    simple_bucket.consume_tokens(5, now)
    assert simple_bucket.get_bucket_level(now) == 0

    # wait for refill: at least 1 token should be available after 6 seconds
    wait_time = (60 / 10) + 0.1
    simple_bucket._refill(now + wait_time)
    print(simple_bucket._tokens)
    assert simple_bucket.get_bucket_level(now + wait_time) == 1

    # wait to fill up the bucket
    assert simple_bucket.get_bucket_level(now + wait_time * 10) == 10


def test_adjust_bucket_level(simple_bucket: Bucket):
    now = time.monotonic()

    # adjust when full
    simple_bucket.adjust_bucket_level(5, now)
    assert simple_bucket._tokens == 10

    # zero delta
    simple_bucket.adjust_bucket_level(0, now)
    assert simple_bucket._tokens == 10

    # consume and adjust
    simple_bucket.consume_tokens(5, now)
    simple_bucket.adjust_bucket_level(-5, now)
    assert simple_bucket._tokens == 0

    # adjust down when empty
    simple_bucket.adjust_bucket_level(-5, now)
    assert simple_bucket._tokens == 0.0

    # adjust up when empty
    simple_bucket.adjust_bucket_level(5, now)
    assert simple_bucket._tokens == 5

    # adjust past capacity
    simple_bucket.adjust_bucket_level(20, now)
    assert simple_bucket._tokens == 10


def test_estimate_next_refill_time(simple_bucket: Bucket):
    now = time.monotonic()
    assert simple_bucket.estimate_next_refill_time(5, now) == 0.0

    simple_bucket.consume_tokens(10, now)
    assert simple_bucket._last_refill_time == now
    assert simple_bucket.estimate_next_refill_time(5, now) == pytest.approx(
        (5 / simple_bucket._rate_per_second) + 1e-9
    )

    # set effective refill rate to 1.0 (emulate info from server response headers)
    simple_bucket._last_sync_time = now - 10
    simple_bucket._last_sync_rate = 1.0
    simple_bucket._server_reset_time = None
    assert simple_bucket.estimate_next_refill_time(5, now) == pytest.approx(
        (5 / 1.0) + 1e-9
    )

    # server reset time is in the future: use it for refill time estimation
    simple_bucket._server_reset_time = now + 10
    assert simple_bucket.estimate_next_refill_time(5, now) == pytest.approx(10 + 1e-9)

    # server reset time is in the future, but less than refill time: use refill time
    simple_bucket._server_reset_time = now + 1
    assert simple_bucket.estimate_next_refill_time(5, now) == pytest.approx(
        (5 / 1.0) + 1e-9
    )

    # set effective rate to 0 and server reset time to None: infinite refill time
    simple_bucket._last_sync_rate = 0
    simple_bucket._server_reset_time = None
    assert simple_bucket.estimate_next_refill_time(5, now) == float("inf")

    # set effective rate to 0 and server reset time to the future: use server reset time
    simple_bucket._server_reset_time = now + 10
    assert simple_bucket.estimate_next_refill_time(5, now) == pytest.approx(10 + 1e-9)

    # last_sync_rate is past valid duration: use default rate
    simple_bucket._last_sync_time = now - 61
    simple_bucket._server_reset_time = None
    assert simple_bucket.estimate_next_refill_time(5, now) == pytest.approx(
        (5 / simple_bucket._rate_per_second) + 1e-9
    )


def test_sync_from_response_header(simple_bucket: Bucket):
    now = time.monotonic()

    # bad inputs
    with pytest.raises(
        ValueError,
        match="Expected ``server_tokens_remaining`` to be a non-negative integer",
    ):
        simple_bucket.sync_from_response_header(
            server_tokens_remaining=-1,
            server_reset_time=now,
            server_token_limit=None,
            current_time=now,
        )

    with pytest.raises(
        ValueError,
        match="Expected ``server_reset_time`` to be a non-negative float",
    ):
        simple_bucket.sync_from_response_header(
            server_tokens_remaining=10,
            server_reset_time=-1,
            server_token_limit=None,
            current_time=now,
        )

    with pytest.raises(
        ValueError,
        match="Expected ``server_token_limit`` to be a positive integer",
    ):
        simple_bucket.sync_from_response_header(
            server_tokens_remaining=10,
            server_reset_time=now,
            server_token_limit=0,
            current_time=now,
        )

    with pytest.raises(
        ValueError,
        match="Expected ``current_time`` to be a non-negative float",
    ):
        simple_bucket.sync_from_response_header(
            server_tokens_remaining=10,
            server_reset_time=now,
            server_token_limit=10,
            current_time=-1,
        )

    # server_tokens_remaining less than capacity
    simple_bucket.sync_from_response_header(
        server_tokens_remaining=5,
        server_reset_time=now,
        server_token_limit=10,
        current_time=now,
    )
    assert simple_bucket._tokens == 5
    assert simple_bucket._last_sync_time == now
    assert simple_bucket._last_sync_rate == (float(10) / 60)
    assert simple_bucket._server_reset_time == now

    # server_tokens_remaining is greater than capacity and there is no token limit
    simple_bucket.sync_from_response_header(
        server_tokens_remaining=15,
        server_reset_time=now + 3,
        server_token_limit=None,
        current_time=now,
    )
    assert simple_bucket._tokens == 10
    assert simple_bucket._last_sync_time == now
    assert simple_bucket._last_sync_rate is None
    assert simple_bucket._server_reset_time == now + 3
