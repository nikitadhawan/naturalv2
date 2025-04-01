import asyncio
import contextlib
import time
from unittest import mock

import pytest

from naturalv2.models.rate_limiter.rate_limiter import (
    RateLimiter,
    RateLimiterAcquisitionHandle,
    _parse_rate_limit_info_from_response_headers,
    _parse_reset_time,
)

from .time_mock import MockTime


RPM_LIMIT = 120
TPM_LIMIT = 2_000
RPD_LIMIT = 1_000
TPD_LIMIT = 10_000
MAX_REQ_BURST = 60
MAX_TOKEN_BURST = 1_000


@pytest.fixture
def mock_time():
    """Pytest fixture providing an encapsulated mock time environment."""
    mock_time_instance = MockTime()

    module_path = "naturalv2.models.rate_limiter.rate_limiter"
    bucket_module_path = "naturalv2.models.rate_limiter._bucket"

    # patch time.monotonic and asyncio.get_running_loop where they are used
    patches: list[mock._patcher] = [
        mock.patch(f"{module_path}.time.monotonic", new=mock_time_instance.monotonic),
        mock.patch(
            f"{module_path}.asyncio.get_running_loop",
            new=mock_time_instance.get_running_loop,
        ),
        mock.patch(
            f"{bucket_module_path}.time.monotonic", new=mock_time_instance.monotonic
        ),
    ]

    try:
        for p in patches:
            p.start()
        yield mock_time_instance  # provide the mock time instance to the test
    finally:
        for p in patches:
            try:
                p.stop()
            except RuntimeError as e:
                print(f"Failed to stop patch: {e}")
                raise e


@pytest.fixture
def simple_rate_limiter():
    return RateLimiter(requests_per_minute=RPM_LIMIT)


@pytest.fixture
def multi_bucket_rate_limiter():
    return RateLimiter(
        requests_per_minute=RPM_LIMIT,
        requests_per_day=RPD_LIMIT,
        tokens_per_minute=TPM_LIMIT,
        tokens_per_day=TPD_LIMIT,
        max_request_burst=MAX_REQ_BURST,
        max_token_burst=MAX_TOKEN_BURST,
    )


@pytest.mark.parametrize(
    "rpm, tpm, rpd, tpd, max_req_burst, max_token_burst",
    [
        (0, 10, 100, 1000, 10, 100),
        (10, 0, 100, 1000, 10, 100),
        (10, 10, 0, 1000, 10, 100),
        (10, 10, 100, 0, 10, 100),
        (10, 10, 100, 1000, 0, 100),
        (10, 10, 100, 1000, 10, 0),
        (10, 10, 100, 1000, -1, 100),
        (10, 10, 100, 1000, 10, -1),
    ],
)
def test_failed_initialization(rpm, tpm, rpd, tpd, max_req_burst, max_token_burst):
    with pytest.raises(ValueError, match="to be a positive integer"):
        RateLimiter(
            requests_per_minute=rpm,
            tokens_per_minute=tpm,
            requests_per_day=rpd,
            tokens_per_day=tpd,
            max_request_burst=max_req_burst,
            max_token_burst=max_token_burst,
        )


def test_attributes(multi_bucket_rate_limiter: RateLimiter):
    assert multi_bucket_rate_limiter.requests_per_minute == RPM_LIMIT
    assert multi_bucket_rate_limiter.tokens_per_minute == TPM_LIMIT
    assert multi_bucket_rate_limiter.requests_per_day == RPD_LIMIT
    assert multi_bucket_rate_limiter.tokens_per_day == TPD_LIMIT
    assert multi_bucket_rate_limiter.max_request_burst == MAX_REQ_BURST
    assert multi_bucket_rate_limiter.max_token_burst == MAX_TOKEN_BURST

    assert "rpm" in multi_bucket_rate_limiter._request_buckets
    assert "tpm" in multi_bucket_rate_limiter._token_buckets
    assert "rpd" in multi_bucket_rate_limiter._request_buckets
    assert "tpd" in multi_bucket_rate_limiter._token_buckets

    assert (
        multi_bucket_rate_limiter._request_buckets["rpm"].tokens_per_period == RPM_LIMIT
    )
    assert (
        multi_bucket_rate_limiter._token_buckets["tpm"].tokens_per_period == TPM_LIMIT
    )
    assert (
        multi_bucket_rate_limiter._request_buckets["rpd"].tokens_per_period == RPD_LIMIT
    )
    assert (
        multi_bucket_rate_limiter._token_buckets["tpd"].tokens_per_period == TPD_LIMIT
    )

    assert multi_bucket_rate_limiter._request_buckets["rpm"].capacity == MAX_REQ_BURST
    assert multi_bucket_rate_limiter._token_buckets["tpm"].capacity == MAX_TOKEN_BURST
    assert multi_bucket_rate_limiter._request_buckets["rpd"].capacity == RPD_LIMIT
    assert multi_bucket_rate_limiter._token_buckets["tpd"].capacity == TPD_LIMIT

    assert multi_bucket_rate_limiter._request_buckets["rpm"].time_period == 60
    assert multi_bucket_rate_limiter._token_buckets["tpm"].time_period == 60
    assert multi_bucket_rate_limiter._request_buckets["rpd"].time_period == 86400
    assert multi_bucket_rate_limiter._token_buckets["tpd"].time_period == 86400


@pytest.mark.asyncio
async def test_over_acquire(multi_bucket_rate_limiter: RateLimiter):
    with pytest.raises(
        ValueError,
        match=f"Requested number of tokens {TPM_LIMIT + 1} exceeds the capacity",
    ):
        # exceed the capacity of the tpm bucket
        await multi_bucket_rate_limiter.acquire(TPM_LIMIT + 1)

    with pytest.raises(
        ValueError,
        match=f"Requested number of tokens {TPD_LIMIT + 1} exceeds the capacity",
    ):
        # exceed the capacity of the tpd bucket
        await multi_bucket_rate_limiter.acquire(TPD_LIMIT + 1)


@pytest.mark.asyncio
async def test_acquire_immediate(simple_rate_limiter: RateLimiter):
    current_time = time.monotonic()
    assert (
        simple_rate_limiter._request_buckets["rpm"].get_bucket_level(current_time)
        == RPM_LIMIT
    )

    handle = await simple_rate_limiter.acquire(256)
    assert isinstance(handle, RateLimiterAcquisitionHandle)
    assert handle.estimated_tokens == 256

    assert (
        simple_rate_limiter._request_buckets["rpm"].get_bucket_level(current_time)
        == RPM_LIMIT - 1
    )


@pytest.mark.asyncio
async def test_acquire_blocks_on_requests(
    multi_bucket_rate_limiter: RateLimiter, mock_time: MockTime
):
    rpm_bucket_ref = multi_bucket_rate_limiter._request_buckets["rpm"]
    tasks = [  # consume all available request tokens in a burst
        asyncio.create_task(multi_bucket_rate_limiter.acquire(10))
        for _ in range(MAX_REQ_BURST)
    ]
    await asyncio.gather(*tasks)

    assert rpm_bucket_ref.get_bucket_level(mock_time.monotonic()) == pytest.approx(0.0)

    # next request should be blocked
    blocked_request = asyncio.create_task(multi_bucket_rate_limiter.acquire(10))
    await asyncio.sleep(0)  # allow the task to be scheduled
    assert not blocked_request.done()
    assert len(multi_bucket_rate_limiter._waiters) == 1

    # advance time, but not enough to unblock the request
    await mock_time.advance_time((60 / RPM_LIMIT) - 0.2)
    assert not blocked_request.done()
    assert len(multi_bucket_rate_limiter._waiters) == 1
    assert rpm_bucket_ref.get_bucket_level(mock_time.monotonic()) < 1

    # advance time to unblock the request (1 token)
    await mock_time.advance_time(0.25)
    assert blocked_request.done()
    await blocked_request
    assert len(multi_bucket_rate_limiter._waiters) == 0

    total_time_passed = (60 / RPM_LIMIT) + 0.05
    rpm_bucket_ref._refill(mock_time.monotonic())
    assert rpm_bucket_ref._tokens == pytest.approx(
        (total_time_passed * (RPM_LIMIT / 60)) - 1
    )


@pytest.mark.asyncio
async def test_acquire_blocks_on_tokens(
    multi_bucket_rate_limiter: RateLimiter, mock_time: MockTime
):
    rpm_bucket_ref = multi_bucket_rate_limiter._request_buckets["rpm"]
    tpm_bucket_ref = multi_bucket_rate_limiter._token_buckets["tpm"]

    handle = await multi_bucket_rate_limiter.acquire(MAX_TOKEN_BURST - 100)
    assert isinstance(handle, RateLimiterAcquisitionHandle)
    assert handle.estimated_tokens == MAX_TOKEN_BURST - 100
    assert tpm_bucket_ref.get_bucket_level(mock_time.monotonic()) == pytest.approx(100)
    assert rpm_bucket_ref.get_bucket_level(mock_time.monotonic()) == pytest.approx(
        MAX_REQ_BURST - 1
    )

    # next task will need 200 tokens, but only 100 are available
    blocked_request = asyncio.create_task(multi_bucket_rate_limiter.acquire(200))
    await asyncio.sleep(0)  # allow the task to be scheduled
    assert not blocked_request.done()
    assert len(multi_bucket_rate_limiter._waiters) == 1

    # advance time, but not enough to unblock the request
    time_to_fill = 100 * (60 / TPM_LIMIT)
    await mock_time.advance_time(time_to_fill - 0.2)
    assert not blocked_request.done()
    assert len(multi_bucket_rate_limiter._waiters) == 1
    assert tpm_bucket_ref.get_bucket_level(mock_time.monotonic()) < 200

    # advance time to unblock the request (200 tokens available)
    await mock_time.advance_time(0.3)
    assert blocked_request.done()
    await blocked_request
    assert len(multi_bucket_rate_limiter._waiters) == 0

    total_time_passed = time_to_fill + 0.1
    assert tpm_bucket_ref.get_bucket_level(mock_time.monotonic()) == pytest.approx(
        int(100 + (total_time_passed * (TPM_LIMIT / 60)) - 200)
    )
    assert rpm_bucket_ref.get_bucket_level(mock_time.monotonic()) == pytest.approx(
        MAX_REQ_BURST - 1
    )


@pytest.mark.asyncio
async def test_multiple_waiters(
    multi_bucket_rate_limiter: RateLimiter, mock_time: MockTime
):
    rpm_bucket_ref = multi_bucket_rate_limiter._request_buckets["rpm"]

    tasks = [  # consume all available request tokens in a burst
        asyncio.create_task(multi_bucket_rate_limiter.acquire(1))
        for _ in range(MAX_REQ_BURST)
    ]
    await asyncio.gather(*tasks)
    assert rpm_bucket_ref.get_bucket_level(mock_time.monotonic()) == pytest.approx(0.0)

    # queue waiters with different token requirements (lower tokens == higher priority)
    medium_request = asyncio.create_task(multi_bucket_rate_limiter.acquire(50))
    await asyncio.sleep(0)  # allow the task to be scheduled
    small_request = asyncio.create_task(multi_bucket_rate_limiter.acquire(10))
    await asyncio.sleep(0)  # allow the task to be scheduled
    large_request = asyncio.create_task(multi_bucket_rate_limiter.acquire(150))
    await asyncio.sleep(0)  # allow the task to be scheduled

    assert (
        not small_request.done()
        and not medium_request.done()
        and not large_request.done()
    )
    assert len(multi_bucket_rate_limiter._waiters) == 3
    assert multi_bucket_rate_limiter._waiters[0][0] == 10  # small_request on top

    # advance time for one request to be unblocked
    await mock_time.advance_time((60 / RPM_LIMIT) + 0.1)
    assert small_request.done()  # small_request should be done
    assert not medium_request.done() and not large_request.done()
    assert len(multi_bucket_rate_limiter._waiters) == 2
    assert multi_bucket_rate_limiter._waiters[0][0] == 50  # medium_request on top
    assert rpm_bucket_ref.get_bucket_level(mock_time.monotonic()) == pytest.approx(0.0)

    # advance time for another request to be unblocked
    await mock_time.advance_time(60 / RPM_LIMIT)
    assert medium_request.done()  # medium_request should be done
    assert not large_request.done()
    assert len(multi_bucket_rate_limiter._waiters) == 1
    assert multi_bucket_rate_limiter._waiters[0][0] == 150  # large_request on top
    assert rpm_bucket_ref.get_bucket_level(mock_time.monotonic()) == pytest.approx(0.0)

    # advance time for the last request to be unblocked
    await mock_time.advance_time(60 / RPM_LIMIT)
    assert large_request.done()  # large_request should be done
    assert len(multi_bucket_rate_limiter._waiters) == 0
    assert rpm_bucket_ref.get_bucket_level(mock_time.monotonic()) == pytest.approx(0.0)


@pytest.mark.asyncio
async def test_adjust_return_tokens(
    multi_bucket_rate_limiter: RateLimiter, mock_time: MockTime
):
    tpm_bucket_ref = multi_bucket_rate_limiter._token_buckets["tpm"]

    handle = await multi_bucket_rate_limiter.acquire(MAX_TOKEN_BURST - 100)
    assert isinstance(handle, RateLimiterAcquisitionHandle)
    assert handle.estimated_tokens == MAX_TOKEN_BURST - 100
    assert tpm_bucket_ref.get_bucket_level(mock_time.monotonic()) == pytest.approx(100)

    # next task will need 200 tokens, but only 100 are available
    blocked_request = asyncio.create_task(multi_bucket_rate_limiter.acquire(200))
    await asyncio.sleep(0)
    assert not blocked_request.done()

    # return 100 tokens to the bucket
    await multi_bucket_rate_limiter.adjust(handle, actual_tokens=800)
    await asyncio.sleep(0)  # allow the task to be scheduled

    # blocked request needs 100 tokens to be unblocked, capacity should now exist
    assert blocked_request.done()
    await blocked_request

    assert tpm_bucket_ref.get_bucket_level(mock_time.monotonic()) == pytest.approx(0.0)
    assert multi_bucket_rate_limiter._request_buckets["rpm"].get_bucket_level(
        mock_time.monotonic()
    ) == pytest.approx(MAX_REQ_BURST - 2)  # two request tokens were consumed


@pytest.mark.asyncio
async def test_adjust_failed_request(
    multi_bucket_rate_limiter: RateLimiter, mock_time: MockTime
):
    rpm_bucket_ref = multi_bucket_rate_limiter._request_buckets["rpm"]

    handles = []
    for _ in range(MAX_REQ_BURST):
        handle = await multi_bucket_rate_limiter.acquire(1)
        handles.append(handle)
    assert rpm_bucket_ref.get_bucket_level(mock_time.monotonic()) == pytest.approx(0.0)

    blocked_request = asyncio.create_task(multi_bucket_rate_limiter.acquire(1))
    await asyncio.sleep(0)
    assert not blocked_request.done()

    # adjust the first handle as a failed request (actual_tokens == 0)
    await multi_bucket_rate_limiter.adjust(handles[0], actual_tokens=0)
    await asyncio.sleep(0)

    # blocked_request should now be unblocked
    assert blocked_request.done()
    await blocked_request

    assert rpm_bucket_ref.get_bucket_level(mock_time.monotonic()) == pytest.approx(0.0)
    assert multi_bucket_rate_limiter._token_buckets["tpm"].get_bucket_level(
        mock_time.monotonic()
    ) == pytest.approx(MAX_TOKEN_BURST - MAX_REQ_BURST)


@pytest.mark.asyncio
async def test_adjust_negative_delta(
    multi_bucket_rate_limiter: RateLimiter, mock_time: MockTime
):
    tpm_bucket_ref = multi_bucket_rate_limiter._token_buckets["tpm"]

    handle = await multi_bucket_rate_limiter.acquire(MAX_TOKEN_BURST - 100)
    assert isinstance(handle, RateLimiterAcquisitionHandle)
    assert handle.estimated_tokens == MAX_TOKEN_BURST - 100
    assert tpm_bucket_ref.get_bucket_level(mock_time.monotonic()) == pytest.approx(100)

    # adjust to consume 50 more tokens than estimated
    await multi_bucket_rate_limiter.adjust(handle, actual_tokens=MAX_TOKEN_BURST - 50)
    await asyncio.sleep(0)
    assert tpm_bucket_ref.get_bucket_level(mock_time.monotonic()) == pytest.approx(50.0)

    second_handle = await multi_bucket_rate_limiter.acquire(25)
    assert isinstance(second_handle, RateLimiterAcquisitionHandle)
    assert second_handle.estimated_tokens == 25
    assert tpm_bucket_ref.get_bucket_level(mock_time.monotonic()) == pytest.approx(25.0)

    # adjust to consume more tokens into the negative
    await multi_bucket_rate_limiter.adjust(second_handle, actual_tokens=75)
    await asyncio.sleep(0)
    assert tpm_bucket_ref.get_bucket_level(mock_time.monotonic()) == pytest.approx(0.0)

    # verify refill after the negative adjustment
    await mock_time.advance_time((60 / TPM_LIMIT))
    assert tpm_bucket_ref.get_bucket_level(mock_time.monotonic()) == pytest.approx(1.0)


@pytest.mark.asyncio
async def test_adjust_with_response_headers(
    multi_bucket_rate_limiter: RateLimiter, mock_time: MockTime
):
    rpd_bucket_ref = multi_bucket_rate_limiter._request_buckets["rpd"]
    tpd_bucket_ref = multi_bucket_rate_limiter._token_buckets["tpd"]

    handle = await multi_bucket_rate_limiter.acquire(MAX_TOKEN_BURST)
    assert rpd_bucket_ref.get_bucket_level(mock_time.monotonic()) == pytest.approx(
        RPD_LIMIT - 1
    )
    assert tpd_bucket_ref.get_bucket_level(mock_time.monotonic()) == pytest.approx(
        TPD_LIMIT - MAX_TOKEN_BURST
    )

    # simulate server response headers indicating different state
    server_reset_tokens = mock_time.monotonic() + 6000.0
    headers = {
        "x-ratelimit-limit-requests": str(RPD_LIMIT),
        "x-ratelimit-remaining-requests": str(100),
        "x-ratelimit-reset-requests": "5000s",  # relative time string -> 5000s delta
        "x-ratelimit-limit-tokens": str(TPD_LIMIT + 5000),  # server says more TPD
        "x-ratelimit-remaining-tokens": str(TPD_LIMIT - MAX_TOKEN_BURST + 1000),
        "x-ratelimit-reset-tokens": str(server_reset_tokens),  # absolute time string
    }

    # adjust with delta=0 to update the headers
    await multi_bucket_rate_limiter.adjust(
        handle, actual_tokens=MAX_TOKEN_BURST, response_headers=headers
    )
    await asyncio.sleep(0)  # allow the task to run

    current_time = mock_time.monotonic()
    rpd_level_after_sync = rpd_bucket_ref.get_bucket_level(current_time)
    tpd_level_after_sync = tpd_bucket_ref.get_bucket_level(current_time)

    assert rpd_level_after_sync == pytest.approx(100)
    assert rpd_bucket_ref.capacity == RPD_LIMIT  # bucket capacity does not change
    assert tpd_level_after_sync == pytest.approx(TPD_LIMIT - MAX_TOKEN_BURST + 1000)
    assert tpd_bucket_ref.capacity == TPD_LIMIT  # capacity does not change


@pytest.mark.asyncio
async def test_cancel_waiters(
    multi_bucket_rate_limiter: RateLimiter, mock_time: MockTime
):
    rpm_bucket_ref = multi_bucket_rate_limiter._request_buckets["rpm"]
    tasks = [  # consume all available request tokens in a burst
        asyncio.create_task(multi_bucket_rate_limiter.acquire(10))
        for _ in range(MAX_REQ_BURST)
    ]
    await asyncio.gather(*tasks)
    assert rpm_bucket_ref.get_bucket_level(mock_time.monotonic()) == pytest.approx(0.0)

    large_request = asyncio.create_task(multi_bucket_rate_limiter.acquire(100))
    await asyncio.sleep(0)
    medium_request = asyncio.create_task(multi_bucket_rate_limiter.acquire(50))
    await asyncio.sleep(0)

    assert not large_request.done() and not medium_request.done()
    assert len(multi_bucket_rate_limiter._waiters) == 2
    assert multi_bucket_rate_limiter._waiters[0][0] == 50  # medium_request on top

    # cancel the medium_request
    medium_request.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await medium_request

    # advance time to process cancellation
    await mock_time.advance_time(0.1)
    assert len(multi_bucket_rate_limiter._waiters) == 1
    assert multi_bucket_rate_limiter._waiters[0][0] == 100

    # advance time to unblock the large_request
    await mock_time.advance_time(60 / RPM_LIMIT)
    assert large_request.done()
    await large_request
    assert len(multi_bucket_rate_limiter._waiters) == 0


@pytest.mark.parametrize(
    "value, expected_delta",
    [
        ("10", 10.0),
        ("15.5", 15.5),
        ("8.64s", 8.64),
        ("1m", 60.0),
        ("1.5m", 90.0),
        ("1h", 3600.0),
        ("100ms", 0.1),
        ("1m30s", 90.0),
        (" 2m ", 120.0),
        ("1h15m10s500ms", 4510.5),
        ("0", 0.0),
        ("-5", 0.0),  # Negative/zero delta -> 0 delay
    ],
)
def test_parse_reset_time_valid(value, expected_delta):
    current_time = time.monotonic()
    expected_time = current_time + expected_delta
    parsed_time = _parse_reset_time(value, current_time)
    assert parsed_time == pytest.approx(expected_time)


@pytest.mark.parametrize(
    "value",
    [
        ("abc"),
        ("10 seconds"),
        ("1 h"),
        ("1m 1d"),
        ("1m 30s"),
        ("1h 15m 10.5s"),
        ("1.2.3"),
        (""),
        (None),
    ],
)
def test_parse_reset_time_invalid(value):
    current_time = time.monotonic()

    value_str = str(value) if value is not None else ""
    if not value_str:  # Handle empty string case directly if needed
        assert _parse_reset_time(value_str, current_time) is None
    else:
        parsed = _parse_reset_time(value_str, current_time)
        assert parsed is None


def test_parse_rate_limit_headers_complete():
    current_time = time.monotonic()

    reset_req_delta = 60.0
    reset_tok_delta = 120.5
    headers = {
        "X-RateLimit-Limit-Requests": "100",
        "X-RateLimit-Limit-Tokens": "10000",
        "x-ratelimit-remaining-requests": "50",
        "x-ratelimit-remaining-tokens": "5000",
        "X-RateLimit-Reset-Requests": f"{reset_req_delta}",  # Delta seconds
        "X-RATELIMIT-RESET-TOKENS": f"{reset_tok_delta}s",  # Delta seconds with unit
    }
    expected = {
        "requests_limit": 100,
        "tokens_limit": 10000,
        "requests_remaining": 50,
        "tokens_remaining": 5000,
        "requests_reset": current_time + reset_req_delta,  # Expected monotonic time
        "tokens_reset": current_time + reset_tok_delta,  # Expected monotonic time
    }
    parsed = _parse_rate_limit_info_from_response_headers(headers, current_time)

    # Use approx for float comparison of reset times
    assert parsed["requests_limit"] == expected["requests_limit"]
    assert parsed["tokens_limit"] == expected["tokens_limit"]
    assert parsed["requests_remaining"] == expected["requests_remaining"]
    assert parsed["tokens_remaining"] == expected["tokens_remaining"]
    assert parsed["requests_reset"] == pytest.approx(expected["requests_reset"])
    assert parsed["tokens_reset"] == pytest.approx(expected["tokens_reset"])
