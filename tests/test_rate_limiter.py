import asyncio

import pytest

from naturalv2.models.rate_limiter.rate_limiter import RateLimiter


async def wait_for_n_done(tasks, n):
    """Wait for n (or more) tasks to have completed"""
    iteration = 0
    remainder = len(tasks) - n
    pending_count = len(tasks)
    while iteration <= 5:
        iteration += 1
        _, pending = await asyncio.wait(
            tasks, timeout=0, return_when=asyncio.FIRST_COMPLETED
        )
        if len(pending) <= remainder:
            break
        if len(pending) < pending_count:
            iteration = 0
            pending_count = len(pending)
    assert len(pending) <= remainder
    return pending


async def acquire_task(limiter: RateLimiter):
    await limiter.acquire(10)


@pytest.fixture
def simple_rate_limiter():
    return RateLimiter(requests_per_minute=360)


@pytest.fixture
def multi_bucket_rate_limiter():
    return RateLimiter(
        requests_per_minute=10,
        requests_per_day=100,
        tokens_per_minute=100,
        tokens_per_day=1000,
        max_request_burst=5,
        max_token_burst=50,
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
    assert multi_bucket_rate_limiter.requests_per_minute == 10
    assert multi_bucket_rate_limiter.tokens_per_minute == 100
    assert multi_bucket_rate_limiter.requests_per_day == 100
    assert multi_bucket_rate_limiter.tokens_per_day == 1000
    assert multi_bucket_rate_limiter.max_request_burst == 5
    assert multi_bucket_rate_limiter.max_token_burst == 50

    assert "rpm" in multi_bucket_rate_limiter._request_buckets
    assert "tpm" in multi_bucket_rate_limiter._token_buckets
    assert "rpd" in multi_bucket_rate_limiter._request_buckets
    assert "tpd" in multi_bucket_rate_limiter._token_buckets

    assert multi_bucket_rate_limiter._request_buckets["rpm"].tokens_per_period == 10
    assert multi_bucket_rate_limiter._token_buckets["tpm"].tokens_per_period == 100
    assert multi_bucket_rate_limiter._request_buckets["rpd"].tokens_per_period == 100
    assert multi_bucket_rate_limiter._token_buckets["tpd"].tokens_per_period == 1000

    assert multi_bucket_rate_limiter._request_buckets["rpm"].capacity == 5
    assert multi_bucket_rate_limiter._token_buckets["tpm"].capacity == 50
    assert multi_bucket_rate_limiter._request_buckets["rpd"].capacity == 100
    assert multi_bucket_rate_limiter._token_buckets["tpd"].capacity == 1000

    assert multi_bucket_rate_limiter._request_buckets["rpm"].time_period == 60
    assert multi_bucket_rate_limiter._token_buckets["tpm"].time_period == 60
    assert multi_bucket_rate_limiter._request_buckets["rpd"].time_period == 86400
    assert multi_bucket_rate_limiter._token_buckets["tpd"].time_period == 86400


@pytest.mark.asyncio(loop_scope="function")
async def test_over_acquire(multi_bucket_rate_limiter: RateLimiter):
    with pytest.raises(
        ValueError, match="Requested number of tokens 51 exceeds the capacity"
    ):
        # exceed the capacity of the tpm bucket
        await multi_bucket_rate_limiter.acquire(51)

    with pytest.raises(
        ValueError, match="Requested number of tokens 101 exceeds the capacity"
    ):
        # exceed the capacity of the tpd bucket
        await multi_bucket_rate_limiter.acquire(101)


@pytest.mark.asyncio(loop_scope="function")
async def test_acquire(simple_rate_limiter: RateLimiter):
    tasks = [
        asyncio.ensure_future(acquire_task(simple_rate_limiter)) for _ in range(365)
    ]

    # saturate the rate limiter
    pending = await wait_for_n_done(tasks, 360)
    assert len(pending) == 5

    # advance time to release 1 request token (with some buffer)
    await asyncio.sleep(60 / 360)
    pending = await wait_for_n_done(pending, 1)
    assert len(pending) == 4

    # advance time to release 2 request tokens
    await asyncio.sleep((60 / 360) * 2)
    pending = await wait_for_n_done(pending, 2)
    assert len(pending) == 2

    # release the remaining request tokens
    await asyncio.sleep((60 / 360) * 2)
    pending = await wait_for_n_done(pending, 2)
    assert len(pending) == 0


@pytest.mark.asyncio(loop_scope="function")
async def test_adjust():
    # Configure rate limiter with a higher rate for faster testing
    rate_limiter = RateLimiter(
        requests_per_minute=360,  # 1 request every ~0.167 seconds
        tokens_per_minute=60000,  # 1 token every ~0.001 seconds
        max_request_burst=60,
        max_token_burst=300,
    )

    # First, let's get some handles by acquiring tokens
    handles = []
    for _ in range(5):
        # Acquire with estimated 50 tokens each
        handle = await rate_limiter.acquire(50)
        handles.append(handle)

    # This should have consumed 5 requests and 250 tokens
    # Now try to acquire one more - should be blocked on tokens
    acquisition_task = asyncio.create_task(rate_limiter.acquire(60))

    # Wait a moment to ensure the task is blocked
    await asyncio.sleep(0.01)
    assert not acquisition_task.done()

    # Adjust one handle to return 30 tokens (estimated was 50, actual is 20)
    await rate_limiter.adjust(handles[0], 20)

    # This should have returned 30 tokens, allowing our blocked task to proceed
    await asyncio.sleep(0.01)
    assert acquisition_task.done()
    result_handle = await acquisition_task
    assert result_handle is not None

    # Now test a failed request scenario
    # First, consume all available request tokens
    remaining_requests = 55  # 60 burst - 5 already used
    pending_handles = []

    for _ in range(remaining_requests):
        handle = await rate_limiter.acquire(1)
        pending_handles.append(handle)

    # Try to make one more request, should be blocked
    blocked_request = asyncio.create_task(rate_limiter.acquire(1))
    await asyncio.sleep(0.01)
    assert not blocked_request.done()

    # Simulate a failed request by adjusting with 0 tokens
    await rate_limiter.adjust(pending_handles[0], 0)

    # This should have returned a request token, allowing our blocked task to proceed
    await asyncio.sleep(0.01)
    assert blocked_request.done()

    # Test header-based adjustment

    # Consume all available tokens again
    all_tokens = asyncio.create_task(rate_limiter.acquire(300))
    await asyncio.sleep(0.01)
    all_tokens_handle = await all_tokens

    # Try to acquire more tokens, should be blocked
    blocked_tokens = asyncio.create_task(rate_limiter.acquire(10))
    await asyncio.sleep(0.01)
    assert not blocked_tokens.done()

    # Adjust with headers that indicate more capacity is available
    mock_headers = {
        "x-ratelimit-remaining-requests": "100",  # More than we expect
        "x-ratelimit-limit-requests": "360",
        "x-ratelimit-reset-requests": "2",  # 2 seconds until reset
    }
    await rate_limiter.adjust(all_tokens_handle, 290, mock_headers)

    # This should allow our blocked task to proceed due to server-reported capacity
    await asyncio.sleep(0.17)
    assert blocked_tokens.done()
