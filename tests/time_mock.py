import asyncio
import heapq
from dataclasses import dataclass
from typing import Any, Callable, Optional


@dataclass
class MockTimer:
    fire_at: float
    callback: Callable
    args: tuple
    handle: Optional["MockTimerHandle"] = (
        None  # Reference back to handle for cancellation check
    )

    def __lt__(self, other: "MockTimer") -> bool:
        """Make timers sortable by fire_at time."""
        if not isinstance(other, MockTimer):
            return NotImplemented
        return self.fire_at < other.fire_at


@dataclass
class MockTimerHandle:
    timer_ref: MockTimer
    cancelled: bool = False

    def cancel(self) -> None:
        """Cancel the timer."""
        self.cancelled = True


class MockLoop:
    """A mock asyncio event loop."""

    def __init__(self, time_provider: "MockTime") -> None:
        self._time_provider = time_provider
        self._tasks = []  # keeps track of tasks created by ``_wake_next_timer``

    def call_later(
        self, delay: float, callback: Callable, *args: tuple[Any, ...]
    ) -> MockTimerHandle:
        """Schedule a callback to be called after a delay."""
        return self._time_provider.schedule_timer(delay, callback, *args)

    def create_future(self) -> asyncio.Future:
        return asyncio.Future()

    def call_soon(self, callback: Callable, *args: tuple[Any, ...]) -> None:
        """Schedule a callback to be called immediately."""
        try:
            callback(*args)
        except Exception as e:
            raise RuntimeError(
                f"Exception during mocked call_soon execution: {e}", exc_info=True
            ) from e

    def time(self) -> float:
        """Return the current mock time."""
        return self._time_provider.monotonic()

    def create_task(self, coro):
        """Schedule a coroutine for execution."""

        async def task_wrapper():
            try:
                await coro
            except Exception as e:
                raise e

        task = asyncio.create_task(task_wrapper())
        self._tasks.append(task)  # keep track if needed for cleanup/waiting
        return task


class MockTime:
    """Encapsulates mocked time state and provides mock functions."""

    def __init__(self):
        self.mock_loop = MockLoop(self)

        self._current_time: float = 0.0

        # min-heap for timers for efficient retrieval of the next one
        self._timers: list[MockTimer] = []

    def monotonic(self) -> float:
        """Return the current mock monotonic time."""
        return self._current_time

    def get_running_loop(self) -> MockLoop:
        """Return the mock loop."""
        return self.mock_loop

    def schedule_timer(
        self, delay: float, callback: Callable, *args
    ) -> MockTimerHandle:
        fire_at = self._current_time + delay
        timer = MockTimer(fire_at=fire_at, callback=callback, args=args)
        handle = MockTimerHandle(timer_ref=timer)
        timer.handle = handle  # link timer back to its handle
        heapq.heappush(self._timers, timer)

        return handle

    async def advance_time(self, delta: float) -> None:
        """Advance mock time by delta, firing due timers."""
        if delta < 0:
            raise ValueError("Cannot advance time backwards")

        target_time = self._current_time + delta
        processed_callbacks = False

        # process all timers scheduled to fire at or before the target time
        while self._timers and self._timers[0].fire_at <= target_time:
            next_timer = heapq.heappop(self._timers)
            self._current_time = next_timer.fire_at

            if next_timer.handle and next_timer.handle.cancelled:
                # skip cancelled timers
                continue

            try:
                # execute the timer callback (e.g., _wake_next_timer)
                next_timer.callback(*next_timer.args)
                processed_callbacks = True

                # allow any scheduled tasks to run before advancing
                await asyncio.sleep(0)
            except Exception:
                pass

        # set time to the final target time
        self._current_time = target_time

        if processed_callbacks:  # allow any last tasks spawned by timers to run
            await asyncio.sleep(0)
