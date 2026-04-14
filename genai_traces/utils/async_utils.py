"""
Async utilities for GenAI-Traces.
"""

import asyncio
from typing import Any, Callable, TypeVar, Coroutine
from functools import wraps
import concurrent.futures

T = TypeVar("T")


def ensure_async(func: Callable[..., T]) -> Callable[..., Coroutine[Any, Any, T]]:
    """
    Ensure a function is async. If sync, wrap it to run in executor.
    
    Args:
        func: Function to wrap
        
    Returns:
        Async version of the function
    """
    if asyncio.iscoroutinefunction(func):
        return func
    
    @wraps(func)
    async def wrapper(*args, **kwargs) -> T:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: func(*args, **kwargs))
    
    return wrapper


async def run_sync_in_executor(
    func: Callable[..., T],
    *args,
    executor: concurrent.futures.Executor = None,
    **kwargs
) -> T:
    """
    Run a synchronous function in an executor.
    
    Args:
        func: Synchronous function to run
        *args: Positional arguments
        executor: Optional executor (defaults to thread pool)
        **kwargs: Keyword arguments
        
    Returns:
        Function result
    """
    loop = asyncio.get_event_loop()
    
    if kwargs:
        wrapped = lambda: func(*args, **kwargs)
        return await loop.run_in_executor(executor, wrapped)
    else:
        return await loop.run_in_executor(executor, func, *args)


def run_async(coro: Coroutine[Any, Any, T]) -> T:
    """
    Run an async coroutine from sync code.
    
    Args:
        coro: Coroutine to run
        
    Returns:
        Coroutine result
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None
    
    if loop is not None:
        import nest_asyncio
        nest_asyncio.apply()
        return loop.run_until_complete(coro)
    else:
        return asyncio.run(coro)


async def gather_with_concurrency(
    n: int,
    *coros: Coroutine,
) -> list:
    """
    Run coroutines with limited concurrency.
    
    Args:
        n: Maximum concurrent tasks
        *coros: Coroutines to run
        
    Returns:
        List of results
    """
    semaphore = asyncio.Semaphore(n)
    
    async def sem_coro(coro):
        async with semaphore:
            return await coro
    
    return await asyncio.gather(*(sem_coro(c) for c in coros))


class AsyncBatcher:
    """
    Batches async operations for efficiency.
    """
    
    def __init__(
        self,
        batch_size: int = 10,
        max_wait_ms: float = 100,
    ):
        self.batch_size = batch_size
        self.max_wait_ms = max_wait_ms
        self._queue: asyncio.Queue = None
        self._results: dict = {}
        self._lock = asyncio.Lock()
        self._counter = 0
    
    async def _ensure_queue(self):
        if self._queue is None:
            self._queue = asyncio.Queue()
    
    async def add(self, item: Any) -> int:
        """Add item to batch, returns ticket ID."""
        await self._ensure_queue()
        
        async with self._lock:
            ticket = self._counter
            self._counter += 1
        
        await self._queue.put((ticket, item))
        return ticket
    
    async def get_result(self, ticket: int, timeout: float = 5.0) -> Any:
        """Get result for a ticket."""
        start = asyncio.get_event_loop().time()
        
        while True:
            if ticket in self._results:
                return self._results.pop(ticket)
            
            if asyncio.get_event_loop().time() - start > timeout:
                raise TimeoutError(f"Timeout waiting for result {ticket}")
            
            await asyncio.sleep(0.01)
    
    async def process_batch(
        self,
        processor: Callable[[list], Coroutine[Any, Any, list]],
    ) -> int:
        """
        Process a batch of items.
        
        Args:
            processor: Async function that takes list of items and returns list of results
            
        Returns:
            Number of items processed
        """
        await self._ensure_queue()
        
        items = []
        tickets = []
        
        try:
            while len(items) < self.batch_size:
                try:
                    ticket, item = await asyncio.wait_for(
                        self._queue.get(),
                        timeout=self.max_wait_ms / 1000
                    )
                    items.append(item)
                    tickets.append(ticket)
                except asyncio.TimeoutError:
                    break
        except Exception:
            pass
        
        if not items:
            return 0
        
        results = await processor(items)
        
        for ticket, result in zip(tickets, results):
            self._results[ticket] = result
        
        return len(items)
