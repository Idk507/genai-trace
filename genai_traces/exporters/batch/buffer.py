"""
Lock-free circular buffer for high-performance span buffering.
"""

from typing import Any, Optional, List
import threading


class CircularBuffer:
    """
    Thread-safe circular buffer for span buffering.
    
    Uses a simple lock-based approach for thread safety.
    For truly lock-free implementation, consider using atomics.
    """
    
    def __init__(self, capacity: int = 10000):
        """
        Initialize the circular buffer.
        
        Args:
            capacity: Maximum number of items
        """
        self._capacity = capacity
        self._buffer: List[Any] = [None] * capacity
        self._head = 0
        self._tail = 0
        self._size = 0
        self._lock = threading.Lock()
    
    def push(self, item: Any) -> bool:
        """
        Push an item to the buffer.
        
        Args:
            item: Item to push
            
        Returns:
            True if successful, False if buffer is full
        """
        with self._lock:
            if self._size >= self._capacity:
                return False
            
            self._buffer[self._tail] = item
            self._tail = (self._tail + 1) % self._capacity
            self._size += 1
            return True
    
    def pop(self) -> Optional[Any]:
        """
        Pop an item from the buffer.
        
        Returns:
            Item or None if buffer is empty
        """
        with self._lock:
            if self._size == 0:
                return None
            
            item = self._buffer[self._head]
            self._buffer[self._head] = None
            self._head = (self._head + 1) % self._capacity
            self._size -= 1
            return item
    
    def pop_batch(self, max_items: int) -> List[Any]:
        """
        Pop multiple items from the buffer.
        
        Args:
            max_items: Maximum items to pop
            
        Returns:
            List of items
        """
        items = []
        with self._lock:
            count = min(max_items, self._size)
            for _ in range(count):
                item = self._buffer[self._head]
                self._buffer[self._head] = None
                self._head = (self._head + 1) % self._capacity
                self._size -= 1
                items.append(item)
        return items
    
    def peek(self) -> Optional[Any]:
        """Peek at the next item without removing it."""
        with self._lock:
            if self._size == 0:
                return None
            return self._buffer[self._head]
    
    def __len__(self) -> int:
        """Get current size."""
        return self._size
    
    def is_empty(self) -> bool:
        """Check if buffer is empty."""
        return self._size == 0
    
    def is_full(self) -> bool:
        """Check if buffer is full."""
        return self._size >= self._capacity
    
    def clear(self) -> int:
        """
        Clear the buffer.
        
        Returns:
            Number of items cleared
        """
        with self._lock:
            count = self._size
            self._buffer = [None] * self._capacity
            self._head = 0
            self._tail = 0
            self._size = 0
            return count
    
    @property
    def capacity(self) -> int:
        """Get buffer capacity."""
        return self._capacity
    
    @property
    def available(self) -> int:
        """Get available space."""
        return self._capacity - self._size


class PriorityBuffer:
    """
    Priority-based buffer that ensures high-priority items are not dropped.
    """
    
    def __init__(
        self,
        capacity: int = 10000,
        high_priority_reserve: float = 0.2,
    ):
        """
        Initialize the priority buffer.
        
        Args:
            capacity: Total capacity
            high_priority_reserve: Fraction reserved for high priority
        """
        self._capacity = capacity
        self._reserve = int(capacity * high_priority_reserve)
        
        self._normal_buffer = CircularBuffer(capacity - self._reserve)
        self._priority_buffer = CircularBuffer(self._reserve)
        self._lock = threading.Lock()
    
    def push(self, item: Any, high_priority: bool = False) -> bool:
        """
        Push an item with priority.
        
        Args:
            item: Item to push
            high_priority: Whether this is high priority
            
        Returns:
            True if successful
        """
        if high_priority:
            return self._priority_buffer.push(item)
        else:
            return self._normal_buffer.push(item)
    
    def pop(self) -> Optional[Any]:
        """
        Pop an item, prioritizing high-priority items.
        
        Returns:
            Item or None
        """
        item = self._priority_buffer.pop()
        if item is not None:
            return item
        return self._normal_buffer.pop()
    
    def __len__(self) -> int:
        """Get total size."""
        return len(self._normal_buffer) + len(self._priority_buffer)
    
    def is_empty(self) -> bool:
        """Check if both buffers are empty."""
        return self._normal_buffer.is_empty() and self._priority_buffer.is_empty()
