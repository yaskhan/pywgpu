"""Utility classes for indirect validation."""

from typing import List, Iterator, Set


class UniqueIndexScratch:
    """
    Scratch space for tracking unique indices.
    
    Uses a set to efficiently track which indices have been seen.
    """
    
    def __init__(self):
        """Create new unique index scratch space."""
        self._seen: Set[int] = set()
    
    def clear(self) -> None:
        """Clear the scratch space."""
        self._seen.clear()
    
    def insert(self, index: int) -> bool:
        """
        Try to insert an index.
        
        Args:
            index: The index to insert.
            
        Returns:
            True if index was not seen before, False if already seen.
        """
        if index in self._seen:
            return False
        self._seen.add(index)
        return True


class UniqueIndexIterator:
    """
    Iterator that yields only unique indices.
    
    Filters out duplicate indices using scratch space.
    """
    
    def __init__(self, inner: Iterator[int], scratch: UniqueIndexScratch):
        """
        Create unique index iterator.
        
        Args:
            inner: The inner iterator.
            scratch: Scratch space for tracking seen indices.
        """
        self.inner = inner
        self.scratch = scratch
        scratch.clear()
    
    def __iter__(self) -> 'UniqueIndexIterator':
        return self
    
    def __next__(self) -> int:
        """Get next unique index."""
        for index in self.inner:
            if self.scratch.insert(index):
                return index
        raise StopIteration


def unique_indices(iterator: Iterator[int], scratch: UniqueIndexScratch) -> UniqueIndexIterator:
    """
    Create an iterator that yields only unique indices.
    
    Args:
        iterator: The source iterator.
        scratch: Scratch space for tracking.
        
    Returns:
        Iterator yielding unique indices.
    """
    return UniqueIndexIterator(iterator, scratch)


class BufferBarrierScratch:
    """
    Scratch space for buffer barriers.
    
    Accumulates buffer barriers for batch submission.
    """
    
    def __init__(self):
        """Create new buffer barrier scratch space."""
        self._barriers: List[dict] = []
    
    def clear(self) -> None:
        """Clear all barriers."""
        self._barriers.clear()
    
    def extend(self, barriers: Iterator[dict]) -> None:
        """
        Add barriers from iterator.
        
        Args:
            barriers: Iterator of buffer barriers.
        """
        self._barriers.extend(barriers)
    
    def get_barriers(self) -> List[dict]:
        """Get all accumulated barriers."""
        return self._barriers


class BufferBarriers:
    """
    Builder for buffer barriers.
    
    Accumulates barriers and encodes them in batch.
    """
    
    def __init__(self, scratch: BufferBarrierScratch):
        """
        Create buffer barriers builder.
        
        Args:
            scratch: Scratch space to use.
        """
        self.scratch = scratch
    
    def extend(self, barriers: Iterator[dict]) -> 'BufferBarriers':
        """
        Add barriers from iterator.
        
        Args:
            barriers: Iterator of buffer barriers.
            
        Returns:
            Self for chaining.
        """
        self.scratch.extend(barriers)
        return self
    
    def encode(self, encoder: any) -> None:
        """
        Encode all barriers to command encoder.
        
        Args:
            encoder: The command encoder.
        """
        barriers = self.scratch.get_barriers()
        if barriers and hasattr(encoder, 'transition_buffers'):
            try:
                encoder.transition_buffers(barriers)
            except Exception:
                pass
        
        # Clear scratch for reuse
        self.scratch.clear()
