"""Advanced caching system for AI Solutions Lab performance optimization."""

import time
import hashlib
import json
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
from collections import OrderedDict
import threading
import pickle
from pathlib import Path

@dataclass
class CacheEntry:
    """Represents a cached item with metadata."""
    data: Any
    timestamp: float
    access_count: int
    size_bytes: int
    ttl: Optional[float] = None  # Time to live in seconds

class CacheManager:
    """Advanced caching system with multiple strategies."""
    
    def __init__(self, max_size_mb: int = 100, default_ttl: int = 3600):
        self.max_size_bytes = max_size_mb * 1024 * 1024
        self.default_ttl = default_ttl
        self.current_size_bytes = 0
        
        # Cache storage with LRU ordering
        self.cache: OrderedDict[str, CacheEntry] = OrderedDict()
        
        # Cache statistics
        self.stats = {
            'hits': 0,
            'misses': 0,
            'evictions': 0,
            'total_requests': 0
        }
        
        # Thread safety
        self.lock = threading.RLock()
        
        # Cache persistence
        self.cache_dir = Path("./data/cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Load existing cache
        self._load_cache()
    
    def _generate_key(self, *args, **kwargs) -> str:
        """Generate a cache key from function arguments."""
        # Create a deterministic string representation
        key_data = {
            'args': args,
            'kwargs': sorted(kwargs.items())
        }
        key_string = json.dumps(key_data, sort_keys=True, default=str)
        
        # Generate hash for consistent key length
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def _estimate_size(self, data: Any) -> int:
        """Estimate the size of data in bytes."""
        try:
            # Try to serialize to get size estimate
            serialized = pickle.dumps(data)
            return len(serialized)
        except:
            # Fallback to string length estimation
            return len(str(data)) * 2
    
    def _evict_if_needed(self, required_size: int):
        """Evict cache entries if needed to make space."""
        while self.current_size_bytes + required_size > self.max_size_bytes and self.cache:
            # Remove least recently used entry
            key, entry = self.cache.popitem(last=False)
            self.current_size_bytes -= entry.size_bytes
            self.stats['evictions'] += 1
    
    def get(self, key: str) -> Optional[Any]:
        """Retrieve an item from cache."""
        with self.lock:
            self.stats['total_requests'] += 1
            
            if key not in self.cache:
                self.stats['misses'] += 1
                return None
            
            entry = self.cache[key]
            
            # Check TTL
            if entry.ttl and time.time() - entry.timestamp > entry.ttl:
                del self.cache[key]
                self.current_size_bytes -= entry.size_bytes
                self.stats['misses'] += 1
                return None
            
            # Update access count and move to end (LRU)
            entry.access_count += 1
            self.cache.move_to_end(key)
            
            self.stats['hits'] += 1
            return entry.data
    
    def set(self, key: str, data: Any, ttl: Optional[int] = None) -> bool:
        """Store an item in cache."""
        with self.lock:
            size_bytes = self._estimate_size(data)
            
            # Check if we need to evict entries
            self._evict_if_needed(size_bytes)
            
            # Create cache entry
            entry = CacheEntry(
                data=data,
                timestamp=time.time(),
                access_count=1,
                size_bytes=size_bytes,
                ttl=ttl or self.default_ttl
            )
            
            # Remove existing entry if it exists
            if key in self.cache:
                old_entry = self.cache[key]
                self.current_size_bytes -= old_entry.size_bytes
            
            # Add new entry
            self.cache[key] = entry
            self.current_size_bytes += size_bytes
            
            # Move to end (LRU)
            self.cache.move_to_end(key)
            
            return True
    
    def delete(self, key: str) -> bool:
        """Remove an item from cache."""
        with self.lock:
            if key in self.cache:
                entry = self.cache[key]
                del self.cache[key]
                self.current_size_bytes -= entry.size_bytes
                return True
            return False
    
    def clear(self):
        """Clear all cache entries."""
        with self.lock:
            self.cache.clear()
            self.current_size_bytes = 0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self.lock:
            hit_rate = (self.stats['hits'] / max(self.stats['total_requests'], 1)) * 100
            
            return {
                'hits': self.stats['hits'],
                'misses': self.stats['misses'],
                'total_requests': self.stats['total_requests'],
                'hit_rate_percent': round(hit_rate, 2),
                'evictions': self.stats['evictions'],
                'current_size_bytes': self.current_size_bytes,
                'max_size_bytes': self.max_size_bytes,
                'current_size_mb': round(self.current_size_bytes / (1024 * 1024), 2),
                'max_size_mb': round(self.max_size_bytes / (1024 * 1024), 2),
                'entries_count': len(self.cache)
            }
    
    def _save_cache(self):
        """Save cache to disk."""
        try:
            cache_file = self.cache_dir / "cache.pkl"
            
            # Prepare data for serialization
            cache_data = {}
            for key, entry in self.cache.items():
                # Only save entries that haven't expired
                if not entry.ttl or time.time() - entry.timestamp <= entry.ttl:
                    cache_data[key] = entry
            
            with open(cache_file, 'wb') as f:
                pickle.dump(cache_data, f)
                
        except Exception as e:
            print(f"Warning: Could not save cache: {e}")
    
    def _load_cache(self):
        """Load cache from disk."""
        try:
            cache_file = self.cache_dir / "cache.pkl"
            if cache_file.exists():
                with open(cache_file, 'rb') as f:
                    cache_data = pickle.load(f)
                
                # Restore cache entries
                for key, entry in cache_data.items():
                    # Check if entry is still valid
                    if not entry.ttl or time.time() - entry.timestamp <= entry.ttl:
                        self.cache[key] = entry
                        self.current_size_bytes += entry.size_bytes
                
                print(f"Loaded {len(self.cache)} cache entries")
                
        except Exception as e:
            print(f"Warning: Could not load cache: {e}")
    
    def save_cache(self):
        """Manually save cache to disk."""
        with self.lock:
            self._save_cache()
    
    def cleanup_expired(self):
        """Remove expired cache entries."""
        with self.lock:
            expired_keys = []
            current_time = time.time()
            
            for key, entry in self.cache.items():
                if entry.ttl and current_time - entry.timestamp > entry.ttl:
                    expired_keys.append(key)
            
            for key in expired_keys:
                entry = self.cache[key]
                del self.cache[key]
                self.current_size_bytes -= entry.size_bytes
            
            if expired_keys:
                print(f"Cleaned up {len(expired_keys)} expired cache entries")
    
    def get_top_accessed(self, limit: int = 10) -> List[Tuple[str, int]]:
        """Get top accessed cache entries."""
        with self.lock:
            sorted_entries = sorted(
                self.cache.items(),
                key=lambda x: x[1].access_count,
                reverse=True
            )
            return [(key, entry.access_count) for key, entry in sorted_entries[:limit]]


class SearchCache:
    """Specialized cache for search results."""
    
    def __init__(self, cache_manager: CacheManager):
        self.cache_manager = cache_manager
        self.search_prefix = "search:"
        self.rag_prefix = "rag:"
    
    def _get_search_key(self, query: str, search_type: str, filters: Dict[str, Any], max_results: int) -> str:
        """Generate cache key for search queries."""
        key_data = {
            'query': query.lower().strip(),
            'search_type': search_type,
            'filters': sorted(filters.items()) if filters else [],
            'max_results': max_results
        }
        return self.search_prefix + self.cache_manager._generate_key(key_data)
    
    def _get_rag_key(self, query: str, k: int, generate_answer: bool, conversation_history: Optional[List[Dict[str, str]]] = None) -> str:
        """Generate cache key for RAG queries."""
        key_data = {
            'query': query.lower().strip(),
            'k': k,
            'generate_answer': generate_answer,
            'conversation_history': conversation_history or []
        }
        return self.rag_prefix + self.cache_manager._generate_key(key_data)
    
    def get_search_results(self, query: str, search_type: str, filters: Dict[str, Any], max_results: int) -> Optional[Any]:
        """Get cached search results."""
        key = self._get_search_key(query, search_type, filters, max_results)
        return self.cache_manager.get(key)
    
    def cache_search_results(self, query: str, search_type: str, filters: Dict[str, Any], max_results: int, results: Any, ttl: int = 1800) -> bool:
        """Cache search results."""
        key = self._get_search_key(query, search_type, filters, max_results)
        return self.cache_manager.set(key, results, ttl)
    
    def get_rag_results(self, query: str, k: int, generate_answer: bool, conversation_history: Optional[List[Dict[str, str]]] = None) -> Optional[Any]:
        """Get cached RAG results."""
        key = self._get_rag_key(query, k, generate_answer, conversation_history)
        return self.cache_manager.get(key)
    
    def cache_rag_results(self, query: str, k: int, generate_answer: bool, conversation_history: Optional[List[Dict[str, str]]], results: Any, ttl: int = 3600) -> bool:
        """Cache RAG results."""
        key = self._get_rag_key(query, k, generate_answer, conversation_history)
        return self.cache_manager.set(key, results, ttl)
    
    def invalidate_search_cache(self, query_pattern: Optional[str] = None):
        """Invalidate search cache entries."""
        with self.cache_manager.lock:
            keys_to_remove = []
            
            for key in self.cache_manager.cache.keys():
                if key.startswith(self.search_prefix):
                    if query_pattern is None:
                        keys_to_remove.append(key)
                    else:
                        # For exact query invalidation, we need to check if the query matches
                        # Since keys are hashed, we'll invalidate all search cache entries
                        # In a production system, you might want to maintain a reverse lookup
                        keys_to_remove.append(key)
            
            for key in keys_to_remove:
                self.cache_manager.delete(key)
    
    def invalidate_rag_cache(self, query_pattern: Optional[str] = None):
        """Invalidate RAG cache entries."""
        with self.cache_manager.lock:
            keys_to_remove = []
            
            for key in self.cache_manager.cache.keys():
                if key.startswith(self.rag_prefix):
                    if query_pattern is None:
                        keys_to_remove.append(key)
                    else:
                        # For exact query invalidation, we need to check if the query matches
                        # Since keys are hashed, we'll invalidate all RAG cache entries
                        # In a production system, you might want to maintain a reverse lookup
                        keys_to_remove.append(key)
            
            for key in keys_to_remove:
                self.cache_manager.delete(key)


def main():
    """Demo the cache manager."""
    print("Cache Manager Demo")
    print("=" * 40)
    
    # Create cache manager
    cache_manager = CacheManager(max_size_mb=10, default_ttl=60)
    search_cache = SearchCache(cache_manager)
    
    # Test basic caching
    print("Testing basic caching...")
    cache_manager.set("test_key", "test_value", ttl=30)
    result = cache_manager.get("test_key")
    print(f"Cached value: {result}")
    
    # Test search caching
    print("\nTesting search caching...")
    search_filters = {"file_type": ".md", "min_confidence": 0.5}
    search_results = {"results": ["doc1", "doc2"], "count": 2}
    
    search_cache.cache_search_results(
        "machine learning",
        "hybrid",
        search_filters,
        10,
        search_results
    )
    
    cached_results = search_cache.get_search_results(
        "machine learning",
        "hybrid",
        search_filters,
        10
    )
    print(f"Cached search results: {cached_results}")
    
    # Show statistics
    stats = cache_manager.get_stats()
    print(f"\nCache statistics: {stats}")
    
    # Save cache
    cache_manager.save_cache()
    print("Cache saved to disk")


if __name__ == "__main__":
    main()
