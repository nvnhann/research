from collections import OrderedDict


class LRU_Cache(object):

    def __init__(self, capacity: int):
        """
        Initialize the class variables
        """
        self.capacity = capacity
        self.cache = OrderedDict()

    def get(self, key: int) -> int:
        """
        Retrieve item from provided key. Return -1 if nonexistent.
        """
        if key not in self.cache:
            return -1
        else:
            self.cache.move_to_end(key)  # Move the current key to the end (most recent)
            return self.cache[key]

    def set(self, key: int, value: int):
        """
        Set the value if the key is not present in the cache. If the cache is at capacity, remove the oldest item and then insert the new item.
        """
        if key in self.cache:
            self.cache.move_to_end(key)  # Move the current key to the end (most recent)
        self.cache[key] = value
        if len(self.cache) > self.capacity:
            self.cache.popitem(last=False)  # Remove the least recently used item (first item)


# Example usage with test cases
our_cache = LRU_Cache(5)

our_cache.set(1, 1)
our_cache.set(2, 2)
our_cache.set(3, 3)
our_cache.set(4, 4)

print(our_cache.get(1))  # Returns 1
print(our_cache.get(2))  # Returns 2
print(our_cache.get(9))  # Returns -1 because 9 is not present in the cache

our_cache.set(5, 5)
our_cache.set(6, 6)

print(our_cache.get(3))  # Returns -1 because the cache reached its capacity and 3 was the least recently used entry

# Additional test cases

# Test Case 1: Very large values
our_cache_large = LRU_Cache(3)
our_cache_large.set(1000000, 1)
our_cache_large.set(2000000, 2)
our_cache_large.set(3000000, 3)
print(our_cache_large.get(1000000))  # Returns 1

# Test Case 2: Null or empty values
our_cache_null = LRU_Cache(3)
our_cache_null.set(1, None)
print(our_cache_null.get(1))  # Returns None
our_cache_null.set(2, 0)
print(our_cache_null.get(2))  # Returns 0

# Test Case 3: Cache with limited size
our_cache_small = LRU_Cache(2)
our_cache_small.set(1, 1)
our_cache_small.set(2, 2)
print(our_cache_small.get(1))  # Returns 1
our_cache_small.set(3, 3)
print(our_cache_small.get(2))  # Returns -1 (2 was removed due to capacity limit)
print(our_cache_small.get(3))  # Returns 3
