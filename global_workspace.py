# File: global_workspace.py

import threading
import time
import logging
import json
from collections import defaultdict
from typing import Any, Dict, Optional, Callable, List, Set, Tuple

logging.basicConfig(level=logging.DEBUG)

class GlobalWorkspace:
    """
    A thread-safe global workspace for inter-component communication.
    Implements a shared memory space with read/write access, event notifications,
    data persistence, and more.

    Inspired by the Global Workspace Theory (GWT) of consciousness.
    """

    def __init__(self):
        self._shared_memory: Dict[str, Any] = {}
        self._lock = threading.Lock()
        self._subscribers: Dict[str, List[Callable[[Any], None]]] = defaultdict(list)
        self._event_conditions: Dict[str, threading.Condition] = defaultdict(threading.Condition)
        self._data_timestamps: Dict[str, float] = {}
        self._data_versions: Dict[str, int] = {}
        self._data_expirations: Dict[str, float] = {}
        self._max_data_age: float = float('inf')  # Data does not expire by default
        self._cleanup_interval: float = 60.0  # Default cleanup interval in seconds
        self._cleanup_thread = threading.Thread(target=self._cleanup_loop, daemon=True)
        self._cleanup_thread.start()
        self._logging_enabled: bool = True
        self._persistence_file: Optional[str] = None
        self._persistence_interval: float = 300.0  # Save every 5 minutes by default
        self._persistence_thread = threading.Thread(target=self._persistence_loop, daemon=True)
        self._persistence_thread.start()
        self._observers: Set[Callable[[str, Any], None]] = set()
        self._stop_event = threading.Event()

    def enable_logging(self, enable: bool):
        """
        Enable or disable logging.

        :param enable: True to enable logging, False to disable.
        """
        self._logging_enabled = enable

    def set_persistence_file(self, filename: str):
        """
        Set the file to use for persisting shared memory data.

        :param filename: The filename to use for persistence.
        """
        self._persistence_file = filename

    def set_persistence_interval(self, interval: float):
        """
        Set the interval for persisting data to disk.

        :param interval: Interval in seconds.
        """
        self._persistence_interval = interval

    def set_max_data_age(self, max_age: float):
        """
        Set the maximum age for data before it expires.

        :param max_age: Maximum age in seconds.
        """
        self._max_data_age = max_age

    def write(self, key: str, value: Any, expiration: Optional[float] = None):
        """
        Write a value to the shared memory.

        :param key: The key under which to store the value.
        :param value: The value to store.
        :param expiration: Optional expiration time in seconds from now.
        """
        with self._lock:
            self._shared_memory[key] = value
            timestamp = time.time()
            self._data_timestamps[key] = timestamp
            self._data_versions[key] = self._data_versions.get(key, 0) + 1
            if expiration is not None:
                self._data_expirations[key] = timestamp + expiration
            elif self._max_data_age != float('inf'):
                self._data_expirations[key] = timestamp + self._max_data_age
            else:
                self._data_expirations.pop(key, None)

            if self._logging_enabled:
                logging.debug(f"Write: key={key}, value={value}, timestamp={timestamp}")

            # Notify subscribers
            for callback in self._subscribers.get(key, []):
                threading.Thread(target=callback, args=(value,), daemon=True).start()

            # Notify condition variables
            condition = self._event_conditions.get(key)
            if condition:
                with condition:
                    condition.notify_all()

            # Notify observers
            for observer in self._observers:
                threading.Thread(target=observer, args=(key, value), daemon=True).start()

    def read(self, key: str, default: Any = None) -> Any:
        """
        Read a value from the shared memory.

        :param key: The key to read.
        :param default: Default value if key does not exist.
        :return: The value associated with the key, or default if not found.
        """
        with self._lock:
            value = self._shared_memory.get(key, default)
            if self._logging_enabled:
                logging.debug(f"Read: key={key}, value={value}")
            return value

    def wait_for(self, key: str, timeout: Optional[float] = None) -> Any:
        """
        Wait for a key to be written to the shared memory.

        :param key: The key to wait for.
        :param timeout: Optional timeout in seconds.
        :return: The value associated with the key.
        """
        condition = self._event_conditions.setdefault(key, threading.Condition())
        with condition:
            if key in self._shared_memory:
                return self._shared_memory[key]
            notified = condition.wait(timeout=timeout)
            if not notified:
                raise TimeoutError(f"Timeout waiting for key '{key}'")
            return self._shared_memory[key]

    def subscribe(self, key: str, callback: Callable[[Any], None]):
        """
        Subscribe to changes to a key in the shared memory.

        :param key: The key to subscribe to.
        :param callback: The callback to call when the key is updated.
        """
        with self._lock:
            self._subscribers[key].append(callback)
            if self._logging_enabled:
                logging.debug(f"Subscribed to key: {key}")

    def unsubscribe(self, key: str, callback: Callable[[Any], None]):
        """
        Unsubscribe from changes to a key in the shared memory.

        :param key: The key to unsubscribe from.
        :param callback: The callback to remove.
        """
        with self._lock:
            if callback in self._subscribers.get(key, []):
                self._subscribers[key].remove(callback)
                if self._logging_enabled:
                    logging.debug(f"Unsubscribed from key: {key}")

    def add_observer(self, observer: Callable[[str, Any], None]):
        """
        Add an observer that gets notified on any write operation.

        :param observer: The observer function taking key and value.
        """
        with self._lock:
            self._observers.add(observer)
            if self._logging_enabled:
                logging.debug("Observer added.")

    def remove_observer(self, observer: Callable[[str, Any], None]):
        """
        Remove an observer.

        :param observer: The observer to remove.
        """
        with self._lock:
            self._observers.discard(observer)
            if self._logging_enabled:
                logging.debug("Observer removed.")

    def get_data_age(self, key: str) -> Optional[float]:
        """
        Get the age of the data associated with a key.

        :param key: The key to check.
        :return: Age in seconds, or None if key does not exist.
        """
        with self._lock:
            timestamp = self._data_timestamps.get(key)
            if timestamp is not None:
                age = time.time() - timestamp
                if self._logging_enabled:
                    logging.debug(f"Data age for key '{key}': {age} seconds")
                return age
            else:
                return None

    def get_version(self, key: str) -> Optional[int]:
        """
        Get the version number of the data associated with a key.

        :param key: The key to check.
        :return: Version number, or None if key does not exist.
        """
        with self._lock:
            version = self._data_versions.get(key)
            if self._logging_enabled:
                logging.debug(f"Data version for key '{key}': {version}")
            return version

    def keys(self) -> List[str]:
        """
        Get a list of all keys in the shared memory.

        :return: List of keys.
        """
        with self._lock:
            keys = list(self._shared_memory.keys())
            if self._logging_enabled:
                logging.debug(f"Keys in shared memory: {keys}")
            return keys

    def clear(self):
        """
        Clear all data from the shared memory.
        """
        with self._lock:
            self._shared_memory.clear()
            self._data_timestamps.clear()
            self._data_versions.clear()
            self._data_expirations.clear()
            if self._logging_enabled:
                logging.debug("Shared memory cleared.")

    def _cleanup_loop(self):
        """
        Background thread that periodically cleans up expired data.
        """
        while not self._stop_event.is_set():
            time.sleep(self._cleanup_interval)
            self._cleanup_expired_data()

    def _cleanup_expired_data(self):
        """
        Remove expired data from the shared memory.
        """
        with self._lock:
            current_time = time.time()
            keys_to_delete = [key for key, expiration in self._data_expirations.items()
                              if expiration <= current_time]
            for key in keys_to_delete:
                self._shared_memory.pop(key, None)
                self._data_timestamps.pop(key, None)
                self._data_versions.pop(key, None)
                self._data_expirations.pop(key, None)
                if self._logging_enabled:
                    logging.debug(f"Data expired and removed: key={key}")

    def _persistence_loop(self):
        """
        Background thread that periodically saves data to disk.
        """
        while not self._stop_event.is_set():
            time.sleep(self._persistence_interval)
            self._save_to_disk()

    def _save_to_disk(self):
        """
        Save the shared memory to disk.
        """
        if self._persistence_file is None:
            return
        with self._lock:
            data = {
                'shared_memory': self._shared_memory,
                'data_timestamps': self._data_timestamps,
                'data_versions': self._data_versions,
                'data_expirations': self._data_expirations,
            }
            try:
                with open(self._persistence_file, 'w') as f:
                    json.dump(data, f)
                if self._logging_enabled:
                    logging.debug(f"Shared memory saved to {self._persistence_file}")
            except Exception as e:
                logging.error(f"Error saving shared memory to disk: {e}")

    def load_from_disk(self):
        """
        Load the shared memory from disk.
        """
        if self._persistence_file is None:
            return
        with self._lock:
            try:
                with open(self._persistence_file, 'r') as f:
                    data = json.load(f)
                    self._shared_memory = data.get('shared_memory', {})
                    self._data_timestamps = data.get('data_timestamps', {})
                    self._data_versions = data.get('data_versions', {})
                    self._data_expirations = data.get('data_expirations', {})
                if self._logging_enabled:
                    logging.debug(f"Shared memory loaded from {self._persistence_file}")
            except Exception as e:
                logging.error(f"Error loading shared memory from disk: {e}")

    def shutdown(self):
        """
        Shutdown the global workspace, stopping background threads.
        """
        self._stop_event.set()
        self._cleanup_thread.join()
        self._persistence_thread.join()
        if self._logging_enabled:
            logging.debug("Global workspace shutdown.")

    def __enter__(self):
        """
        Enter the runtime context related to this object.
        """
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """
        Exit the runtime context and perform cleanup.
        """
        self.shutdown()

    def __del__(self):
        """
        Destructor to ensure proper shutdown.
        """
        self.shutdown()

    def get_state(self) -> Dict[str, Any]:
        """
        Get a copy of the entire shared memory state.

        :return: A dictionary representing the shared memory state.
        """
        with self._lock:
            state = self._shared_memory.copy()
            if self._logging_enabled:
                logging.debug(f"Shared memory state retrieved.")
            return state

    def wait_for_condition(self, condition_func: Callable[[], bool], timeout: Optional[float] = None) -> bool:
        """
        Wait until a condition function returns True.

        :param condition_func: A function that returns a boolean.
        :param timeout: Optional timeout in seconds.
        :return: True if the condition was met, False if timed out.
        """
        end_time = None if timeout is None else time.time() + timeout
        while True:
            with self._lock:
                if condition_func():
                    return True
            if timeout is not None and time.time() >= end_time:
                return False
            time.sleep(0.01)  # Sleep briefly to avoid busy waiting

    def set_cleanup_interval(self, interval: float):
        """
        Set the cleanup interval for expired data.

        :param interval: Interval in seconds.
        """
        self._cleanup_interval = interval
        if self._logging_enabled:
            logging.debug(f"Cleanup interval set to {interval} seconds.")

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the shared memory.

        :return: A dictionary of statistics.
        """
        with self._lock:
            stats = {
                'total_keys': len(self._shared_memory),
                'total_subscribers': sum(len(s) for s in self._subscribers.values()),
                'total_observers': len(self._observers),
                'total_data_size': sum(self._get_size(value) for value in self._shared_memory.values()),
            }
            if self._logging_enabled:
                logging.debug(f"Shared memory statistics: {stats}")
            return stats

    def _get_size(self, obj: Any) -> int:
        """
        Get the approximate memory size of an object.

        :param obj: The object to measure.
        :return: Size in bytes.
        """
        import sys
        return sys.getsizeof(obj)

    def remove_key(self, key: str):
        """
        Remove a key from the shared memory.

        :param key: The key to remove.
        """
        with self._lock:
            self._shared_memory.pop(key, None)
            self._data_timestamps.pop(key, None)
            self._data_versions.pop(key, None)
            self._data_expirations.pop(key, None)
            self._subscribers.pop(key, None)
            self._event_conditions.pop(key, None)
            if self._logging_enabled:
                logging.debug(f"Key removed from shared memory: {key}")

    def set_value_if_unset(self, key: str, value: Any) -> bool:
        """
        Set a value only if the key does not already exist.

        :param key: The key to set.
        :param value: The value to set.
        :return: True if the value was set, False if the key already exists.
        """
        with self._lock:
            if key not in self._shared_memory:
                self.write(key, value)
                return True
            else:
                return False

    def increment(self, key: str, amount: float = 1.0) -> float:
        """
        Increment a numerical value in the shared memory.

        :param key: The key to increment.
        :param amount: The amount to increment by.
        :return: The new value.
        """
        with self._lock:
            value = self._shared_memory.get(key, 0.0)
            if not isinstance(value, (int, float)):
                raise TypeError(f"Value for key '{key}' is not numeric.")
            new_value = value + amount
            self.write(key, new_value)
            return new_value

    def decrement(self, key: str, amount: float = 1.0) -> float:
        """
        Decrement a numerical value in the shared memory.

        :param key: The key to decrement.
        :param amount: The amount to decrement by.
        :return: The new value.
        """
        return self.increment(key, -amount)

    def append_to_list(self, key: str, value: Any):
        """
        Append a value to a list in the shared memory.

        :param key: The key of the list.
        :param value: The value to append.
        """
        with self._lock:
            lst = self._shared_memory.get(key, [])
            if not isinstance(lst, list):
                raise TypeError(f"Value for key '{key}' is not a list.")
            lst.append(value)
            self.write(key, lst)

    def get_list(self, key: str) -> List[Any]:
        """
        Get a list from the shared memory.

        :param key: The key of the list.
        :return: The list.
        """
        value = self.read(key)
        if not isinstance(value, list):
            raise TypeError(f"Value for key '{key}' is not a list.")
        return value

    def set_if_version(self, key: str, value: Any, expected_version: int) -> bool:
        """
        Set the value only if the current version matches the expected version.

        :param key: The key to set.
        :param value: The new value.
        :param expected_version: The expected current version.
        :return: True if the value was set, False otherwise.
        """
        with self._lock:
            current_version = self._data_versions.get(key, 0)
            if current_version == expected_version:
                self.write(key, value)
                return True
            else:
                return False

    def atomic_transaction(self, operations: List[Tuple[str, Any]]):
        """
        Perform multiple write operations atomically.

        :param operations: A list of (key, value) tuples.
        """
        with self._lock:
            for key, value in operations:
                self._shared_memory[key] = value
                timestamp = time.time()
                self._data_timestamps[key] = timestamp
                self._data_versions[key] = self._data_versions.get(key, 0) + 1
                self._data_expirations.pop(key, None)
            if self._logging_enabled:
                logging.debug(f"Atomic transaction performed on keys: {[key for key, _ in operations]}")

    def merge_dict(self, key: str, updates: Dict[Any, Any]):
        """
        Merge updates into a dictionary stored in the shared memory.

        :param key: The key of the dictionary.
        :param updates: A dictionary of updates to merge.
        """
        with self._lock:
            existing = self._shared_memory.get(key, {})
            if not isinstance(existing, dict):
                raise TypeError(f"Value for key '{key}' is not a dictionary.")
            existing.update(updates)
            self.write(key, existing)

    def get_dict(self, key: str) -> Dict[Any, Any]:
        """
        Get a dictionary from the shared memory.

        :param key: The key of the dictionary.
        :return: The dictionary.
        """
        value = self.read(key)
        if not isinstance(value, dict):
            raise TypeError(f"Value for key '{key}' is not a dictionary.")
        return value

    def acquire_lock(self, key: str, timeout: Optional[float] = None) -> bool:
        """
        Acquire a lock associated with a key.

        :param key: The key of the lock.
        :param timeout: Optional timeout in seconds.
        :return: True if the lock was acquired, False if timed out.
        """
        condition = self._event_conditions.setdefault(key, threading.Condition())
        acquired = condition.acquire(timeout=timeout)
        if self._logging_enabled:
            logging.debug(f"Lock {'acquired' if acquired else 'not acquired'} for key '{key}'")
        return acquired

    def release_lock(self, key: str):
        """
        Release a lock associated with a key.

        :param key: The key of the lock.
        """
        condition = self._event_conditions.get(key)
        if condition:
            condition.release()
            if self._logging_enabled:
                logging.debug(f"Lock released for key '{key}'")

    # Additional methods can be added here as needed.

