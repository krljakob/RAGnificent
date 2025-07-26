"""
Resource management module for RAGnificent.

Provides utilities for managing system resources, connection pooling,
memory management, and graceful cleanup of resources.
"""

import atexit
import gc
import logging
import os
import signal
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import psutil

logger = logging.getLogger(__name__)


try:
    from .stats import StatsMixin
except ImportError:
    from stats import StatsMixin


class ResourceManager(StatsMixin):
    """
    Centralized resource manager for the application.

    Handles:
    - Connection pooling for external services
    - Memory monitoring and management
    - Graceful shutdown and resource cleanup
    - Thread and process management
    """

    def __init__(
        self,
        max_memory_percent: float = 80.0,
        max_connections: int = 100,
        max_thread_workers: int = 20,
        cleanup_interval: int = 60,
        enable_monitoring: bool = True,
    ):
        """
        Initialize the resource manager.

        Args:
            max_memory_percent: Maximum memory usage percentage before cleanup
            max_connections: Maximum number of connections to maintain
            max_thread_workers: Maximum number of thread workers
            cleanup_interval: Interval in seconds for periodic cleanup
            enable_monitoring: Whether to enable resource monitoring
        """
        self.max_memory_percent = max_memory_percent
        self.max_connections = max_connections
        self.max_thread_workers = max_thread_workers
        self.cleanup_interval = cleanup_interval
        super().__init__(enable_stats=enable_monitoring)

        self.connection_pools: Dict[str, Any] = {}

        self.executor = ThreadPoolExecutor(max_workers=max_thread_workers)

        self.active_resources: Set[str] = set()
        self.resource_locks: Dict[str, threading.Lock] = {}

        self.monitoring_thread = None
        self.monitoring_stop_event = threading.Event()

        atexit.register(self.cleanup_all)
        signal.signal(signal.SIGTERM, self._signal_handler)
        signal.signal(signal.SIGINT, self._signal_handler)

        if enable_monitoring:
            self.start_monitoring()

        logger.info(
            f"ResourceManager initialized with max_memory_percent={max_memory_percent}%, "
            f"max_connections={max_connections}, max_thread_workers={max_thread_workers}"
        )

    def start_monitoring(self):
        """Start the resource monitoring thread."""
        if self.monitoring_thread is not None:
            return

        self.monitoring_stop_event.clear()
        self.monitoring_thread = threading.Thread(
            target=self._monitoring_loop, daemon=True, name="ResourceMonitor"
        )
        self.monitoring_thread.start()
        logger.info("Resource monitoring started")

    def stop_monitoring(self):
        """Stop the resource monitoring thread."""
        if self.monitoring_thread is None:
            return

        self.monitoring_stop_event.set()
        self.monitoring_thread.join(timeout=5)
        self.monitoring_thread = None
        logger.info("Resource monitoring stopped")

    def _monitoring_loop(self):
        """Background thread for monitoring resources."""
        while not self.monitoring_stop_event.is_set():
            try:
                self._check_memory_usage()
                self._check_connection_pools()

                self.monitoring_stop_event.wait(self.cleanup_interval)
            except Exception as e:
                logger.error(f"Error in resource monitoring: {e}")
                time.sleep(5)  # Sleep briefly to avoid tight loop on error

    def _check_memory_usage(self):
        """Check memory usage and trigger cleanup if necessary."""
        memory_percent = psutil.virtual_memory().percent

        if memory_percent > self.max_memory_percent:
            logger.warning(
                f"High memory usage detected: {memory_percent:.1f}% > {self.max_memory_percent:.1f}%"
            )
            self.cleanup_memory()

    def _check_connection_pools(self):
        """Check connection pools and close idle connections."""
        for pool_name, pool in self.connection_pools.items():
            if hasattr(pool, "close_idle_connections"):
                try:
                    pool.close_idle_connections()
                    logger.debug(f"Closed idle connections in pool: {pool_name}")
                except Exception as e:
                    logger.error(f"Error closing idle connections in {pool_name}: {e}")

    def cleanup_memory(self):
        """
        Perform memory cleanup operations.

        This includes:
        - Triggering garbage collection
        - Clearing caches
        - Closing idle connections
        """
        logger.info("Performing memory cleanup")

        collected = gc.collect(generation=2)
        logger.info(f"Garbage collection: collected {collected} objects")

        for pool_name, pool in self.connection_pools.items():
            if hasattr(pool, "close_idle_connections"):
                try:
                    pool.close_idle_connections()
                except Exception as e:
                    logger.error(f"Error closing idle connections in {pool_name}: {e}")

        memory_percent = psutil.virtual_memory().percent
        logger.info(f"Memory usage after cleanup: {memory_percent:.1f}%")

    def cleanup_all(self):
        """
        Clean up all resources before shutdown.

        This is automatically called on program exit.
        """
        logger.info("Cleaning up all resources")

        self.stop_monitoring()

        self.executor.shutdown(wait=True)

        for pool_name, pool in self.connection_pools.items():
            try:
                if hasattr(pool, "close"):
                    pool.close()
                elif hasattr(pool, "shutdown"):
                    pool.shutdown()
                logger.info(f"Closed connection pool: {pool_name}")
            except Exception as e:
                logger.error(f"Error closing connection pool {pool_name}: {e}")

        self.active_resources.clear()
        self.connection_pools.clear()

        logger.info("Resource cleanup completed")

    def _signal_handler(self, signum, frame):
        """Handle termination signals."""
        logger.info(f"Received signal {signum}, cleaning up resources")
        self.cleanup_all()
        signal.default_int_handler(signum, frame)

    def register_connection_pool(self, name: str, pool: Any) -> None:
        """
        Register a connection pool for management.

        Args:
            name: Name of the connection pool
            pool: The connection pool object
        """
        self.connection_pools[name] = pool
        logger.info(f"Registered connection pool: {name}")

    def get_connection_pool(self, name: str) -> Optional[Any]:
        """
        Get a registered connection pool.

        Args:
            name: Name of the connection pool

        Returns:
            The connection pool or None if not found
        """
        return self.connection_pools.get(name)

    def track_resource(self, resource_id: str) -> None:
        """
        Track a resource for cleanup.

        Args:
            resource_id: Unique identifier for the resource
        """
        self.active_resources.add(resource_id)
        if resource_id not in self.resource_locks:
            self.resource_locks[resource_id] = threading.Lock()

    def release_resource(self, resource_id: str) -> None:
        """
        Release a tracked resource.

        Args:
            resource_id: Unique identifier for the resource
        """
        if resource_id in self.active_resources:
            self.active_resources.remove(resource_id)

        if resource_id in self.resource_locks:
            del self.resource_locks[resource_id]

    @contextmanager
    def resource_lock(self, resource_id: str):
        """
        Context manager for safely accessing a resource.

        Args:
            resource_id: Unique identifier for the resource

        Yields:
            None
        """
        if resource_id not in self.resource_locks:
            self.resource_locks[resource_id] = threading.Lock()

        lock = self.resource_locks[resource_id]
        try:
            lock.acquire()
            yield
        finally:
            lock.release()

    def submit_task(self, fn: Callable, *args, **kwargs):
        """
        Submit a task to the thread pool.

        Args:
            fn: Function to execute
            *args: Arguments to pass to the function
            **kwargs: Keyword arguments to pass to the function

        Returns:
            Future object representing the task
        """
        return self.executor.submit(fn, *args, **kwargs)

    def get_memory_usage(self) -> Dict[str, float]:
        """
        Get current memory usage statistics.

        Returns:
            Dictionary with memory usage information
        """
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()

        virtual_memory = psutil.virtual_memory()

        return {
            "rss_mb": memory_info.rss / (1024 * 1024),  # Resident Set Size in MB
            "vms_mb": memory_info.vms / (1024 * 1024),  # Virtual Memory Size in MB
            "percent": process.memory_percent(),  # Process memory as percentage of total
            "system_total_mb": virtual_memory.total
            / (1024 * 1024),  # Total system memory in MB
            "system_available_mb": virtual_memory.available
            / (1024 * 1024),  # Available system memory in MB
            "system_percent": virtual_memory.percent,  # System memory usage percentage
        }

    def get_resource_usage(self) -> Dict[str, Any]:
        """
        Get resource usage statistics.

        Returns:
            Dictionary with resource usage information
        """
        memory_usage = self.get_memory_usage()

        return {
            "memory": memory_usage,
            "cpu_percent": psutil.cpu_percent(interval=0.1),
            "thread_count": threading.active_count(),
            "active_resources": len(self.active_resources),
            "connection_pools": {
                name: self._get_pool_stats(pool)
                for name, pool in self.connection_pools.items()
            },
        }

    def _get_pool_stats(self, pool: Any) -> Dict[str, Any]:
        """Get statistics for a connection pool."""
        stats = {
            attr: getattr(pool, attr)
            for attr in [
                "size",
                "qsize",
                "maxsize",
                "idle",
                "in_use",
                "connections",
            ]
            if hasattr(pool, attr)
        }
        for method in ["get_stats", "stats", "get_info"]:
            if hasattr(pool, method) and callable(getattr(pool, method)):
                try:
                    method_stats = getattr(pool, method)()
                    if isinstance(method_stats, dict):
                        stats |= method_stats
                except Exception:
                    pass

        return stats


_RESOURCE_MANAGER = None


def get_resource_manager() -> ResourceManager:
    """
    Get the singleton resource manager instance.

    Returns:
        ResourceManager instance
    """
    global _RESOURCE_MANAGER
    if _RESOURCE_MANAGER is None:
        _RESOURCE_MANAGER = ResourceManager()
    return _RESOURCE_MANAGER


@contextmanager
def managed_resource(resource_id: str):
    """
    Context manager for safely using a managed resource.

    Args:
        resource_id: Unique identifier for the resource

    Yields:
        None
    """
    manager = get_resource_manager()
    manager.track_resource(resource_id)

    try:
        with manager.resource_lock(resource_id):
            yield
    finally:
        manager.release_resource(resource_id)


class ConnectionPool:
    """
    Generic connection pool for external services.

    This class provides a reusable pattern for connection pooling
    that can be extended for specific services (HTTP, database, etc.).
    """

    def __init__(
        self,
        name: str,
        max_connections: int = 10,
        connection_timeout: float = 10.0,
        idle_timeout: float = 60.0,
        connection_factory: Optional[Callable[[], Any]] = None,
    ):
        """
        Initialize the connection pool.

        Args:
            name: Name of the connection pool
            max_connections: Maximum number of connections
            connection_timeout: Timeout for acquiring a connection
            idle_timeout: Timeout for idle connections
            connection_factory: Function to create new connections
        """
        self.name = name
        self.max_connections = max_connections
        self.connection_timeout = connection_timeout
        self.idle_timeout = idle_timeout
        self.connection_factory = connection_factory

        self.connections: List[Tuple[Any, float]] = []  # (connection, last_used_time)
        self.in_use: Set[Any] = set()
        self.lock = threading.RLock()

        get_resource_manager().register_connection_pool(name, self)

    @property
    def size(self) -> int:
        """Total number of connections (idle + in-use)."""
        return len(self.connections) + len(self.in_use)

    @property
    def idle(self) -> int:
        """Number of idle connections."""
        return len(self.connections)

    @property
    def active(self) -> int:
        """Number of active connections."""
        return len(self.in_use)

    def get_connection(self) -> Any:
        """
        Get a connection from the pool.

        Returns:
            A connection object

        Raises:
            TimeoutError: If no connection is available within the timeout
        """
        start_time = time.time()

        while time.time() - start_time < self.connection_timeout:
            with self.lock:
                if self.connections:
                    conn, _ = self.connections.pop(0)
                    self.in_use.add(conn)
                    return conn

                if self.size < self.max_connections and self.connection_factory:
                    conn = self.connection_factory()
                    self.in_use.add(conn)
                    return conn

            time.sleep(0.1)

        raise TimeoutError(f"Timeout waiting for connection from pool: {self.name}")

    def release_connection(self, conn: Any) -> None:
        """
        Release a connection back to the pool.

        Args:
            conn: Connection to release
        """
        with self.lock:
            if conn in self.in_use:
                self.in_use.remove(conn)
                self.connections.append((conn, time.time()))

    def close_idle_connections(self) -> int:
        """
        Close idle connections that have exceeded the idle timeout.

        Returns:
            Number of connections closed
        """
        closed_count = 0
        current_time = time.time()

        with self.lock:
            active_connections = []

            for conn, last_used in self.connections:
                if current_time - last_used > self.idle_timeout:
                    self._close_connection(conn)
                    closed_count += 1
                else:
                    active_connections.append((conn, last_used))

            self.connections = active_connections

        return closed_count

    def _close_connection(self, conn: Any) -> None:
        """Close a single connection."""
        try:
            if hasattr(conn, "close") and callable(conn.close):
                conn.close()
            elif hasattr(conn, "disconnect") and callable(conn.disconnect):
                conn.disconnect()
        except Exception as e:
            logger.warning(f"Error closing connection in pool {self.name}: {e}")

    def close(self) -> None:
        """Close all connections in the pool."""
        with self.lock:
            for conn, _ in self.connections:
                self._close_connection(conn)

            for conn in self.in_use:
                self._close_connection(conn)

            self.connections = []
            self.in_use = set()

    @contextmanager
    def connection(self):
        """
        Context manager for safely using a connection from the pool.

        Yields:
            A connection from the pool
        """
        conn = self.get_connection()
        try:
            yield conn
        finally:
            self.release_connection(conn)

    def _get_stats_implementation(self) -> Dict[str, Any]:
        """
        Get statistics about the connection pool.

        Returns:
            Dictionary with pool statistics
        """
        with self.lock:
            return {
                "name": self.name,
                "size": self.size,
                "max_connections": self.max_connections,
                "idle": self.idle,
                "in_use": self.active,
                "utilization": (
                    self.active / self.max_connections
                    if self.max_connections > 0
                    else 0
                ),
            }


@dataclass
class MemoryMonitor:
    """
    Memory usage monitor for tracking and limiting memory usage.

    This class provides utilities for monitoring memory usage and
    applying backpressure when memory usage exceeds thresholds.
    """

    max_memory_percent: float = 80.0
    check_interval: float = 1.0
    warning_threshold: float = 70.0
    critical_threshold: float = 90.0
    callbacks: Dict[str, Callable] = field(default_factory=dict)

    def __post_init__(self):
        """Initialize after instance creation."""
        self.last_check_time = 0.0
        self.last_memory_percent = 0.0

    def check_memory(self, force: bool = False) -> float:
        """
        Check current memory usage.

        Args:
            force: Whether to force a check even if the interval hasn't elapsed

        Returns:
            Current memory usage percentage
        """
        current_time = time.time()

        if not force and current_time - self.last_check_time < self.check_interval:
            return self.last_memory_percent

        memory_percent = psutil.virtual_memory().percent
        self.last_memory_percent = memory_percent
        self.last_check_time = current_time

        if memory_percent >= self.critical_threshold:
            self._trigger_callbacks("critical")
        elif memory_percent >= self.warning_threshold:
            self._trigger_callbacks("warning")
        else:
            self._trigger_callbacks("normal")

        return memory_percent

    def _trigger_callbacks(self, level: str) -> None:
        """Trigger callbacks for a specific threshold level."""
        if level in self.callbacks:
            try:
                self.callbacks[level]()
            except Exception as e:
                logger.error(f"Error in memory monitor callback for {level}: {e}")

    def register_callback(self, level: str, callback: Callable) -> None:
        """
        Register a callback for a specific threshold level.

        Args:
            level: Threshold level ('normal', 'warning', or 'critical')
            callback: Function to call when threshold is reached
        """
        if level in {"normal", "warning", "critical"}:
            self.callbacks[level] = callback
        else:
            raise ValueError("Level must be 'normal', 'warning', or 'critical'")

    def should_apply_backpressure(self) -> bool:
        """
        Check if backpressure should be applied based on memory usage.

        Returns:
            True if backpressure should be applied, False otherwise
        """
        memory_percent = self.check_memory()
        return memory_percent >= self.max_memory_percent

    def get_backpressure_delay(self) -> float:
        """
        Calculate delay to apply as backpressure based on memory usage.

        Returns:
            Delay in seconds to apply
        """
        memory_percent = self.check_memory()

        if memory_percent >= self.critical_threshold:
            return 1.0
        if memory_percent >= self.warning_threshold:
            severity = (memory_percent - self.warning_threshold) / (
                self.critical_threshold - self.warning_threshold
            )
            return min(1.0, max(0.1, severity))
        return 0.0

    @contextmanager
    def backpressure(self):
        """
        Context manager that applies backpressure if needed.

        This will sleep for an appropriate amount of time if memory
        usage exceeds thresholds.

        Yields:
            None
        """
        delay = self.get_backpressure_delay()

        if delay > 0:
            logger.debug(f"Applying backpressure: sleeping for {delay:.2f}s")
            time.sleep(delay)

        yield

        self.check_memory(force=True)
