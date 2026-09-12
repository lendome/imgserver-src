"""Production-quality request queue system with worker thread."""

import logging
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class Job:
    """Represents a queued generation job."""
    id: str
    request: dict
    status: str = "queued"  # queued, processing, done, failed, cancelled
    result: Optional[Any] = None
    error: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    position: int = 0
    
    def to_dict(self) -> dict:
        """Convert job to dictionary for API responses."""
        return {
            "id": self.id,
            "status": self.status,
            "result": self.result,
            "error": self.error,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "position": self.position,
            "wait_time": (self.started_at - self.created_at) if self.started_at else None,
            "processing_time": (self.completed_at - self.started_at) if self.completed_at and self.started_at else None,
        }


class RequestQueue:
    """
    Thread-safe request queue with job tracking.
    
    Singleton pattern ensures one queue per process.
    Uses deque for O(1) append/popleft operations.
    """
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls, max_size: int = 50, cleanup_after: float = 300.0):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self, max_size: int = 50, cleanup_after: float = 300.0):
        # Only initialize once
        if self._initialized:
            return
        
        self._queue: deque = deque()  # Queue of job IDs
        self._jobs: Dict[str, Job] = {}  # O(1) lookup by ID
        self._queue_lock = threading.Lock()
        self._condition = threading.Condition(self._queue_lock)
        self._max_size = max_size
        self._cleanup_after = cleanup_after  # Seconds before completed jobs are removed
        self._processing_count = 0
        self._shutdown = False
        self._initialized = True
        
        # Start cleanup thread
        self._cleanup_thread = threading.Thread(
            target=self._cleanup_loop,
            daemon=True,
            name="QueueCleanup"
        )
        self._cleanup_thread.start()
        
        logger.info(f"RequestQueue initialized (max_size={max_size}, cleanup_after={cleanup_after}s)")
    
    def submit(self, request: dict) -> Job:
        """
        Submit a job to the queue.
        
        Args:
            request: Generation parameters dictionary
            
        Returns:
            Job object (status will be 'queued')
            
        Raises:
            RuntimeError: If queue is full
        """
        with self._condition:
            if len(self._queue) >= self._max_size:
                raise RuntimeError(f"Queue is full (max {self._max_size} jobs)")
            
            job = Job(
                id=str(uuid.uuid4()),
                request=request,
                position=len(self._queue) + 1
            )
            
            self._jobs[job.id] = job
            self._queue.append(job.id)
            
            # Notify waiting workers
            self._condition.notify()
            
            logger.info(f"Job {job.id[:8]} queued at position {job.position}")
            return job
    
    def get_next(self, timeout: Optional[float] = None) -> Optional[Job]:
        """
        Get the next job for processing. Blocks if queue is empty.
        
        Called by worker threads. Moves job to 'processing' status.
        
        Args:
            timeout: Max seconds to wait (None = wait forever)
            
        Returns:
            Job to process, or None if timeout/shutdown
        """
        with self._condition:
            # Wait for job or shutdown
            while not self._queue and not self._shutdown:
                if not self._condition.wait(timeout=timeout):
                    return None  # Timeout
            
            if self._shutdown:
                return None
            
            job_id = self._queue.popleft()
            job = self._jobs.get(job_id)
            
            if job is None:
                logger.warning(f"Job {job_id[:8]} not found in storage")
                return self.get_next(timeout)  # Try next
            
            # Update job status
            job.status = "processing"
            job.started_at = time.time()
            self._processing_count += 1
            
            # Update positions for remaining queued jobs
            self._update_positions()
            
            logger.info(f"Job {job.id[:8]} started processing")
            return job
    
    def get_job(self, job_id: str) -> Optional[Job]:
        """Get job by ID."""
        with self._queue_lock:
            return self._jobs.get(job_id)
    
    def get_queue_status(self) -> dict:
        """
        Get current queue status.
        
        Returns:
            Dict with queue_depth, processing_count, and list of jobs
        """
        with self._queue_lock:
            # Get queued jobs in order
            queued_jobs = []
            for job_id in self._queue:
                job = self._jobs.get(job_id)
                if job:
                    queued_jobs.append(job.to_dict())
            
            # Get processing jobs
            processing_jobs = [
                j.to_dict() for j in self._jobs.values() 
                if j.status == "processing"
            ]
            
            # Recent completed/failed (last 10)
            completed_jobs = sorted(
                [j for j in self._jobs.values() if j.status in ("done", "failed", "cancelled")],
                key=lambda j: j.completed_at or 0,
                reverse=True
            )[:10]
            
            return {
                "queue_depth": len(self._queue),
                "processing_count": self._processing_count,
                "max_size": self._max_size,
                "queued": queued_jobs,
                "processing": processing_jobs,
                "recent_completed": [j.to_dict() for j in completed_jobs],
            }
    
    def complete_job(self, job_id: str, result: Any):
        """Mark job as completed with result."""
        with self._queue_lock:
            job = self._jobs.get(job_id)
            if job is None:
                logger.warning(f"Cannot complete unknown job {job_id[:8]}")
                return
            
            job.status = "done"
            job.result = result
            job.completed_at = time.time()
            self._processing_count = max(0, self._processing_count - 1)
            
            processing_time = job.completed_at - (job.started_at or job.created_at)
            logger.info(f"Job {job_id[:8]} completed in {processing_time:.2f}s")
    
    def fail_job(self, job_id: str, error: str):
        """Mark job as failed with error message."""
        with self._queue_lock:
            job = self._jobs.get(job_id)
            if job is None:
                logger.warning(f"Cannot fail unknown job {job_id[:8]}")
                return
            
            job.status = "failed"
            job.error = error
            job.completed_at = time.time()
            self._processing_count = max(0, self._processing_count - 1)
            
            logger.error(f"Job {job_id[:8]} failed: {error}")
    
    def cancel_job(self, job_id: str, abort_controller=None) -> bool:
        """
        Cancel a job. Supports both queued and processing jobs.
        
        Args:
            job_id: Job ID to cancel
            abort_controller: Optional AbortController to signal processing job cancellation
            
        Returns:
            True if cancelled, False if not found or not cancellable
        """
        with self._queue_lock:
            job = self._jobs.get(job_id)
            if job is None:
                return False
            
            if job.status == "queued":
                # Remove from queue
                try:
                    self._queue.remove(job_id)
                except ValueError:
                    pass  # Already removed
                
                job.status = "cancelled"
                job.completed_at = time.time()
                self._update_positions()
                
                logger.info(f"Job {job_id[:8]} cancelled (was queued)")
                return True
            
            elif job.status == "processing":
                # Signal the processing job to abort
                if abort_controller:
                    abort_controller.abort()
                
                job.status = "cancelled"
                job.completed_at = time.time()
                self._processing_count = max(0, self._processing_count - 1)
                
                logger.info(f"Job {job_id[:8]} cancelled (was processing)")
                return True
            
            else:
                logger.warning(f"Cannot cancel job {job_id[:8]} in status '{job.status}'")
                return False
    
    def _update_positions(self):
        """Update position numbers for all queued jobs."""
        for i, job_id in enumerate(self._queue):
            job = self._jobs.get(job_id)
            if job:
                job.position = i + 1
    
    def _cleanup_loop(self):
        """Background thread to clean up old completed jobs."""
        while not self._shutdown:
            time.sleep(60)  # Check every minute
            
            with self._queue_lock:
                now = time.time()
                to_remove = []
                
                for job_id, job in self._jobs.items():
                    if job.status in ("done", "failed", "cancelled"):
                        if job.completed_at and (now - job.completed_at) > self._cleanup_after:
                            to_remove.append(job_id)
                
                for job_id in to_remove:
                    del self._jobs[job_id]
                
                if to_remove:
                    logger.debug(f"Cleaned up {len(to_remove)} old jobs")
    
    def shutdown(self):
        """Signal shutdown to waiting workers."""
        with self._condition:
            self._shutdown = True
            self._condition.notify_all()
        logger.info("RequestQueue shutdown signaled")
    
    @classmethod
    def reset_instance(cls):
        """Reset singleton (for testing only)."""
        with cls._lock:
            if cls._instance is not None:
                cls._instance.shutdown()
            cls._instance = None


class QueueWorker:
    """
    Worker thread that processes jobs from the queue.
    
    Acquires GPU lock, calls pipeline.generate(), handles errors gracefully.
    """
    
    def __init__(
        self,
        queue: RequestQueue,
        generate_fn: Callable[[dict], Any],
        name: str = "QueueWorker"
    ):
        """
        Initialize queue worker.
        
        Args:
            queue: RequestQueue to pull jobs from
            generate_fn: Function to call for generation (receives request dict, returns result)
            name: Thread name for logging
        """
        self._queue = queue
        self._generate_fn = generate_fn
        self._name = name
        self._thread: Optional[threading.Thread] = None
        self._running = False
        
        # Import here to avoid circular imports
        from .gpu_lock import gpu_lock
        from .abort import abort_controller
        self._gpu_lock = gpu_lock
        self._abort_controller = abort_controller
    
    def start(self):
        """Start the worker thread."""
        if self._thread is not None and self._thread.is_alive():
            logger.warning(f"{self._name} already running")
            return
        
        self._running = True
        self._thread = threading.Thread(
            target=self._worker_loop,
            daemon=True,
            name=self._name
        )
        self._thread.start()
        logger.info(f"{self._name} started")
    
    def stop(self):
        """Stop the worker thread."""
        self._running = False
        logger.info(f"{self._name} stop requested")
    
    def _worker_loop(self):
        """Main worker loop - gets jobs and processes them."""
        logger.info(f"{self._name} entering work loop")
        
        while self._running:
            # Wait for job (1 second timeout to check _running flag)
            job = self._queue.get_next(timeout=1.0)
            
            if job is None:
                continue
            
            self._process_job(job)
        
        logger.info(f"{self._name} exiting work loop")
    
    def _process_job(self, job: Job):
        """Process a single job."""
        start_time = time.time()
        job_id_short = job.id[:8]
        
        logger.info(f"{self._name} processing job {job_id_short}")
        
        # Track the current job for cancellation support
        self._abort_controller.set_current_job(job.id)
        
        try:
            # Acquire GPU lock
            with self._gpu_lock.acquire(holder=f"job-{job_id_short}"):
                # Check for abort before starting
                if self._abort_controller.should_abort():
                    self._queue.fail_job(job.id, "Aborted before start")
                    return
                
                # Signal generation start
                self._abort_controller.start_generation()
                
                try:
                    # Call the generation function
                    result = self._generate_fn(job.request)
                    
                    # Check for abort after generation
                    if self._abort_controller.should_abort():
                        self._queue.fail_job(job.id, "Generation aborted")
                    else:
                        self._queue.complete_job(job.id, result)
                        
                finally:
                    self._abort_controller.end_generation()
                    
        except Exception as e:
            error_msg = f"{type(e).__name__}: {str(e)}"
            logger.exception(f"Job {job_id_short} failed with exception")
            self._queue.fail_job(job.id, error_msg)
        finally:
            # Clear current job tracking
            self._abort_controller.set_current_job(None)
        
        elapsed = time.time() - start_time
        logger.info(f"{self._name} finished job {job_id_short} in {elapsed:.2f}s")


# Singleton instance
request_queue = RequestQueue()


def get_queue() -> RequestQueue:
    """Get the global request queue instance."""
    return request_queue
