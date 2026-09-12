"""Request abort/cancellation system."""
import threading
import logging
import select
import socket
from typing import Optional

logger = logging.getLogger(__name__)


class GenerationAbortedError(Exception):
    """Raised when generation is aborted (client disconnect or explicit abort)."""
    pass


def _extract_socket(wsgi_input) -> Optional[socket.socket]:
    """Traverse werkzeug's stream wrappers to find the raw socket.
    
    Werkzeug wraps the socket as:
      LimitedStream -> BufferedReader -> SocketIO -> socket
    """
    obj = wsgi_input
    # Walk through wrapper layers looking for the underlying socket
    for _depth in range(10):
        # Direct socket object
        if isinstance(obj, socket.socket):
            return obj
        # SocketIO exposes the socket via .raw or ._sock
        if isinstance(obj, socket.SocketIO):
            raw = getattr(obj, '_sock', None)
            if isinstance(raw, socket.socket):
                return raw
        # BufferedReader/BufferedWriter -> .raw
        raw = getattr(obj, 'raw', None)
        if raw is not None:
            obj = raw
            continue
        # Werkzeug LimitedStream -> .stream
        stream = getattr(obj, 'stream', None)
        if stream is not None:
            obj = stream
            continue
        # Some wrappers use _stream
        _stream = getattr(obj, '_stream', None)
        if _stream is not None:
            obj = _stream
            continue
        # Try fileno() to reconstruct the socket
        fileno_fn = getattr(obj, 'fileno', None)
        if fileno_fn is not None:
            try:
                fd = fileno_fn()
                return socket.fromfd(fd, socket.AF_INET, socket.SOCK_STREAM)
            except (OSError, ValueError):
                pass
        break
    return None


def is_client_disconnected() -> bool:
    """Check if the Flask client has disconnected.
    
    Uses socket-level probing (select + recv MSG_PEEK) for reliable
    detection on werkzeug's dev server where wsgi.input.closed is
    unreliable.
    """
    try:
        from flask import request, has_request_context

        if not has_request_context():
            return False

        environ = request.environ
        if environ is None:
            return True

        wsgi_input = environ.get('wsgi.input')

        # Fast path: check if the WSGI input stream reports itself closed
        if wsgi_input is not None and getattr(wsgi_input, 'closed', False):
            return True

        # Socket-level disconnect detection
        sock = _extract_socket(wsgi_input)
        if sock is None:
            return False

        try:
            # Verify the remote end is still there
            sock.getpeername()
        except (OSError, socket.error):
            return True

        try:
            readable, _, exceptional = select.select([sock], [], [sock], 0)
        except (OSError, ValueError, socket.error):
            # Bad fd or closed socket
            return True

        if exceptional:
            return True

        if readable:
            try:
                data = sock.recv(1, socket.MSG_PEEK)
                if not data:
                    # Empty recv == peer closed the connection
                    return True
            except ConnectionError:
                return True
            except BlockingIOError:
                # Socket has no data but is still alive
                pass
            except (OSError, socket.error):
                return True

        return False

    except Exception:
        # On any unexpected error, assume still connected
        return False

class AbortController:
    """Controls request cancellation."""
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._abort_event = threading.Event()
                    cls._instance._active = False
                    cls._instance._current_job_id: Optional[str] = None
        return cls._instance
    
    def start_generation(self):
        """Mark generation as active, clear abort flag."""
        self._abort_event.clear()
        self._active = True
        
    def end_generation(self):
        """Mark generation as complete."""
        self._active = False
        
    def set_current_job(self, job_id: Optional[str]):
        """Track which job is currently processing."""
        self._current_job_id = job_id
    
    def get_current_job(self) -> Optional[str]:
        """Get the currently processing job ID."""
        return self._current_job_id
    
    def abort(self) -> Optional[str]:
        """
        Request abort. Returns the job_id that was aborted, or None if no active job.
        """
        if self._active:
            self._abort_event.set()
            job_id = self._current_job_id
            logger.info(f"Abort requested for job {job_id[:8] if job_id else 'unknown'}")
            return job_id
        return None
    
    def should_abort(self) -> bool:
        """Check if abort was requested or client disconnected."""
        if self._abort_event.is_set():
            return True
        # Also check for client disconnect during active generation
        if self._active and is_client_disconnected():
            logger.info("Client disconnect detected, aborting generation")
            self._abort_event.set()
            return True
        return False
    
    @property
    def is_active(self) -> bool:
        return self._active

abort_controller = AbortController()
