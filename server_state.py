"""Server state tracking."""
import time
from dataclasses import dataclass, field

@dataclass
class ServerState:
    """Tracks server statistics."""
    start_time: float = field(default_factory=time.time)
    generation_count: int = 0
    error_count: int = 0
    last_generation_time: float = 0
    current_task: str = ""
    
    def record_generation(self):
        self.generation_count += 1
        self.last_generation_time = time.time()
    
    def record_error(self):
        self.error_count += 1
    
    @property
    def uptime(self) -> float:
        return time.time() - self.start_time
    
    @property
    def idle_time(self) -> float:
        if self.last_generation_time == 0:
            return self.uptime
        return time.time() - self.last_generation_time

server_state = ServerState()
