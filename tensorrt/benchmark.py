"""Performance benchmark suite for TensorRT vs PyTorch comparison."""
import gc
import json
import logging
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable
import torch

logger = logging.getLogger(__name__)

@dataclass
class BenchmarkResult:
    """Results from a single benchmark run."""
    name: str
    backend: str  # "pytorch" or "tensorrt"
    iterations: int
    warmup_iterations: int
    
    # Timing metrics (milliseconds)
    mean_latency_ms: float
    std_latency_ms: float
    min_latency_ms: float
    max_latency_ms: float
    p50_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    
    # Memory metrics (MB)
    peak_memory_mb: float
    allocated_memory_mb: float
    
    # Throughput
    throughput_per_sec: float
    
    # Metadata
    input_shape: List[int]
    precision: str
    timestamp: str
    
    def to_dict(self) -> dict:
        return asdict(self)

@dataclass
class ComparisonResult:
    """A/B comparison between PyTorch and TensorRT."""
    name: str
    pytorch: BenchmarkResult
    tensorrt: BenchmarkResult
    
    # Comparison metrics
    speedup: float  # TRT vs PyTorch
    memory_reduction: float  # Percentage reduction
    latency_improvement_percent: float
    
    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "pytorch": self.pytorch.to_dict(),
            "tensorrt": self.tensorrt.to_dict(),
            "speedup": self.speedup,
            "memory_reduction": self.memory_reduction,
            "latency_improvement_percent": self.latency_improvement_percent,
        }

class TensorRTBenchmark:
    """Benchmark suite for TensorRT performance measurement."""
    
    def __init__(
        self,
        warmup_iterations: int = 5,
        benchmark_iterations: int = 20,
        output_dir: str = "benchmark_results",
    ):
        self.warmup_iterations = warmup_iterations
        self.benchmark_iterations = benchmark_iterations
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self._results: List[BenchmarkResult] = []
    
    def _get_memory_stats(self) -> Dict[str, float]:
        """Get current GPU memory statistics."""
        if not torch.cuda.is_available():
            return {"peak_mb": 0, "allocated_mb": 0}
        
        return {
            "peak_mb": torch.cuda.max_memory_allocated() / (1024 * 1024),
            "allocated_mb": torch.cuda.memory_allocated() / (1024 * 1024),
        }
    
    def _reset_memory_stats(self):
        """Reset GPU memory statistics."""
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()
            gc.collect()
    
    def benchmark_callable(
        self,
        name: str,
        callable_fn: Callable,
        sample_input: Dict[str, torch.Tensor],
        backend: str = "pytorch",
        precision: str = "fp16",
    ) -> BenchmarkResult:
        """Benchmark a callable function."""
        logger.info(f"Benchmarking {name} ({backend})...")
        
        # Get input shape for recording
        first_input = list(sample_input.values())[0]
        input_shape = list(first_input.shape)
        
        # Reset memory stats
        self._reset_memory_stats()
        
        # Warmup
        logger.debug(f"Running {self.warmup_iterations} warmup iterations...")
        for _ in range(self.warmup_iterations):
            with torch.inference_mode():
                _ = callable_fn(**sample_input)
        
        torch.cuda.synchronize()
        self._reset_memory_stats()
        
        # Benchmark
        latencies = []
        logger.debug(f"Running {self.benchmark_iterations} benchmark iterations...")
        
        for i in range(self.benchmark_iterations):
            torch.cuda.synchronize()
            start = time.perf_counter()
            
            with torch.inference_mode():
                _ = callable_fn(**sample_input)
            
            torch.cuda.synchronize()
            end = time.perf_counter()
            
            latencies.append((end - start) * 1000)  # Convert to ms
        
        # Compute statistics
        latencies_sorted = sorted(latencies)
        mean_latency = sum(latencies) / len(latencies)
        std_latency = (sum((x - mean_latency) ** 2 for x in latencies) / len(latencies)) ** 0.5
        
        memory_stats = self._get_memory_stats()
        
        result = BenchmarkResult(
            name=name,
            backend=backend,
            iterations=self.benchmark_iterations,
            warmup_iterations=self.warmup_iterations,
            mean_latency_ms=mean_latency,
            std_latency_ms=std_latency,
            min_latency_ms=min(latencies),
            max_latency_ms=max(latencies),
            p50_latency_ms=latencies_sorted[len(latencies) // 2],
            p95_latency_ms=latencies_sorted[int(len(latencies) * 0.95)],
            p99_latency_ms=latencies_sorted[int(len(latencies) * 0.99)],
            peak_memory_mb=memory_stats["peak_mb"],
            allocated_memory_mb=memory_stats["allocated_mb"],
            throughput_per_sec=1000.0 / mean_latency,
            input_shape=input_shape,
            precision=precision,
            timestamp=datetime.now().isoformat(),
        )
        
        self._results.append(result)
        logger.info(f"  {backend}: {mean_latency:.2f}ms ± {std_latency:.2f}ms, "
                   f"peak memory: {memory_stats['peak_mb']:.1f}MB")
        
        return result
    
    def compare_backends(
        self,
        name: str,
        pytorch_fn: Callable,
        tensorrt_fn: Callable,
        sample_input: Dict[str, torch.Tensor],
        precision: str = "fp16",
    ) -> ComparisonResult:
        """Compare PyTorch and TensorRT backends."""
        logger.info(f"\n{'='*60}")
        logger.info(f"Comparing backends for: {name}")
        logger.info(f"{'='*60}")
        
        # Benchmark PyTorch
        pytorch_result = self.benchmark_callable(
            name=name,
            callable_fn=pytorch_fn,
            sample_input=sample_input,
            backend="pytorch",
            precision=precision,
        )
        
        # Benchmark TensorRT
        tensorrt_result = self.benchmark_callable(
            name=name,
            callable_fn=tensorrt_fn,
            sample_input=sample_input,
            backend="tensorrt",
            precision=precision,
        )
        
        # Compute comparison metrics
        speedup = pytorch_result.mean_latency_ms / tensorrt_result.mean_latency_ms
        memory_reduction = (pytorch_result.peak_memory_mb - tensorrt_result.peak_memory_mb) / pytorch_result.peak_memory_mb * 100
        latency_improvement = (pytorch_result.mean_latency_ms - tensorrt_result.mean_latency_ms) / pytorch_result.mean_latency_ms * 100
        
        comparison = ComparisonResult(
            name=name,
            pytorch=pytorch_result,
            tensorrt=tensorrt_result,
            speedup=speedup,
            memory_reduction=memory_reduction,
            latency_improvement_percent=latency_improvement,
        )
        
        logger.info(f"\nResults for {name}:")
        logger.info(f"  Speedup: {speedup:.2f}x")
        logger.info(f"  Latency improvement: {latency_improvement:.1f}%")
        logger.info(f"  Memory reduction: {memory_reduction:.1f}%")
        
        return comparison
    
    def save_results(self, filename: Optional[str] = None):
        """Save benchmark results to JSON file."""
        if filename is None:
            filename = f"benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        output_path = self.output_dir / filename
        
        results_dict = {
            "results": [r.to_dict() for r in self._results],
            "config": {
                "warmup_iterations": self.warmup_iterations,
                "benchmark_iterations": self.benchmark_iterations,
            },
            "timestamp": datetime.now().isoformat(),
        }
        
        with open(output_path, 'w') as f:
            json.dump(results_dict, f, indent=2)
        
        logger.info(f"Saved benchmark results to {output_path}")
        return output_path
    
    def print_summary(self):
        """Print summary of all benchmark results."""
        if not self._results:
            logger.info("No benchmark results to summarize")
            return
        
        print("\n" + "="*80)
        print("BENCHMARK SUMMARY")
        print("="*80)
        
        # Group by name
        by_name: Dict[str, List[BenchmarkResult]] = {}
        for result in self._results:
            if result.name not in by_name:
                by_name[result.name] = []
            by_name[result.name].append(result)
        
        for name, results in by_name.items():
            print(f"\n{name}:")
            for r in results:
                print(f"  {r.backend:12s}: {r.mean_latency_ms:8.2f}ms ± {r.std_latency_ms:6.2f}ms "
                      f"(p95: {r.p95_latency_ms:8.2f}ms) | Memory: {r.peak_memory_mb:8.1f}MB")


def create_sample_inputs(
    batch_size: int = 1,
    height: int = 1024,
    width: int = 1024,
    device: str = "cuda",
    dtype: torch.dtype = torch.float16,
) -> Dict[str, torch.Tensor]:
    """Create sample inputs for SDXL UNet benchmarking."""
    latent_height = height // 8
    latent_width = width // 8
    
    return {
        "sample": torch.randn(batch_size, 4, latent_height, latent_width, device=device, dtype=dtype),
        "timestep": torch.tensor([500], device=device, dtype=torch.int64),
        "encoder_hidden_states": torch.randn(batch_size, 77, 2048, device=device, dtype=dtype),
    }

def run_quick_benchmark(
    pytorch_model: torch.nn.Module,
    tensorrt_model: Any,
    name: str = "model",
    iterations: int = 10,
) -> ComparisonResult:
    """Quick benchmark comparison helper."""
    benchmark = TensorRTBenchmark(
        warmup_iterations=3,
        benchmark_iterations=iterations,
    )
    
    sample_input = create_sample_inputs()
    
    return benchmark.compare_backends(
        name=name,
        pytorch_fn=pytorch_model,
        tensorrt_fn=tensorrt_model,
        sample_input=sample_input,
    )
