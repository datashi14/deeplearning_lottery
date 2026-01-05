import time
import torch
import numpy as np
import json
import os
import psutil

class Benchmarker:
    def __init__(self, model, device, batch_sizes=[1, 4, 8], warmup_iters=10, measure_iters=50):
        self.model = model
        self.device = device
        self.batch_sizes = batch_sizes
        self.warmup_iters = warmup_iters
        self.measure_iters = measure_iters
        self.results = {}

    def benchmark(self, input_shape=(1, 28, 28)):
        self.model.eval()
        self.model.to(self.device)
        
        print(f"Starting Benchmark on {self.device}...")
        
        for bs in self.batch_sizes:
            # Prepare input
            input_tensor = torch.randn(bs, *input_shape).to(self.device)
            
            # Warmup
            with torch.no_grad():
                for _ in range(self.warmup_iters):
                    _ = self.model(input_tensor)
                    if self.device.type == 'cuda':
                        torch.cuda.synchronize()
            
            # Measurement
            latencies = []
            
            # Reset memory tracking
            if self.device.type == 'cuda':
                torch.cuda.reset_peak_memory_stats()
            
            with torch.no_grad():
                for _ in range(self.measure_iters):
                    start = time.perf_counter()
                    _ = self.model(input_tensor)
                    if self.device.type == 'cuda':
                        torch.cuda.synchronize()
                    end = time.perf_counter()
                    latencies.append((end - start) * 1000) # ms
            
            # Metrics
            mean_lat = np.mean(latencies)
            p95_lat = np.percentile(latencies, 95)
            throughput = bs / (mean_lat / 1000.0) # samples/sec
            
            memory_gb = 0.0
            if self.device.type == 'cuda':
                memory_gb = torch.cuda.max_memory_allocated() / (1024**3)
            else:
                 # CPU memory approximation (entire process)
                 process = psutil.Process(os.getpid())
                 memory_gb = process.memory_info().rss / (1024**3)

            self.results[bs] = {
                'mean_latency_ms': mean_lat,
                'p95_latency_ms': p95_lat,
                'throughput_img_s': throughput,
                'peak_memory_gb': memory_gb,
                'batch_size': bs
            }
            
            print(f"Batch {bs}: {mean_lat:.2f}ms/batch, {throughput:.2f} imgs/s, {memory_gb:.4f}GB RAM")
            
        # Serving Insight (Honesty Check)
        # Compare BS=1 throughput to some baseline? 
        # Actually this benchmarker doesn't know the baseline unless we pass it.
        # But we can store a heuristic or just the field to be filled later.
        # However, the requirement is "Benchmark MUST output ... serving_insight".
        # Let's add a placeholder or simple logic if we had baseline. 
        # Since we don't calculate relative speedup *here* (we do it in report), 
        # we will add the fields with 'N/A' or calculated if possible.
        # Let's assume the user checks the report for the comparison.
        # But let's add the note field.
        
        self.results['serving_insight'] = {
            'sparsity_speedup_observed': False, # Default to conservative
            'note': "Check Executive Report for relative speedup vs dense baseline. Unstructured pruning often requires sparse kernels to show speedup."
        }
            
        return self.results

    def save_results(self, path):
        with open(path, 'w') as f:
            json.dump(self.results, f, indent=2)

def run_benchmark(model, config, device, artifact_manager, variant_name="unknown"):
    """
    Helper to run benchmark and save artifacts using ArtifactManager.
    """
    bench_config = config.get('benchmark', {})
    if not bench_config.get('enabled', False):
        return None
        
    print(f"\n--- Running Benchmark for {variant_name} ---")
    
    bs_list = bench_config.get('batch_sizes', [1, 32, 64])
    warmup = bench_config.get('warmup_iters', 5)
    measure = bench_config.get('measure_iters', 20)
    
    # Infer input shape from model family (hardcoded for MVP)
    # MNIST
    input_shape = (1, 28, 28)
    
    bencher = Benchmarker(model, device, batch_sizes=bs_list, warmup_iters=warmup, measure_iters=measure)
    results = bencher.benchmark(input_shape)
    
    # Save detailed JSON
    filename = f"benchmark_{variant_name}.json"
    artifact_path = os.path.join(artifact_manager.run_dir, filename)
    bencher.save_results(artifact_path)
    
    # Upload if needed (ArtifactManager doesn't auto-upload arbitrary files unless we walk, 
    # but we will rely on valid 'metrics' logging or final upload scan)
    
    # Log summary to metrics
    # We flatten for "Serving Scorecard"
    # Format: benchmark_{variant}_{bs}_throughput
    for bs, res in results.items():
        artifact_manager.log_metric(f'bench_{variant_name}_bs{bs}_throughput', res['throughput_img_s'])
        artifact_manager.log_metric(f'bench_{variant_name}_bs{bs}_latency', res['mean_latency_ms'])
        
    return results
