import torch
import time
from transformers import AutoTokenizer, AutoModelForCausalLM
import argparse
from tqdm import tqdm

def load_model_for_bench(model_id, device="cuda"):
    print(f"Loading {model_id} for Benchmarking...")
    # Load in BF16 (matching our training stack)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map={"": 0},
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer

def benchmark_inference(model, tokenizer, prompt, max_new_tokens=100, num_runs=5):
    device = model.device
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    # Warmup
    print("🔥 Warming up GPU...")
    for _ in range(2):
        with torch.no_grad():
            model.generate(**inputs, max_new_tokens=10)
    
    times = []
    tokens_generated = []
    
    print(f"🏎️ Running {num_runs} inference cycles...")
    for _ in tqdm(range(num_runs)):
        # Sync CUDA before timing
        torch.cuda.synchronize()
        start_time = time.perf_counter()
        
        with torch.no_grad():
            output = model.generate(
                **inputs, 
                max_new_tokens=max_new_tokens,
                do_sample=False, # Deterministic for benchmarking
                use_cache=True
            )
        
        torch.cuda.synchronize()
        end_time = time.perf_counter()
        
        times.append(end_time - start_time)
        tokens_generated.append(output.shape[1] - inputs.input_ids.shape[1])
    
    avg_time = sum(times) / num_runs
    avg_tokens = sum(tokens_generated) / num_runs
    tps = avg_tokens / avg_time
    
    return tps, avg_time

def main():
    parser = argparse.ArgumentParser(description="TicketSmith Throughput Benchmark")
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.2-1B-Instruct")
    parser.add_argument("--sparsity", type=float, default=0.0, help="Simulate speed of structured sparsity")
    args = parser.parse_args()
    
    model, tokenizer = load_model_for_bench(args.model)
    
    # Standard Benchmark
    prompt = "Explain the concept of neural network pruning in two sentences."
    tps, avg_time = benchmark_inference(model, tokenizer, prompt)
    
    print(f"\n--- Benchmark Results ({args.model}) ---")
    print(f"Mean Inference Time: {avg_time:.3f}s")
    print(f"Throughput:          {tps:.2f} tokens/sec")
    print(f"VRAM Usage:          {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    
    if args.sparsity > 0:
        print(f"\n📈 Calculating Theoretical Sparse Gain ({args.sparsity*100}% sparsity)...")
        # In a real Winning Ticket, structured pruning (removing rows/cols)
        # leads to direct speedups. Unstructured (masking) does not.
        theoretical_tps = tps / (1 - args.sparsity)
        print(f"Theoretical Sparse Throughput: {theoretical_tps:.2f} tokens/sec")
        print(f"Potential Speedup:             {theoretical_tps/tps:.1f}x")

if __name__ == "__main__":
    main()
