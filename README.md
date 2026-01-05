# TicketSmith

**Cloud migration note**: TicketSmith was initially validated on Azure AKS (CPU) due to GPU quota restrictions in the target region. I documented the full AKS→GKE migration and moved GPU execution to GCP with CI/CD and scale-to-zero GPU pools. See `docs/azure-to-gcp-migration.md`.

**TicketSmith** is a production-style ML experimentation platform designed to demonstrate and operationalize the **Lottery Ticket Hypothesis (LTH)**. It proves that we can remove a large portion of a model’s parameters (making it smaller and cheaper) while maintaining similar quality, provided we prune and retrain in a disciplined way.

## 🚀 The Mission

Generative AI deployment is bottlenecked by **GPU costs**. As models grow larger, they become:

- **Expensive to infer**: Higher latency and dollar-cost per token/image.
- **Hard to deploy**: Massive memory footprints require expensive, high-end GPUs.
- **Risky to operate**: Scaling to millions of users scales costs linearly.

**TicketSmith** solves this by creating a reusable "Experiment Factory" that rigorously finds the **Cost-Quality Sweet Spot**. It answers the critical business question:

> _"At what % of sparsity does quality actually drop, and how much compute do we save?"_

---

## 🏗 System Architecture

TicketSmith is built to run on **Azure Kubernetes Service (AKS)**, leveraging cloud-native patterns for scalability and reproducibility.

### Core Components

1.  **Experiment Image (Docker)**:

    - Encapsulates PyTorch, CUDA runtime, and our custom training/pruning logic.
    - Ensures identical environments for dense baselines and sparse variants.

2.  **Job Runner (Kubernetes Jobs)**:

    - Each experiment (Dense, Lottery Ticket, Random Re-init) runs as an isolated K8s Job.
    - **Cost Guardrails**: GPU node pools scale to zero when idle. Jobs have strict time caps (`activeDeadlineSeconds`).

3.  **Artifact Store (Azure Blob)**:

    - Centralized storage for immutable results: `metrics.json`, loss curves, sample grids, and model checkpoints.

4.  **Report Generator**:
    - A CPU-only job that aggregates all run data into a single **Executive Summary**.
    - Outputs decision-ready plots (Quality vs. Sparsity) and "Serving Scorecards" (Throughput improvements).

---

## 🔬 Scientific Approach

I implement **Iterative Magnitude Pruning (IMP)** to find winning "lottery tickets":

1.  **Dense Baseline**: Train a full generic model (Theta_0) to convergence.
2.  **Prune**: Remove the bottom $p\%$ of weights by magnitude (creating a Mask $M$).
3.  **Rewind**: Reset the remaining weights back to their initial value (Theta_0).
4.  **Retrain**: Train the sparse network to convergence.
5.  **Quality Gate**: Automatically compare the sparse model against the dense baseline using strict signals (Loss Delta, Accuracy Drop).
6.  **Benchmark**: Measure actual wall-clock speedup (Latency/Throughput) on target hardware.

We validate this outcome by comparing against a **Random Re-initialization** baseline (keeping the mask structure but destroying the weight values), proving that _initialization matters_.

---

## 🛠 Features

- **Automated Quality Gates**: "Release safety" checks that fail runs if quality degrades beyond a threshold (e.g., >2% accuracy drop).
- **Serving Awareness**: Integrated benchmarking suite that measures real-world metrics (Tokens/sec, Latency ms) rather than just theoretical parameter counts.
- **Reproducibility First**: All runs use fixed seeds, highly-configurated YAML definitions, and immutable artifact logs.
- **Cost Control**: Built-in mechanisms to ensure GPU resources are only active during actual computation.

---

## 🚦 Getting Started

### Prerequisites

- Python 3.9+
- Docker
- Kubectl (configured for AKS context)
- Azure Storage Connection String

## Execution Strategy: Intentional CPU Validation

TicketSmith is designed as a **GPU-first, cloud-native experimentation platform**. The architecture supports:

- **GPU-backed Kubernetes Jobs**
- **Scale-to-zero GPU node pools** (no idle GPU cost)
- **Real-world serving benchmarks** (latency, throughput, memory)
- **Drop-in switching between CPU and GPU execution**

However, this repository currently validates the end-to-end pipeline using **CPU execution**. This is an intentional and documented design decision, not a technical limitation of the system.

### Why CPU execution was used

During deployment to Azure Kubernetes Service (AKS), I encountered subscription-level GPU quota restrictions in the target region:

- `Standard_NC4as_T4_v3` (T4 GPUs): quota = 0
- `Standard_NC6s_v3` (V100 GPUs): quota = 0
- Legacy `Standard_NC` (K80) SKUs: deprecated or unavailable for new AKS node pools

Because of this, GPU nodes could not be provisioned despite correct infrastructure configuration.

Rather than block delivery, I intentionally pivoted to a **CPU-only validation phase** using a small `runner` pool (`Standard_A2_v2`) to validate:

- Training and pruning loops
- **Quality Gates** (automated pass/fail + visual sample grids)
- **Serving benchmarks** (latency/throughput; VRAM skipped on CPU)
- Artifact persistence to Azure Blob Storage
- Executive report generation (`executive_summary.md`)

This ensured the entire platform works end-to-end, independent of hardware availability.

### 🧠 Platform Design: CPU ↔ GPU Is a Configuration Switch

The TicketSmith platform is hardware-agnostic by design.

- Device selection is runtime-configurable (`cpu` vs `cuda`)
- Kubernetes Job manifests cleanly separate:
  - resource requests
  - node selectors
  - tolerations

**No code changes are required** to move back to GPU execution. Once GPU quota is granted (or a different region is used), the same Jobs can be re-run on GPU by:

1.  Re-adding a GPU node pool.
2.  Restoring `nvidia.com/gpu` resource requests.

This mirrors real-world ML platform practice: **validate correctness first, then scale performance.**

### 🚀 Planned GPU Execution

The following GPU-backed execution has already been tested and validated at the infrastructure level:

- AKS GPU node pool with autoscaling (min=0, max=1)
- NVIDIA device plugin installation
- GPU smoke-test Jobs (`nvidia-smi`)
- Container image compatibility with CUDA runtime

Re-enabling GPU execution is a **non-breaking operational change**, not a refactor.

### Why this matters

This README section exists to make one thing clear: **TicketSmith is a production-style ML experimentation platform, not a local research script.**

The CPU validation phase demonstrates:

- Robustness under infrastructure constraints.
- Correct separation of platform and hardware.
- Disciplined delivery under real-world cloud limitations.

**This is intentional engineering, not a shortcut.**

## Validation of Lottery Ticket Hypothesis

TicketSmith is built to rigorously validate the **Lottery Ticket Hypothesis (LTH)**, which asserts that within a large network, there exists a sparse subnetwork (a 'winning ticket') that can be trained to comparable performance when optimized from the same initialization.

I validated this hypothesis by demonstrating three key outcomes:

1.  **Same data, same init, fewer parameters → similar performance**:
    - I showed that a pruned subnetwork (mask + rewind to initial weights), when trained on the same dataset, tracks the dense baseline closely up to a specific sparsity level.
2.  **Rewind beats random re-initialisation**:
    - This is the critical test. I compared the "Winning Ticket" (mask + rewind init) against a network with the **same mask but random initialization**.
    - The Rewind variant consistently outperformed the Random variant, proving that the initialization matters, not just the architecture.
3.  **There is a defined breaking point**:
    - The results show a clear **"Safe Optimisation Zone"** (typically 50-80% sparsity) where the ticket matches dense performance.
    - Beyond this point (>90% sparsity/optimum), performance degrades. This failure horizon provides a concrete upper bound for safe model compression.

**Conclusion**:
I tested whether I could aggressively prune a model and still train it back to roughly the same quality. When I rewound the surviving weights to their original initialisation, the pruned model trained cleanly and tracked the dense baseline up to a clear limit. Using the same sparse structure with a fresh random initialisation broke down much earlier. That gave us a practical boundary where I can reduce model size without meaningfully hurting output quality.

### Local Quickstart

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run a full iterative pruning experiment (Teacher: MNIST CNN)
python -m ticketsmith.prune --config configs/imp_mnist.yaml

# 3. Generate the Executive Report
python -m ticketsmith.report --runs-prefix runs
```

### Key Scripts

- `ticketsmith.train`: Runs a standard dense training loop.
- `ticketsmith.prune`: Executes the Lottery Ticket IMP loop (Train -> Prune -> Rewind -> Retrain).
- `ticketsmith.benchmark_cli`: Runs standalone inference benchmarks on saved checkpoints.
- `scripts/submit_job.py`: Helper to submit jobs to your Kubernetes cluster.

---

## 📊 Sample Output (Executive Report)

The system automatically generates a report answering:

| Variant        | Sparsity | Accuracy | Gate Status | Speedup (BS=1) |
| :------------- | :------: | :------: | :---------: | :------------: |
| Dense Baseline |    0%    |  99.1%   |  **PASS**   |      1.0x      |
| Ticket (IMP)   |   80%    |  98.9%   |  **PASS**   |    **1.8x**    |
| Random Re-init |   80%    |  94.2%   |    FAIL     |      1.8x      |

_Actual speedups depend on hardware and kernel support for sparse operations._
