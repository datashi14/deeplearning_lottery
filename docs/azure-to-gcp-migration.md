# Azure to GCP Migration: Engineering Record

## 1. Executive Summary

TicketSmith was initially deployed to Azure AKS. During validation, GPU provisioning was blocked by subscription-level quota restrictions in the target region (East US) for modern GPU VM families.
We completed CPU validation on AKS to prove the platform end-to-end, then migrated to GCP (GKE + Artifact Registry + GCS) to run the full pipeline on a real GPU with scale-to-zero cost controls.

### Outcome

- ✅ **Platform validated on AKS (CPU)**: container runtime, job orchestration, artifact pipeline, reporting.
- ✅ **GPU execution enabled on GKE**: reproducible runs with real serving benchmarks on supported GPU SKUs.
- ✅ **CI/CD established**: versioned images, traceable deployments, repeatable experiments.

## 2. Background and Goals

### Goals

- Run TicketSmith jobs on GPU with scale-to-zero node pools.
- Preserve reproducibility: immutable configs, run IDs, config hashes, artifact storage.
- Make infrastructure portable: minimal cloud-specific logic in code.
- Establish CI/CD so builds and runs are audit-friendly.

### Non-goals

- Multi-node distributed training
- Production inference service
- Full observability stack (Grafana/Prometheus dashboards)

## 3. Azure AKS Implementation (What We Built)

### Architecture on Azure

- **Compute**: AKS cluster with system pool and planned GPU pool
- **Registry**: ACR (`acrticketsmith`)
- **Storage**: Azure Blob (`ticketsmith5541`) for artifacts
- **Workloads**: Kubernetes Jobs (train/prune/benchmark/report)
- **Cost Controls**: autoscaler, `activeDeadlineSeconds`, scale-to-zero GPU pool design

### What worked

- ACR image pulls succeeded (validated with `image-pull-test`)
- CPU runner pool successfully executed dense training
- Training logs confirmed deterministic run metadata:
  - run ID
  - config hash
  - device selection
- Artifacts were generated locally per run directory

### Evidence (links/log excerpts)

`kubectl` logs showing training completion and run path:

```text
Initialized Run: ...
Config Hash: ...
Training complete. Artifacts saved to runs/...
```

## 4. Azure GPU Blocker (Root Cause)

### Symptom

GPU jobs remained `Pending` with autoscaler not providing nodes.

### Root Cause

Quota restrictions in East US:

- `Standard NCASv3_T4 Family vCPUs`: limit 0 (T4 blocked)
- `Standard NCSv3 Family vCPUs`: limit 0 (V100 blocked)
- Legacy `Standard NC Family vCPUs`: quota existed, but AKS could not provision supported SKUs:
  - `VMSizeNotSupported` for `Standard_NC6` in AKS nodepool creation

### Why waiting did not help

This was not an autoscaler cold start. Azure could not allocate the requested VM family under the subscription/region constraints.

### Decision

Proceed with **CPU validation** on AKS to complete end-to-end functional verification while avoiding stalled delivery.

## 5. CPU Validation on AKS (Deliberate Strategy)

### Why CPU validation was still valuable

CPU runs validated the platform architecture independent of hardware:

- training loop correctness
- artifact generation and storage wiring
- reproducibility (config hashing)
- job-based execution pattern
- reporting output

### Constraints

- Small node sizes (`Standard_A2_v2`) required tight memory requests.
- Large "universal" image size increased cold start time, but ensured parity with GPU image.

### Results

Dense baseline ran successfully:

- Epoch logs show stable training and high MNIST accuracy (~99% test accuracy)
- Run folder created with deterministic ID
- Established that core execution path works in-cloud

## 6. Migration Decision: Why GCP

### What we needed

- Real GPU availability with quota we can access
- Clean autoscaling GPU nodepool with scale-to-zero
- Simple artifact storage and image registry integration
- CI/CD that doesn't require long-lived credentials

### Why GCP fit

- Better access to modern GPU SKUs (L4/A100) depending on region/project
- GKE + Artifact Registry + GCS have straightforward integration
- GitHub Actions + Workload Identity Federation avoids static keys

## 7. GCP Target Architecture

### Services

- **GKE** (Kubernetes)
- **Artifact Registry** (container images)
- **GCS** (artifact persistence)
- **Workload Identity Federation** (GitHub Actions auth)
- _Optional: Cloud Build (if needed)_

### Mapping: Azure → GCP

| Capability         | Azure                         | GCP                |
| :----------------- | :---------------------------- | :----------------- |
| Kubernetes         | AKS                           | GKE                |
| Container registry | ACR                           | Artifact Registry  |
| Artifact storage   | Blob Storage                  | GCS                |
| Auth (CI)          | Managed identity / ACR attach | WIF (OIDC)         |
| GPU node pool      | NC\* family                   | L4/A100 pool       |
| Scale to zero      | cluster autoscaler            | cluster autoscaler |

## 8. CI/CD Pipeline (First-Class Requirement)

### Why CI/CD first

Migration only counts if it is reproducible:

- every image is versioned
- every run can be linked to a commit
- "works on my machine" is eliminated

### CI stages

1.  Build Docker image
2.  Tag with git SHA + `latest`
3.  Push to Artifact Registry
4.  _(Manual dispatch)_ submit GKE Jobs for smoke test / training

### Auth approach

GitHub Actions uses OIDC → Workload Identity Pool → impersonates GCP SA.
**No service account keys stored in GitHub.**

## 9. GKE GPU Execution Validation

### Validation sequence

1.  GPU smoke test (`nvidia-smi`)
2.  Dense training job
3.  Artifact upload verification (GCS listing)
4.  Pruning run + report generation
5.  Benchmark table produced on GPU

### Success definition

- GPU node scales up for job and scales down after completion
- Artifacts and executive summary exist in GCS
- Serving scorecard contains real GPU metrics

## 10. Lessons Learned

- Quota and SKU availability are first-order design constraints for ML platforms.
- Job-based execution with strict timeouts prevents runaway cost.
- Keeping the code hardware-agnostic enabled a clean cloud migration.
- A universal image increases cold-start but reduces environment drift.
- CI/CD should be the starting point, not the end.

## 11. How to Reproduce (High Level)

1.  `make ci-build` (or GitHub Actions run)
2.  `kubectl apply -f k8s/gke/gpu-smoke.yaml`
3.  `kubectl apply -f k8s/gke/train-dense.yaml`
4.  verify artifacts in `gs://ticketsmith-runs/...`

_(Exact commands in `docs/gcp-setup.md` and `.github/workflows/`.)_

## 12. Appendix: Evidence and References

- Quota output snippets (Azure)
- Error message: `VMSizeNotSupported`
- Links to run logs
- Sample executive summary
