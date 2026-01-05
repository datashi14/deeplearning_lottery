# GCP Environment Setup Runbook

This guide contains the exact commands to provision the accessible, GPU-enabled environment for TicketSmith on Google Cloud Platform (GCP).

## 1. Prerequisites

Ensure you have the Google Cloud CLI (`gcloud`) installed and authenticated.

````bash
```bash
# Login and set project
gcloud auth login
gcloud config set project propane-net-247501
gcloud config set compute/region us-central1
gcloud config set compute/zone us-central1-a
````

````

## 2. Artifact Registry & Storage

We replace ACR and Blob Storage with Artifact Registry and GCS.

```bash
# 1. Create Artifact Registry (Docker)
gcloud artifacts repositories create ticketsmith \
    --repository-format=docker \
    --location=us-central1 \
    --description="TicketSmith container images"

# 2. Create GCS Bucket for Artifacts
# Note: Bucket names must be globally unique. Append a random suffix or your project ID.
gsutil mb -l us-central1 gs://ticketsmith-runs-$(gcloud config get-value project)
````

## 3. GKE Cluster Provisioning

We use a **GKE Standard** cluster to have full control over GPU drivers and node pool scaling logic.

### 3.1 Create Control Plane & System Pool

The system pool runs core Kubernetes services (DNS, metrics) without consuming expensive GPU resources.

```bash
gcloud container clusters create ticketsmith \
    --zone us-central1-a \
    --machine-type e2-standard-4 \
    --num-nodes 1 \
    --scopes cloud-platform
```

### 3.2 Add Scale-to-Zero GPU Pool

We use **L4 GPUs** (NVIDIA Ada Lovelace) for the best price/performance ratio for inference/training validation.

- **Machine Type**: `g2-standard-8` (4 vCPUs, 1 L4 GPU, 32GB RAM)
- **Autoscaling**: 0 to 1 nodes.

```bash
gcloud container node-pools create gpu-pool \
    --cluster ticketsmith \
    --zone us-central1-a \
    --machine-type g2-standard-8 \
    --accelerator type=nvidia-l4,count=1 \
    --num-nodes 0 \
    --enable-autoscaling \
    --min-nodes 0 \
    --max-nodes 1
```

_Note: If L4 quota is unavailable, swap `--machine-type` to `n1-standard-4` and `--accelerator` to `type=nvidia-tesla-t4,count=1`._

### 3.3 Configure kubectl

```bash
gcloud container clusters get-credentials ticketsmith --zone us-central1-a
```

### 3.4 Install NVIDIA Drivers

GKE requires a daemonset to expose the drivers to containers on COS (Container-Optimized OS).

```bash
kubectl apply -f https://raw.githubusercontent.com/GoogleCloudPlatform/container-engine-accelerators/master/nvidia-driver-installer/cos/daemonset-preloaded.yaml
```

## 4. Setup Secrets (GCS Access)

To allow our jobs to write to GCS without hardcoded keys, we create a Kubernetes Service Account (KSA) bound to a Google Service Account (GSA) via **Workload Identity**.

```bash
# 1. Create Google Service Account (GSA)
gcloud iam service-accounts create ticketsmith-sa

# 2. Grant Permissions to GSA
PROJECT_ID=$(gcloud config get-value project)
gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member "serviceAccount:ticketsmith-sa@$PROJECT_ID.iam.gserviceaccount.com" \
    --role "roles/storage.objectAdmin"

# 3. Enable Workload Identity on GKE (if not already)
# (Already enabled by default on new Autopilot/Standard clusters usually, but good to check)
gcloud container clusters update ticketsmith \
    --zone us-central1-a \
    --workload-pool=$PROJECT_ID.svc.id.goog

# 4. Create Kubernetes Namespace & Service Account
kubectl create namespace ticketsmith
kubectl create serviceaccount ticketsmith-ksa --namespace ticketsmith

# 5. Bind KSA to GSA
gcloud iam service-accounts add-iam-policy-binding ticketsmith-sa@$PROJECT_ID.iam.gserviceaccount.com \
    --role roles/iam.workloadIdentityUser \
    --member "serviceAccount:$PROJECT_ID.svc.id.goog[ticketsmith/ticketsmith-ksa]"

# 6. Annotate KSA
kubectl annotate serviceaccount ticketsmith-ksa \
    --namespace ticketsmith \
    iam.gke.io/gcp-service-account=ticketsmith-sa@$PROJECT_ID.iam.gserviceaccount.com
```

## 5. Verification

Run the GPU smoke test to verify autoscaling and driver installation.

```bash
kubectl apply -f k8s/gcp/gpu-smoke.yaml
kubectl -n ticketsmith logs -f job/gpu-smoke
```

## 6. Cleanup

To pause costs without deleting the cluster data:

```bash
gcloud container node-pools delete gpu-pool --cluster ticketsmith --zone us-central1-a
gcloud container clusters delete ticketsmith --zone us-central1-a
```
