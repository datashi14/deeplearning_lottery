# Check if az is installed
if (-not (Get-Command "az" -ErrorAction SilentlyContinue)) {
  # Try adding standard path
  $AzPath = "C:\Program Files\Microsoft SDKs\Azure\CLI2\wbin"
  if (Test-Path $AzPath) {
    Write-Host "Found Azure CLI in standard path, adding to session..."
    $env:Path += ";$AzPath"
  }
}

if (-not (Get-Command "az" -ErrorAction SilentlyContinue)) {
  Write-Error "Azure CLI (az) command not found. If you just installed it, please RESTART YOUR TERMINAL (or VS Code) to update the PATH."
  exit 1
}

# Check login status
try {
  az account show --output none
}
catch {
  Write-Error "You are NOT logged in to Azure. Please run 'az login' in your terminal first, then re-run this script."
  exit 1
}

# 1. Connect kubectl
Write-Host "Connecting to AKS..."
az aks get-credentials -g aks-ticketsmith_group -n aks-ticketsmith --overwrite-existing
kubectl get nodes -o wide

# 2. ACR Setup
Write-Host "Creating ACR..."
az acr create -g aks-ticketsmith_group -n acrticketsmith --sku Basic
az aks update -g aks-ticketsmith_group -n aks-ticketsmith --attach-acr acrticketsmith

# 3. GPU Pool
Write-Host "Adding GPU Pool..."
az aks nodepool add `
  --resource-group aks-ticketsmith_group `
  --cluster-name aks-ticketsmith `
  --name gpu `
  --node-vm-size Standard_NC4as_T4_v3 `
  --enable-cluster-autoscaler `
  --min-count 0 `
  --max-count 1 `
  --node-count 0 `
  --node-taints sku=gpu:NoSchedule `
  --labels agentpool=gpu sku=gpu

# 4. NVIDIA Plugin
Write-Host "Installing NVIDIA Device Plugin..."
# Using the static deployment from main branch which is stable location
kubectl apply -f https://raw.githubusercontent.com/NVIDIA/k8s-device-plugin/main/deployments/static/nvidia-device-plugin.yml

# 5. Blob Storage
Write-Host "Setting up Blob Storage..."
$Random = Get-Random -Minimum 1000 -Maximum 9999
$StorageName = "ticketsmith$Random" # Shortened to keep under 24 char limit
if ($StorageName.Length -gt 24) {
  $StorageName = $StorageName.Substring(0, 24)
}
az storage account create -g aks-ticketsmith_group -n $StorageName -l eastus --sku Standard_LRS

$ConnString = az storage account show-connection-string -g aks-ticketsmith_group -n $StorageName --query connectionString -o tsv
az storage container create --account-name $StorageName --name ticketsmith-runs --connection-string $ConnString

# Secrets
Write-Host "Creating K8s Secrets..."
kubectl create namespace ticketsmith --dry-run=client -o yaml | kubectl apply -f -
kubectl -n ticketsmith create secret generic blob-secret `
  --from-literal=AZURE_STORAGE_CONNECTION_STRING="$ConnString" `
  --from-literal=AZURE_BLOB_CONTAINER="ticketsmith-runs" `
  --dry-run=client -o yaml | kubectl apply -f -

Write-Host "Setup Complete!"
