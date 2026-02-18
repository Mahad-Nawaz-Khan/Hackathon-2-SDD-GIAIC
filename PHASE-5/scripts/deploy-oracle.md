# Oracle OKE Deployment Guide

## Prerequisites (What YOU need to do)

### 1. Oracle Cloud Account Setup
1. Go to https://cloud.oracle.com and sign up/login
2. You should have a Free Tier account (Always Free + 30-day trial)

### 2. Create OKE Cluster
1. Navigate to **Developer Services → Kubernetes Clusters (OKE)**
2. Click **Create Cluster**
3. Choose **Quick Create** (easiest option)
4. Configure:
   - **Name**: `todo-chatbot-cluster`
   - **Compartment**: Select your root compartment or create one named `todo-app`
   - **Kubernetes Version**: Select the latest stable (1.28 or higher)
   - **Node Shape**: `VM.Standard.E4.Flex` (ARM - Free Tier eligible) or `VM.Standard2.1` (x86)
   - **Number of Nodes**: `1` (Free Tier) or `3` (Production)
   - **OCPU**: `1`
   - **Memory**: `6GB` or more
5. Click **Create** and wait ~10-15 minutes

### 3. Install OCI CLI
```powershell
# Windows PowerShell
Set-ExecutionPolicy Bypass -Scope Process -Force; Invoke-Expression ((New-Object System.Net.WebClient).DownloadString('https://raw.githubusercontent.com/oracle/oci-cli/master/scripts/install/install.ps1'))
```

### 4. Configure OCI CLI
```bash
oci setup config
# Follow prompts:
# - Enter your OCID (from Oracle Cloud Console → Profile → User Settings)
# - Enter your Tenancy OCID (from Console → Administration → Tenancy)
# - Choose region (e.g., ap-mumbai-1, us-phoenix-1)
# - Generate new RSA key pair
```

### 5. Get kubeconfig for OKE
```bash
oci ce cluster create-kubeconfig --cluster-id <CLUSTER_OCID> --file $HOME/.kube/config --region <YOUR_REGION>
```

### 6. Create Container Registry (OCIR)
1. Go to **Developer Services → Container Registry**
2. It's auto-created, but note your namespace:
   ```bash
   oci os namespace get
   ```
3. Your image URLs will be: `<region>.ocir.io/<tenancy-namespace>/todo-chatbot-backend:latest`

### 7. Create Auth Token for Docker Login
1. Go to **Identity → Users → Your User**
2. Click **Auth Tokens** → **Generate Token**
3. **SAVE THIS TOKEN** - you won't see it again!
4. Use it as your Docker password

---

## Deployment Steps (Scripts will handle)

### Step 1: Login to OCIR
```bash
docker login <region>.ocir.io
# Username: <tenancy-namespace>/your-email@example.com
# Password: <auth-token>
```

### Step 2: Build and Push Images
```bash
# Run the deploy script
./scripts/build-push-oracle.sh
```

### Step 3: Deploy to OKE
```bash
./scripts/deploy-to-oke.sh
```

---

## What Gets Deployed

| Component | Oracle Cloud Service |
|-----------|---------------------|
| Kubernetes Cluster | OKE (Oracle Kubernetes Engine) |
| Container Registry | OCIR (Oracle Container Image Registry) |
| Database | Neon PostgreSQL (external - already configured) |
| Kafka | Strimzi (self-hosted on OKE) |
| Redis | Bitnami Redis (self-hosted on OKE) |
| Dapr | Self-hosted on OKE |
| Ingress | NGINX Ingress Controller |
| TLS | Cert-Manager + Let's Encrypt |
| Secrets | Kubernetes Secrets |

---

## Estimated Costs (Free Tier)

| Resource | Free Tier Limit |
|----------|----------------|
| OKE Cluster | 1 cluster free |
| Compute (ARM) | 4 OCPUs + 24GB RAM/month |
| Storage | 200GB total |
| Bandwidth | 10TB/month outbound |

**Your setup should be FREE if using ARM nodes within limits.**

---

## After Deployment

1. Get your Load Balancer IP:
   ```bash
   kubectl get svc -n ingress-nginx
   ```

2. Configure your domain DNS:
   - `todo.todo-app.cloud` → Load Balancer IP
   - `api.todo-app.cloud` → Load Balancer IP

3. TLS certificates auto-provision via Cert-Manager
