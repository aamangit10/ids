# 🚀 Render Deployment Guide for Explainable SSL IDS

This repository is fully configured for deployment on **[Render](https://render.com/)**, allowing you to host both the **Frontend (React UI)** and **Backend (FastAPI + PyTorch/ML)** online. Once deployed, Render gives you a public HTTPS URL that you can open and run on any laptop, desktop, tablet, or phone worldwide.

---

## 📋 Prerequisites
1. **GitHub Repository**: Make sure your local code is pushed to your GitHub repo:
   `https://github.com/aamangit10/ids`
2. **Render Account**: Log into your Render dashboard:
   [https://dashboard.render.com](https://dashboard.render.com/)

---

## ⚡ Method 1: Automatic 1-Click Deployment (Render Blueprint) — *Recommended*

The repository contains an automated blueprint definition: [`render.yaml`](./render.yaml).

### Steps:
1. Open the [Render Dashboard](https://dashboard.render.com/).
2. Click the **"New +"** button in the top navigation bar.
3. Select **"Blueprint"**.
4. Connect your GitHub account and select repository: **`aamangit10/ids`**.
5. Render will automatically detect `render.yaml` and configure two services:
   - **`ids-backend`** (Web Service — Python 3.11, FastAPI, PyTorch CPU, Scikit-Learn)
   - **`ids-frontend`** (Static Site — Vite, React 18, Global CDN)
6. Click **"Apply"**.
7. Render will build and deploy both services automatically!
8. When deployment finishes, Render provides your live URLs:
   - **Frontend App**: `https://ids-frontend-xxxx.onrender.com`
   - **Backend API**: `https://ids-backend-xxxx.onrender.com`
   - **Swagger Docs**: `https://ids-backend-xxxx.onrender.com/docs`

---

## 🛠️ Method 2: Manual Step-by-Step Deployment on Render

If you prefer to configure each service manually in the Render UI:

### Step 1: Deploy Backend (Web Service)
1. On [Render Dashboard](https://dashboard.render.com/), click **New +** > **Web Service**.
2. Select your repository: `aamangit10/ids`.
3. Fill in the following settings:
   - **Name**: `ids-backend`
   - **Region**: `Oregon (US West)` (or closest region)
   - **Branch**: `main`
   - **Root Directory**: *(leave blank for repository root)*
   - **Runtime**: `Python 3`
   - **Build Command**:
     ```bash
     pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu && pip install -r requirements.txt
     ```
   - **Start Command**:
     ```bash
     uvicorn backend.app.main:app --host 0.0.0.0 --port $PORT
     ```
   - **Instance Type**: `Free`
4. Under **Environment Variables**, add:
   - `PYTHON_VERSION`: `3.11.9`
   - `PORT`: `10000`
5. Click **Create Web Service**. Wait 2–3 minutes for it to build and start.
6. Copy your live backend URL (e.g. `https://ids-backend-xxxx.onrender.com`).

---

### Step 2: Deploy Frontend (Static Site)
1. On [Render Dashboard](https://dashboard.render.com/), click **New +** > **Static Site**.
2. Select repository: `aamangit10/ids`.
3. Fill in the following settings:
   - **Name**: `ids-frontend`
   - **Branch**: `main`
   - **Root Directory**: `frontend`
   - **Build Command**: `npm install && npm run build`
   - **Publish Directory**: `dist`
4. Under **Environment Variables**, add:
   - **Key**: `VITE_API_URL`
   - **Value**: `https://ids-backend-xxxx.onrender.com` *(paste your backend URL from Step 1)*
5. Under **Redirects/Rewrites**:
   - **Type**: `Rewrite`
   - **Source**: `/*`
   - **Destination**: `/index.html`
6. Click **Create Static Site**.
7. Once deployed, open your live frontend URL (e.g. `https://ids-frontend-xxxx.onrender.com`) on any computer!

---

## 📦 Method 3: Run Locally or on Any Laptop/PC

If you or anyone else wants to run the full application on any laptop or desktop:

### Option A: Standard Local Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/aamangit10/ids.git
   cd ids
   ```
2. Install Python dependencies:
   ```bash
   pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
   pip install -r requirements.txt
   ```
3. Start backend:
   ```bash
   uvicorn backend.app.main:app --reload --port 8000
   ```
4. In another terminal, start frontend:
   ```bash
   cd frontend
   npm install
   npm run dev
   ```
5. Open `http://localhost:3000` in your browser.

---

### Option B: Docker Compose (1 Command)
```bash
docker compose up --build
```
- Frontend UI: `http://localhost:3000`
- Backend API: `http://localhost:8000`
- API Docs: `http://localhost:8000/docs`

---

## 💡 Important Render Free Tier Tips

1. **PyTorch CPU Wheel**: Render's free tier has 512 MB RAM. Standard PyTorch downloads ~2.5 GB of CUDA packages which times out. The build command specifically uses `--index-url https://download.pytorch.org/whl/cpu` to install the lightweight ~180 MB CPU build.
2. **Cold Starts**: Render free Web Services spin down after 15 minutes of inactivity. When you visit the app after inactivity, the first backend request may take ~30–50 seconds to wake up.
3. **CORS Configured**: CORS is set to allow requests from any client domain, so your frontend Static Site connects cleanly to the backend Web Service without browser errors.

