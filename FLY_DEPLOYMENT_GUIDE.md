# Fly.io Deployment Guide - DDSP Neural Cello
## Step-by-Step Instructions

---

### ✅ **PRE-REQUIREMENTS**

Before you start, make sure you have:
- ✅ Fly.io account (sign up at fly.io)
- ✅ Fly CLI installed
- ✅ Git installed
- ✅ Your DDSP project ready

---

## 📋 **STEP 1: INSTALL FLY CLI**

### Windows (PowerShell):
```powershell
iwr https://fly.io/install.ps1 -useb | iex
```

### Mac:
```bash
curl -L https://fly.io/install.sh | sh
```

### Linux:
```bash
curl -L https://fly.io/install.sh | sh
```

---

## 📋 **STEP 2: LOGIN TO FLY.IO**

Open PowerShell/Terminal and run:
```bash
fly auth login
```

This will open your browser to sign in.

---

## 📋 **STEP 3: PREPARE YOUR PROJECT**

### 3.1 Create `fly.toml` (already created ✓)

We've created a `fly.toml` file. Make sure to change the app name:

**Edit `fly.toml`:**
```toml
app = "your-app-name"  # Change to your unique app name
primary_region = "iad"
```

### 3.2 Ensure `Dockerfile` exists (already created ✓)

We've created a `Dockerfile` for you.

### 3.3 Ensure `.dockerignore` exists (already created ✓)

We've created a `.dockerignore` to exclude unnecessary files.

---

## 📋 **STEP 4: INITIALIZE FLY APP**

Run this command in your project directory:
```bash
cd h:\DDSP
fly launch
```

**When prompted:**
- App name: `ddsp-neural-cello` (or your choice)
- Region: `iad` (Virginia)
- Would you like a Postgres DB? → **No**
- Would you like a Redis DB? → **No**
- Would you like to deploy? → **Yes** (or say No for now)

---

## 📋 **STEP 5: CREATE PERSISTENT VOLUME**

For storing uploaded models:
```bash
fly volumes create model_storage --size 1 --region iad
```

This creates a 1GB persistent volume.

---

## 📋 **STEP 6: DEPLOY TO FLY.IO**

```bash
fly deploy
```

This will:
1. Build your Docker image
2. Push it to Fly.io
3. Deploy your app
4. Show you the URL

**Your app will be live at:** `https://your-app-name.fly.dev`

---

## 📋 **STEP 7: VERIFY DEPLOYMENT**

### 7.1 Check if app is running:
```bash
fly status
```

### 7.2 View logs:
```bash
fly logs
```

### 7.3 Test the API:
```bash
curl https://your-app-name.fly.dev/health
```

---

## 📋 **STEP 8: ACCESS YOUR APP**

Once deployed, visit:
```
https://your-app-name.fly.dev
```

This will serve your HTML frontend automatically!

---

## 🔧 **TROUBLESHOOTING**

### Problem: "Port 8000 not available"
**Solution:**
```bash
fly ssh console
python -c "import os; print(os.getenv('PORT'))"
```

### Problem: "Models not found"
**Solution:**
Check volume is mounted:
```bash
fly ssh console
ls -la /app/models
```

### Problem: "App sleeps after inactivity"
**This is normal!** Fly.io will auto-wake your app when someone visits it (takes ~1-2 seconds).

To prevent sleeping:
```bash
fly scale count 1  # Keeps 1 machine always running
```

---

## 🎯 **QUICK COMMANDS REFERENCE**

```bash
# Deploy app
fly deploy

# View logs
fly logs

# SSH into your app
fly ssh console

# Check status
fly status

# Restart app
fly apps restart your-app-name

# View metrics
fly dashboard

# Update secrets
fly secrets set KEY=value

# Scale up (keep 1 machine always on)
fly scale count 1

# View domains
fly domains list
```

---

## 📊 **WHAT'S INCLUDED**

### Backend (Python):
- ✅ `ddsp_server.py` - Main server
- ✅ `ddsp_sample_based.py` - Sample synthesis
- ✅ `ddsp_trainer_integration.py` - Model wrapper
- ✅ All Python dependencies

### Frontend (Static):
- ✅ `public/index.html` - Serves at root `/`
- ✅ All static assets

### Storage:
- ✅ Persistent volume for uploaded models
- ✅ Output directory for generated audio

---

## 💰 **COST**

**Fly.io Free Tier includes:**
- 3 shared-cpu VMs
- 256MB RAM per VM
- 1GB persistent volume (you have 1GB)
- Unlimited bandwidth
- **Total: $0/month** ✅

**This is perfect for your app!**

---

## 🔄 **UPDATING YOUR APP**

After making changes:

```bash
# 1. Update files locally
# 2. Deploy
fly deploy

# Done! Changes are live in ~2 minutes
```

---

## 📦 **CUSTOM DOMAIN (OPTIONAL)**

### Step 1: Add domain in Fly.io:
```bash
fly domains add yourdomain.com
```

### Step 2: Update DNS:
Add CNAME record:
```
CNAME: yourdomain.com → your-app-name.fly.dev
```

### Step 3: Wait for SSL (automatic, ~5 min)

---

## ✅ **CHECKLIST**

Before deploying, verify:
- [x] `Dockerfile` exists
- [x] `fly.toml` exists  
- [x] `.dockerignore` exists
- [x] `requirements-production.txt` exists
- [x] Models directory exists
- [x] Frontend files in `public/` folder
- [ ] Fly CLI installed
- [ ] Fly auth login done
- [ ] Unique app name chosen

---

## 🚀 **READY TO DEPLOY**

### Full command sequence:

```powershell
# 1. Install Fly CLI
iwr https://fly.io/install.ps1 -useb | iex

# 2. Login
fly auth login

# 3. Navigate to project
cd h:\DDSP

# 4. Launch (first time only)
fly launch

# 5. Create volume
fly volumes create model_storage --size 1 --region iad

# 6. Deploy
fly deploy

# 7. Check logs
fly logs

# 8. Your app is live!
# Visit: https://your-app-name.fly.dev
```

---

## 🎉 **DONE!**

Your app is now live on Fly.io with:
- ✅ Free hosting
- ✅ HTTPS included
- ✅ Global edge network
- ✅ Persistent storage for models
- ✅ Auto-scaling
- ✅ Production ready!

**Your app URL:** `https://your-app-name.fly.dev`

