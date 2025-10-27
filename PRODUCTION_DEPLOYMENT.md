# Production Deployment Guide
## DDSP Neural Cello - embracingearth.space

### System Architecture
```
Frontend (Static HTML) → Backend API (Python) → Audio Processing → Pre-trained Models
```

### Production Requirements

#### Backend Dependencies (Minimal for Production)
- Python 3.8+
- numpy
- scipy
- librosa
- pretty_midi
- mido

**NOTE**: TensorFlow/DDSP NOT needed in production - using pre-trained sample-based models

#### What Works in Production
✅ MIDI file upload
✅ Audio synthesis from MIDI
✅ Sample-based cello synthesis (1,276 trained samples)
✅ Release/Staccato control
✅ Tone control (warm/bright/dark/vintage)
✅ Sample rate selection (44.1/48/96 kHz)
✅ Bit depth selection (16/24/32 bit)
✅ Professional mastering (compression, EQ, reverb)
✅ Audio download
✅ Custom model upload

❌ NOT in Production: Model training (pre-trained models only or user uploads)

---

## Deployment Platform Recommendations

### 🏆 RECOMMENDED: Railway.app
**Best for this project**

**Why:**
- Python + Static Frontend support
- Automatic deployments from Git
- Environment variables
- Persistent storage for models
- Free tier available
- Easy PostgreSQL/MongoDB if needed later

**Setup:**
```bash
# 1. Create account at railway.app
# 2. Connect GitHub repo
# 3. Add these files:

# railway.json
{
  "build": {
    "builder": "NIXPACKS"
  },
  "deploy": {
    "startCommand": "python ddsp_server.py"
  }
}

# Procfile
web: python ddsp_server.py

# requirements.txt (production)
numpy>=1.24.0
scipy>=1.10.0
librosa>=0.10.0
pretty-midi>=0.2.9
mido>=1.2.10
```

**Cost:** $5-10/month

---

### 🚀 ALTERNATIVE: Fly.io
**Great for Docker-based deployment**

**Why:**
- Full control with Docker
- Global edge locations
- Auto-scaling
- Persistent volumes for models

**Setup:**
```dockerfile
# Dockerfile
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY ddsp_server.py .
COPY neural_cello_training_samples/ ./neural_cello_training_samples/
COPY models/ ./models/
COPY output/ ./output/

EXPOSE 8000

CMD ["python", "ddsp_server.py"]
```

**Cost:** $5-10/month

---

### ☁️ BUDGET OPTION: Render.com
**Good for startups**

**Why:**
- Free tier for static sites
- Python service support
- Auto-deploy from Git
- Simple config

**Setup:**
- Static Site: `public/` folder (frontend)
- Web Service: `ddsp_server.py` (backend)
- Environment: Python 3

**Cost:** Free tier available, $7/month for paid

---

### 🎛️ FULL CONTROL: VPS (DigitalOcean, Linode, etc.)
**Maximum flexibility**

**Why:**
- Complete control
- cPanel available on some providers
- Run multiple services
- Custom configurations

**Setup:**
```bash
# 1. Install Python 3.10
# 2. Install dependencies: pip install -r requirements.txt
# 3. Run backend: python ddsp_server.py
# 4. Serve frontend: python -m http.server -d public 8000
# 5. Use Nginx as reverse proxy
```

**Nginx config:**
```nginx
server {
    listen 80;
    server_name yourdomain.com;
    
    location / {
        proxy_pass http://localhost:3000;
    }
    
    location /api {
        proxy_pass http://localhost:8000;
    }
}
```

**Cost:** $5-20/month

---

## Deployment Checklist

### Before Deploy:
- [ ] Remove all TensorFlow dependencies from production
- [ ] Use only sample-based synthesis
- [ ] Test model upload/download
- [ ] Set up environment variables
- [ ] Configure CORS for production domain
- [ ] Add error logging
- [ ] Set up database (optional - for user data)

### Deployment Steps (Railway - Recommended):

1. **Create Railway Project**
   ```bash
   railway init
   railway login
   ```

2. **Configure Build**
   - Language: Python
   - Start command: `python ddsp_server.py`

3. **Add Environment Variables**
   ```
   PORT=8000
   MODEL_PATH=./models
   OUTPUT_PATH=./output
   TRAINING_DATA_PATH=./neural_cello_training_samples
   ```

4. **Deploy**
   ```bash
   git push origin main
   railway up
   ```

5. **Add Domain**
   - In Railway dashboard → Settings → Domains
   - Add custom domain or use railway.app subdomain

---

## Production Architecture

```
┌─────────────────┐
│   User Browser  │
│  (localhost:3000│
│   or domain)    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Static HTML   │◄──────┤ Served from public/
│   (index.html)  │       │ Can also be CDN
└────────┬────────┘
         │
         ▼ HTTP API calls
┌─────────────────┐
│  Python Backend │◄──────┤ Port 8000
│  (ddsp_server)  │       │ Railway/Fly.io
└────────┬────────┘
         │
         ├──► Audio synthesis
         ├──► Model loading
         └──► File storage
```

---

## Key Points for Production

1. **No Training in Prod**: Only pre-trained models or user uploads
2. **Static Frontend**: `public/index.html` can be on CDN (Cloudflare Pages, etc.)
3. **Python Backend**: Needs hosting for audio processing
4. **Model Storage**: Keep trained models in `/models/` directory
5. **Output Storage**: Audio files in `/output/` directory

---

## Recommended Stack

**Best Choice: Railway.app**
- Frontend + Backend in one
- Easy deployment
- Built-in scaling
- Environment variables
- Persistent volumes

**Files needed:**
```
ddsp_server.py          # Backend server
requirements.txt         # Python deps
Procfile                # Start command
railway.json            # Config
public/index.html       # Frontend
```

---

## Next Steps

1. ✅ Fix any remaining bugs
2. ✅ Test full MIDI → Audio flow
3. ✅ Deploy to Railway
4. ✅ Test on production domain
5. ✅ Monitor logs for errors
6. ✅ Add rate limiting (optional)
7. ✅ Add user authentication (optional)

---

## Cost Estimate

| Platform | Cost | Suitable? |
|----------|------|-----------|
| Railway.app | $5-10/mo | ✅ Best |
| Fly.io | $5-10/mo | ✅ Great |
| Render.com | $0-7/mo | ✅ Good |
| VPS | $5-20/mo | ✅ Full control |
| Cloudflare Pages | Free | ❌ Static only |

