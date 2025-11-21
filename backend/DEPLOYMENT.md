# 🚀 Deployment Guide - Python Backend

## Quick Deployment Options

1. **Local Development** - Run directly on your machine
2. **Docker** - Containerized deployment
3. **AWS ECS** - **See `ECS_SETUP.md` for detailed guide!**
4. **Cloud Services** - AWS, GCP, Azure, etc.
5. **Serverless** - AWS Lambda, Cloud Run, etc.

---

## 1. Local Development

### Simple Start
```bash
cd backend
source venv/bin/activate
python server.py
```

---

## 2. Docker Deployment

### Build and Run
```bash
cd backend

# Build image
docker build -t rag-backend-python .

# Run container
docker run -d \
  --name rag-backend \
  -p 3000:3000 \
  -e PINECONE_API_KEY=your-key \
  -e PINECONE_INDEX=rag \
  --add-host host.docker.internal:host-gateway \
  rag-backend-python
```

### Using Docker Compose
```bash
# Create .env file
cat > .env << EOF
PINECONE_API_KEY=your-key
PINECONE_INDEX=rag
OLLAMA_HOST=http://host.docker.internal:11434
OLLAMA_MODEL=llama3
EOF

# Start services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

### Docker Health Check
```bash
# Check container health
docker ps

# Test health endpoint
docker exec rag-backend-python curl http://localhost:3000/health
```

---

## 3. Cloud Deployment

### 3.1 AWS EC2

#### Launch Instance
```bash
# 1. Launch EC2 instance (t3.small or larger)
# 2. SSH into instance
ssh -i your-key.pem ubuntu@ec2-xx-xx-xx-xx.compute.amazonaws.com

# 3. Install Python
sudo apt update
sudo apt install -y python3 python3-pip python3-venv

# 4. Clone and setup
git clone your-repo
cd backend
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 5. Install Ollama
curl -fsSL https://ollama.com/install.sh | sh
ollama serve &
ollama pull llama3

# 6. Configure
cat > .env << EOF
PINECONE_API_KEY=your-key
PINECONE_INDEX=rag
PORT=3000
EOF

# 7. Run with systemd (see below)
```

#### Systemd Service
```bash
# Create service file
sudo tee /etc/systemd/system/rag-backend.service > /dev/null <<EOF
[Unit]
Description=RAG Backend Service
After=network.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/home/ubuntu/backend
Environment="PATH=/home/ubuntu/backend/venv/bin"
ExecStart=/home/ubuntu/backend/venv/bin/python server.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

# Start service
sudo systemctl daemon-reload
sudo systemctl enable rag-backend
sudo systemctl start rag-backend

# Check status
sudo systemctl status rag-backend
```

### 3.2 AWS ECS (Docker)

#### Create Task Definition
```json
{
  "family": "rag-backend",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "1024",
  "memory": "2048",
  "containerDefinitions": [
    {
      "name": "rag-backend",
      "image": "your-registry/rag-backend-python:latest",
      "portMappings": [
        {
          "containerPort": 3000,
          "protocol": "tcp"
        }
      ],
      "environment": [
        {"name": "PORT", "value": "3000"},
        {"name": "PINECONE_INDEX", "value": "rag"}
      ],
      "secrets": [
        {
          "name": "PINECONE_API_KEY",
          "valueFrom": "arn:aws:secretsmanager:region:account:secret:key"
        }
      ],
      "healthCheck": {
        "command": ["CMD-SHELL", "curl -f http://localhost:3000/health || exit 1"],
        "interval": 30,
        "timeout": 5,
        "retries": 3
      }
    }
  ]
}
```

### 3.3 Google Cloud Run

```bash
# 1. Build and push to GCR
gcloud builds submit --tag gcr.io/PROJECT_ID/rag-backend

# 2. Deploy
gcloud run deploy rag-backend \
  --image gcr.io/PROJECT_ID/rag-backend \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --memory 2Gi \
  --cpu 2 \
  --timeout 300 \
  --set-env-vars PORT=8080 \
  --set-secrets PINECONE_API_KEY=pinecone-key:latest

# 3. Get URL
gcloud run services describe rag-backend --format 'value(status.url)'
```

---

## 6. Monitoring & Logging

### Application Logging
```python
# Add to server.py
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/var/log/rag-backend.log'),
        logging.StreamHandler()
    ]
)
```

### System Monitoring
```bash
# Install monitoring tools
pip install prometheus-client

# Add metrics endpoint to server.py
from prometheus_client import make_asgi_app

metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)
```

### Log Rotation
```bash
# /etc/logrotate.d/rag-backend
/var/log/rag-backend.log {
    daily
    rotate 14
    compress
    delaycompress
    notifempty
    create 0644 ubuntu ubuntu
}
```

---

## 7. Performance Tuning

### Uvicorn Workers
```bash
# Calculate optimal workers: (2 × CPU cores) + 1
# For 4 cores: (2 × 4) + 1 = 9 workers

uvicorn server:app --workers 9 --host 0.0.0.0 --port 3000
```

---

## 8. Security Best Practices

### Environment Variables
```bash
# Never commit .env files
# Use secure secret management:
# - AWS Secrets Manager
# - Google Secret Manager
# - HashiCorp Vault
```

### Firewall
```bash
# Only allow necessary ports
sudo ufw allow 22    # SSH
sudo ufw allow 80    # HTTP
sudo ufw allow 443   # HTTPS
sudo ufw enable
```

### API Security
```python
# Add to server.py for API key authentication
from fastapi import Security, HTTPException
from fastapi.security import APIKeyHeader

api_key_header = APIKeyHeader(name="X-API-Key")

def verify_api_key(api_key: str = Security(api_key_header)):
    if api_key != os.getenv("API_KEY"):
        raise HTTPException(status_code=403, detail="Invalid API Key")
    return api_key

# Use in endpoints
@app.post("/query", dependencies=[Depends(verify_api_key)])
async def query_rag(request: QueryRequest):
    ...
```

---

## 9. Backup & Disaster Recovery

### Backup Strategy
```bash
# Backup Pinecone index metadata
# Backup configuration files
# Document deployment steps

# Automated backup script
#!/bin/bash
tar -czf backup-$(date +%Y%m%d).tar.gz \
  .env \
  server.py \
  requirements.txt
```

---

## 10. Cost Optimization

### AWS Cost Estimates

| Setup | Instance | Monthly Cost |
|-------|----------|--------------|
| **Small** | t3.small (2GB) | $15-20 |
| **Medium** | t3.medium (4GB) | $30-40 |
| **Large** | t3.large (8GB) | $60-80 |

### Cost Reduction Tips
1. Use spot instances (70% cheaper)
2. Auto-scaling based on load
3. Reserved instances for long-term
4. Use serverless for low traffic

---

## 🆘 Troubleshooting

### High Memory Usage
```bash
# Monitor memory
watch -n 1 free -h

# Reduce workers
uvicorn server:app --workers 2
```

### Slow Responses
```bash
# Check Ollama
curl http://localhost:11434/api/tags

# Profile Python code
pip install py-spy
sudo py-spy record -o profile.svg -- python server.py
```

### Connection Issues
```bash
# Check if port is open
sudo netstat -tlnp | grep 3000

# Check logs
tail -f /var/log/rag-backend.log
```

---

## ✅ Deployment Checklist

- [ ] Environment variables configured
- [ ] Firewall rules set
- [ ] SSL certificate installed
- [ ] Monitoring enabled
- [ ] Logs configured
- [ ] Backup strategy in place
- [ ] Health checks working
- [ ] Auto-restart on failure
- [ ] Load testing completed
- [ ] Documentation updated

---

**Need help? Check:**
- `README_PYTHON.md` - Python backend docs
- `PYTHON_MIGRATION.md` - Migration guide
- `/docs` endpoint - Interactive API docs

