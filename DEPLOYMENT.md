# Deployment Guide

This guide provides comprehensive instructions for deploying the AI-powered Data Analytics Platform in various environments.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Local Development Setup](#local-development-setup)
3. [Production Deployment](#production-deployment)
4. [Docker Deployment](#docker-deployment)
5. [Cloud Deployment](#cloud-deployment)
6. [Environment Configuration](#environment-configuration)
7. [Security Considerations](#security-considerations)
8. [Monitoring and Logging](#monitoring-and-logging)
9. [Troubleshooting](#troubleshooting)

## Prerequisites

### System Requirements

- **Operating System**: Windows 10/11, macOS 10.14+, or Linux (Ubuntu 18.04+)
- **Python**: Version 3.8 or higher (3.9+ recommended)
- **Memory**: Minimum 8GB RAM (16GB+ recommended for large datasets)
- **Storage**: At least 10GB free space
- **Network**: Internet connection for package installation

### Required Software

- Python 3.8+
- pip (Python package manager)
- Git (for version control)
- Optional: Docker (for containerized deployment)

## Local Development Setup

### 1. Clone the Repository

```bash
git clone <repository-url>
cd ai-data-analytics-platform
```

### 2. Create Virtual Environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Application

```bash
streamlit run app.py
```

The application will be available at `http://localhost:8501`

## Production Deployment

### 1. Server Setup

#### Minimum Server Specifications
- **CPU**: 4 cores
- **RAM**: 16GB
- **Storage**: 50GB SSD
- **Network**: 1Gbps connection

#### Recommended Server Specifications
- **CPU**: 8+ cores
- **RAM**: 32GB+
- **Storage**: 100GB+ SSD
- **Network**: 10Gbps connection

### 2. Environment Setup

```bash
# Update system packages
sudo apt update && sudo apt upgrade -y

# Install Python and dependencies
sudo apt install python3.9 python3.9-venv python3-pip nginx supervisor -y

# Create application user
sudo useradd -m -s /bin/bash appuser
sudo su - appuser
```

### 3. Application Installation

```bash
# Clone repository
git clone <repository-url> /home/appuser/ai-analytics
cd /home/appuser/ai-analytics

# Create virtual environment
python3.9 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 4. Configure Streamlit for Production

Create `/home/appuser/ai-analytics/.streamlit/config.toml`:

```toml
[server]
port = 8501
address = "0.0.0.0"
headless = true
enableCORS = false
enableXsrfProtection = true

[browser]
gatherUsageStats = false

[logger]
level = "info"

[client]
showErrorDetails = false
```

### 5. Process Management with Supervisor

Create `/etc/supervisor/conf.d/ai-analytics.conf`:

```ini
[program:ai-analytics]
command=/home/appuser/ai-analytics/venv/bin/streamlit run app.py
directory=/home/appuser/ai-analytics
user=appuser
autostart=true
autorestart=true
redirect_stderr=true
stdout_logfile=/var/log/ai-analytics.log
environment=PATH="/home/appuser/ai-analytics/venv/bin"
```

Start the service:

```bash
sudo supervisorctl reread
sudo supervisorctl update
sudo supervisorctl start ai-analytics
```

### 6. Reverse Proxy with Nginx

Create `/etc/nginx/sites-available/ai-analytics`:

```nginx
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://localhost:8501;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_cache_bypass $http_upgrade;
    }
}
```

Enable the site:

```bash
sudo ln -s /etc/nginx/sites-available/ai-analytics /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl restart nginx
```

## Docker Deployment

### 1. Create Dockerfile

```dockerfile
FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create non-root user
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Expose port
EXPOSE 8501

# Health check
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

# Run application
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

### 2. Create docker-compose.yml

```yaml
version: '3.8'

services:
  ai-analytics:
    build: .
    ports:
      - "8501:8501"
    environment:
      - PYTHONPATH=/app
    volumes:
      - ./data:/app/data
      - ./logs:/app/logs
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8501/_stcore/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
      - ./ssl:/etc/nginx/ssl
    depends_on:
      - ai-analytics
    restart: unless-stopped
```

### 3. Build and Run

```bash
docker-compose up -d
```

## Cloud Deployment

### AWS Deployment

#### Using EC2

1. Launch EC2 instance (t3.large or larger)
2. Configure security groups (ports 80, 443, 22)
3. Follow production deployment steps
4. Configure Application Load Balancer for high availability

#### Using ECS

1. Create ECS cluster
2. Build and push Docker image to ECR
3. Create task definition
4. Deploy service with auto-scaling

### Google Cloud Platform

#### Using Compute Engine

1. Create VM instance (n1-standard-4 or larger)
2. Configure firewall rules
3. Follow production deployment steps

#### Using Cloud Run

1. Build container image
2. Deploy to Cloud Run
3. Configure custom domain

### Azure Deployment

#### Using Virtual Machines

1. Create VM (Standard_D4s_v3 or larger)
2. Configure Network Security Groups
3. Follow production deployment steps

#### Using Container Instances

1. Build and push image to Azure Container Registry
2. Deploy to Container Instances
3. Configure Application Gateway

## Environment Configuration

### Environment Variables

Create `.env` file:

```bash
# Application Settings
STREAMLIT_SERVER_PORT=8501
STREAMLIT_SERVER_ADDRESS=0.0.0.0

# Security Settings
SECRET_KEY=your-secret-key-here
ENABLE_HTTPS=true

# Database Settings (if applicable)
DATABASE_URL=postgresql://user:password@localhost:5432/dbname

# AI Model Settings
OPENAI_API_KEY=your-openai-key
HUGGINGFACE_API_KEY=your-huggingface-key

# Logging
LOG_LEVEL=INFO
LOG_FILE=/var/log/ai-analytics.log

# Performance Settings
MAX_UPLOAD_SIZE=100
CACHE_TTL=3600
```

### Configuration Management

Use environment-specific configuration files:

- `config/development.py`
- `config/staging.py`
- `config/production.py`

## Security Considerations

### 1. HTTPS Configuration

```bash
# Install Certbot
sudo apt install certbot python3-certbot-nginx

# Obtain SSL certificate
sudo certbot --nginx -d your-domain.com

# Auto-renewal
sudo crontab -e
# Add: 0 12 * * * /usr/bin/certbot renew --quiet
```

### 2. Firewall Configuration

```bash
# Configure UFW
sudo ufw default deny incoming
sudo ufw default allow outgoing
sudo ufw allow ssh
sudo ufw allow 'Nginx Full'
sudo ufw enable
```

### 3. Application Security

- Use environment variables for sensitive data
- Implement rate limiting
- Enable CSRF protection
- Validate all user inputs
- Regular security updates

### 4. Data Protection

- Encrypt data at rest
- Use secure file upload validation
- Implement data retention policies
- Regular backups

## Monitoring and Logging

### 1. Application Monitoring

Install monitoring tools:

```bash
pip install prometheus-client grafana-api
```

### 2. Log Management

Configure log rotation:

```bash
# Create logrotate configuration
sudo nano /etc/logrotate.d/ai-analytics

/var/log/ai-analytics.log {
    daily
    missingok
    rotate 52
    compress
    delaycompress
    notifempty
    create 644 appuser appuser
    postrotate
        supervisorctl restart ai-analytics
    endscript
}
```

### 3. Health Checks

Implement health check endpoints:

```python
# Add to app.py
@st.cache_data
def health_check():
    return {"status": "healthy", "timestamp": datetime.now()}
```

### 4. Performance Monitoring

- Monitor CPU and memory usage
- Track response times
- Monitor error rates
- Set up alerts for critical issues

## Troubleshooting

### Common Issues

#### 1. Port Already in Use

```bash
# Find process using port 8501
sudo lsof -i :8501
# Kill process
sudo kill -9 <PID>
```

#### 2. Permission Denied

```bash
# Fix file permissions
sudo chown -R appuser:appuser /home/appuser/ai-analytics
sudo chmod -R 755 /home/appuser/ai-analytics
```

#### 3. Memory Issues

```bash
# Monitor memory usage
free -h
# Increase swap if needed
sudo fallocate -l 4G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```

#### 4. SSL Certificate Issues

```bash
# Check certificate status
sudo certbot certificates
# Renew certificate
sudo certbot renew --dry-run
```

### Performance Optimization

1. **Enable Caching**: Use Streamlit's caching decorators
2. **Optimize Data Loading**: Implement lazy loading for large datasets
3. **Resource Management**: Monitor and limit resource usage
4. **CDN Integration**: Use CDN for static assets

### Backup and Recovery

1. **Database Backups**: Regular automated backups
2. **Application Backups**: Version control and deployment artifacts
3. **Configuration Backups**: Environment and configuration files
4. **Recovery Testing**: Regular recovery procedure testing

## Support and Maintenance

### Regular Maintenance Tasks

1. **Security Updates**: Monthly security patches
2. **Dependency Updates**: Quarterly dependency reviews
3. **Performance Reviews**: Monthly performance analysis
4. **Backup Verification**: Weekly backup testing

### Getting Help

- Check application logs: `/var/log/ai-analytics.log`
- Review system logs: `journalctl -u supervisor`
- Monitor resource usage: `htop`, `iotop`
- Check network connectivity: `netstat -tlnp`

For additional support, please refer to the main README.md file or contact the development team.