import os

dockerignore_content = """venv/
.venv/
__pycache__/
*.pyc
*.pyo
*.pyd
.git/
.gitignore
.DS_Store
*.log
mlruns/.trash/
"""

dockerfile_content = """FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \\
    PYTHONDONTWRITEBYTECODE=1 \\
    PORT=8501

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \\
    build-essential \\
    curl \\
    git \\
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip && \\
    pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501
EXPOSE 5000

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

CMD ["streamlit", "run", "dashboard/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
"""

docker_compose_content = """version: '3.8'

services:
  retailpulse-dashboard:
    build:
      context: .
      dockerfile: Dockerfile
    container_name: retailpulse_dashboard
    restart: always
    ports:
      - "8501:8501"
    volumes:
      - ./data:/app/data
      - ./monitoring:/app/monitoring
      - ./mlruns:/app/mlruns
    environment:
      - STREAMLIT_SERVER_PORT=8501
      - STREAMLIT_SERVER_ADDRESS=0.0.0.0
    command: streamlit run dashboard/app.py --server.port=8501 --server.address=0.0.0.0

  mlflow-server:
    build:
      context: .
      dockerfile: Dockerfile
    container_name: retailpulse_mlflow
    restart: always
    ports:
      - "5000:5000"
    volumes:
      - ./mlruns:/app/mlruns
    command: python -m mlflow ui --backend-store-uri "./mlruns" --host 0.0.0.0 --port 5000
"""

# Write files to root
files = {
    ".dockerignore": dockerignore_content,
    "Dockerfile": dockerfile_content,
    "docker-compose.yml": docker_compose_content
}

print("🐳 Verifying and generating Docker containerization files...")
for fname, content in files.items():
    with open(fname, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"  ✅ Created/Verified: {os.path.abspath(fname)}")

print("\n🎉 Step 18 Docker files are ready in your repository!")