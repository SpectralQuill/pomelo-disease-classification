# ---------- Base Image ----------
FROM python:3.10-slim

# ---------- Working Directory ----------
WORKDIR /app

# ---------- System Dependencies ----------
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# ---------- Copy only requirements first (for caching) ----------
COPY requirements.txt .

# ---------- Install Python dependencies ----------
RUN pip install --no-cache-dir -r requirements.txt

# ---------- Copy the rest of the backend code ----------
COPY . .

# ---------- Expose Flask Port ----------
EXPOSE 5000

# ---------- Default Command ----------
CMD ["python", "app.py"]
