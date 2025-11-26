# 1. Base Image: CHANGED from 3.10 to 3.11 to match your requirements
FROM python:3.11-slim

# 2. Work Directory
WORKDIR /app

# 3. Copy Requirements
COPY requirements.txt .

# 4. Install Dependencies
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copy Code
COPY . .

# 6. Expose Port
EXPOSE 8000

# 7. Start Command
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]