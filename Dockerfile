FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1
ENV PORT=10000

WORKDIR /app

COPY requirements.txt .

# تثبيت المتطلبات مع منع تثبيت opencv-python
RUN pip install --no-cache-dir --no-deps -r requirements.txt && \
    pip install --no-cache-dir --no-deps fastapi uvicorn ultralytics numpy pillow python-multipart pydantic && \
    pip install --no-cache-dir opencv-python-headless==4.10.0.84

COPY app.py .
COPY last.pt .

EXPOSE $PORT

CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port $PORT"]
