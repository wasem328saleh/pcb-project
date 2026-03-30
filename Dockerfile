FROM python:3.10-slim-bullseye

ENV PYTHONUNBUFFERED=1
ENV PORT=10000

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .
COPY last.pt .

EXPOSE $PORT

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "10000"]
