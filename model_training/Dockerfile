# FROM gcr.io/deeplearning-platform-release/tf2-gpu.2-8
FROM python:3.10.3-slim-bullseye

WORKDIR /app

COPY ./requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
CMD [ "python", "main.py" ]