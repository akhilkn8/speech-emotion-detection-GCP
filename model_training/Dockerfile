# FROM gcr.io/deeplearning-platform-release/tf2-gpu.2-8
FROM python:3.12.3-slim-bullseye

WORKDIR /app

COPY ./requirements.txt .
RUN pip install -r requirements.txt

COPY . .
CMD [ "python", "main.py" ]