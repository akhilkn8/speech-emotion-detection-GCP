FROM python:3.12.3-slim-bullseye

RUN apt-get update && \
    apt-get -y install libsndfile1
WORKDIR /app

COPY ./requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY --chmod=777 . .
CMD [ "python", "main.py" ]