FROM gcr.io/deeplearning-platform-release/tf2-gpu.2-8

WORKDIR /app

COPY ./requirements.txt .
RUN pip install -r requirements.txt

COPY . .
CMD [ "python", "main.py" ]