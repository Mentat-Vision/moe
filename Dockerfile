FROM docker.io/python:3.13.5

WORKDIR /app

COPY ./src .
COPY ./requirements.txt .

RUN apt-get update && apt-get install -y libgl1-mesa-glx

RUN pip install -r requirements.txt

EXPOSE 5000

CMD ["python3", "app.py"]