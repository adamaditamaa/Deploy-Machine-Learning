from python:3.8-slim

copy requirements.txt requirements.txt

copy . /app

workdir /

env FLASK_APP app/app.py

run pip install -r requirements.txt

entrypoint ["python", "-m","flask","run","--host:0.0.0.0","port:9000","--reload"]