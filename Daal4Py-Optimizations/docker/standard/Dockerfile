FROM python:3.8

WORKDIR /inference

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

COPY src src/
ENTRYPOINT ["python", "/inference/src/run.py"]


