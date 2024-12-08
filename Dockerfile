# syntax=docker/dockerfile:1

FROM python:3.10.12

WORKDIR /app

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

COPY . .

CMD ["streamlit","run","main.py"]