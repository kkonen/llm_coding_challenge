FROM python:3.11-slim

WORKDIR /llm_challenge

COPY . /llm_challenge

RUN pip install --no-cache-dir -r requirements.txt
