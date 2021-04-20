FROM python:3.9-slim-buster
ENV PYTHONUNBUFFERED 1
# RUN apt install gcc libpq (no longer needed bc we use psycopg2-binary)

COPY xgboost-1.4.0-py3-none-manylinux2010_x86_64.whl /tmp/
RUN pip install /tmp/xgboost-1.4.0-py3-none-manylinux2010_x86_64.whl
COPY requirements.txt /tmp/
RUN pip install -r /tmp/requirements.txt

RUN mkdir -p /src
COPY src/ /src/
RUN pip install -e /src
COPY tests/ /tests/

WORKDIR /src 