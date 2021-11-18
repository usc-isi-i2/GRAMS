FROM continuumio/anaconda3

RUN apt update && apt install -y build-essential

RUN mkdir -p /usr/share/man/man1 && apt install -y default-jre

RUN pip install poetry && poetry config virtualenvs.create false

RUN pip install -U sm-grams