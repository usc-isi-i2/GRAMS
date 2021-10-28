FROM continuumio/anaconda3

RUN apt update && apt install -y build-essential

RUN pip install poetry && poetry config virtualenvs.create false

RUN pip install -U sm-grams

#RUN conda install gxx_linux-64
#
#ENV CC=/opt/conda/bin/x86_64-conda_cos6-linux-gnu-gcc
#ENV CXX=/opt/conda/bin/x86_64-conda_cos6-linux-gnu-g++
#
#RUN pip install sm-widgets[integration]
