FROM python:3.8.13-slim

LABEL org.opencontainers.image.source https://github.com/pepijnk12/remla

RUN apt-get update \
	&& apt-get install -y --no-install-recommends git \
	&& apt-get install -y gcc g++\
	&& apt-get purge -y --auto-remove \
	&& rm -rf /var/lib/apt/lists/*

WORKDIR /root/

ENV VIRTUAL_ENV=/root/venv
RUN python -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

COPY src/inference_api/requirements.txt .
RUN python -m pip install --upgrade pip &&\
    pip install -r requirements.txt

COPY src/inference_api src
COPY data data

RUN python src/text_preprocessing.py &&\
	python src/text_classification.py

# TODO use dvc repro
#COPY dvc.yaml .
#COPY .dvc .
#RUN dvc repro

EXPOSE 8080
ENTRYPOINT ["python", "src/serve_model.py"]
