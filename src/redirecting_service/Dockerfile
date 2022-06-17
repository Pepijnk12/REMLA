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

COPY requirements.txt .
RUN python -m pip install --upgrade pip &&\
    pip install -r requirements.txt

COPY . src

EXPOSE 8080
ENTRYPOINT ["python", "src/serve.py"]