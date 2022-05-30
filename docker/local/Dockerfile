# Extends basic dockerfile with ability to build locally
FROM remlabase

COPY src src
COPY data data

RUN python src/text_preprocessing.py &&\
	python src/text_classification.py

EXPOSE 8080
ENTRYPOINT ["python", "src/serve_model.py"]
