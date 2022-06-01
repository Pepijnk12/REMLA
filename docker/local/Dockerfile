# Extends basic dockerfile with ability to build locally
FROM remlabase

COPY . .
RUN dvc repro

EXPOSE 8080
ENTRYPOINT ["python", "src/serve_model.py"]
