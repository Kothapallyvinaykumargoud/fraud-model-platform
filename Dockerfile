# FR-5: Containerization. Serves the Model Version named by
# production_pointer.json — must be committed and up to date before
# building (the CI pipeline in .github/workflows/build-and-deploy.yml
# handles this on every push).
FROM python:3.11-slim

WORKDIR /app
ENV PYTHONUNBUFFERED=1

COPY serving/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY model/data.py model/__init__.py model/
COPY serving/ serving/
COPY production_pointer.json .
COPY models/production/ models/production/

EXPOSE 8000
CMD ["uvicorn", "serving.app:app", "--host", "0.0.0.0", "--port", "8000"]
