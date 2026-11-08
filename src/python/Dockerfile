FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY titanic_model.py .

# The data is NOT baked into the image; mount src/data to /app/data at run time.
CMD ["python", "titanic_model.py"]
