FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt /
RUN pip3 install --no-cache-dir -r /requirements.txt
COPY *.py /app/
COPY models/ /app/models/
CMD ["python3", "live_deployment.py"]