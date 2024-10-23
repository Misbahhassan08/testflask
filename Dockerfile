# Use the official Python image.
# https://hub.docker.com/_/python
FROM python:3.9

# Install manually all the missing libraries
# For fixing ImportError: libGL.so.1: cannot open shared object file: No such file or directory
RUN apt-get update
RUN apt install -y libgl1-mesa-glx

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

# Copy local code to the container image.
ENV APP_HOME /app
WORKDIR $APP_HOME
COPY . .

CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 main:app