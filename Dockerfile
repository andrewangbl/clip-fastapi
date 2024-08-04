# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /code

# Install git
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

# add requirements file to image
COPY /requirements.txt /code/requirements.txt

# install python libraries
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# add python code
COPY ./app/ /code/app/



# specify default command
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "80"]
