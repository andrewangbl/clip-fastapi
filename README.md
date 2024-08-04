# CLIP-based Image Recognition API

This project implements an image recognition API using OpenAI's CLIP (Contrastive Language-Image Pre-Training) model, deployed on AWS ECS (Elastic Container Service). The application was initially conceived to facilitate automated pantry management for robots or smart pantry systems.


## Pipeline on AWS ECS
1. Docker Container: The application is containerized using Docker. The Dockerfile is used to build the image, which includes all necessary dependencies and the application code.

```
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
```

2. AWS ECS Cluster: The Docker container is deployed to an ECS cluster, which manages the container instances.


3. Load Balancer: An Application Load Balancer is used to distribute incoming traffic across multiple container instances for improved availability and fault tolerance.

4. API Endpoints: The FastAPI application exposes two main endpoints:
Health check: `GET /`
Image processing: `POST /process-image`

5. Firestore Integration: The application connects to a Firestore database to retrieve and update pantry items.

## Algorithm and Image Processing
The core of the application uses OpenAI's CLIP model for image recognition. Here's how it works:
1. Image Input: The API accepts base64 encoded image data.
2. Image Preprocessing: The input image is decoded and resized to 224x224 pixels to match CLIP's input requirements.
3. Item List Generation: The app combines items from two sources:
Firestore database (pantry items)
Predefined list from a text file
4. CLIP Processing: The CLIP model processes both the image and the combined item list.
5. Similarity Calculation: The model computes similarity scores between the image and each item in the list.
6. Top-K Algorithm: The app uses a top-k algorithm to determine the most likely matches
It selects the top 5 items with the highest similarity scores.
Scores are normalized to sum to 1.
If the highest normalized score exceeds a threshold (0.8), the prediction is considered confident.
7. Response: The API returns the top item, confidence level, and additional information about the top 5 matches.

## CLIP Package Processing
The CLIP model processes images and text as follows:
1. Image Encoding: The image is preprocessed and encoded into a high-dimensional vector representation.
2. Text Encoding: Each item in the combined list is tokenized and encoded into a vector representation.
3. Similarity Computation: The model computes the cosine similarity between the image vector and each text vector.
Probability Calculation: The similarities are converted to probabilities using a softmax function.
This approach allows the model to find the best textual description (from the provided list) that matches the input image.


## Setup and Deployment
1. Build the Docker image using the provided Dockerfile.
2. Push the image to Amazon ECR (Elastic Container Registry).
3. Create an ECS cluster and task definition using the ECR image.
4. Set up an Application Load Balancer and target group.
5. Create an ECS service to run the task and integrate with the load balancer.
6. Ensure proper IAM roles and security groups are configured for ECS and ECR access.
7. For detailed AWS ECS deployment steps, refer to the AWS documentation.
