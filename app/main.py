# path is for docker image
# https://www.youtube.com/watch?v=pJ_nCklQ65w video for reference to deploy fastapi on aws ec2
import torch
import clip
from PIL import Image
import numpy as np
import io

import logging
from fastapi import FastAPI, HTTPException
import base64
from pydantic import BaseModel, Field

from dotenv import load_dotenv
import os

import firebase_admin
from firebase_admin import credentials, firestore

import uvicorn

# Load environment variables from .env file
load_dotenv()

# Get the similarity threshold from the environment variable, default to 0.5 if not set
# SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD"))

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

app = FastAPI()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

cred = credentials.Certificate("app/serviceAccountKey.json")
# Initialize the app with a service account, granting admin privileges
firebase_admin.initialize_app(cred)

# print(clip.available_models())

# Function for embedding images
def Images(image):
    processed_image = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        image_embeddings = model.encode_image(processed_image)
    return image_embeddings


# Function for comparing image and text similarity
def Compare(image, text: str):
    # print(text)
    image = preprocess(image).unsqueeze(0).to(device)
    text = clip.tokenize(text).to(device)

    with torch.no_grad():
        logits_per_image, logits_per_text = model(image, text)
        probs = logits_per_image.softmax(dim=-1).cpu().numpy()
        return np.ravel(probs)


def load_items(file_path):
    with open(file_path, 'r') as file:
        items = file.read().splitlines()
    return items


class ImageRequest(BaseModel):
    image_data: str = Field(..., min_length=1)

def get_confidence_topk(similarity_scores, items, k=5, threshold=0.8):
    # Get indices of top k scores
    top_k_indices = np.argsort(similarity_scores)[-k:][::-1]

    # Get top k items and their scores
    top_k_items = [items[i] for i in top_k_indices]
    top_k_scores = similarity_scores[top_k_indices]

    # Normalize top k scores
    normalized_scores = top_k_scores / np.sum(top_k_scores)

    # Check if the highest score is significantly higher than the others
    is_confident = normalized_scores[0] > threshold

    return is_confident, top_k_items, normalized_scores


@app.get("/")
async def health_check():
    return {"message": "O"}


@app.post("/process-image")
async def process_image(request: ImageRequest):
    try:

        # Reference to Firestore
        db = firestore.client()

        # Reference to the 'pantry' collection
        pantry_ref = db.collection('pantry')

        # Get all documents in the 'pantry' collection
        docs = pantry_ref.stream()
        pantry_list = [doc.id for doc in docs]


        predefined_item = load_items("app/data/items.txt")

        # Combine pantry_list and predefined_items, removing duplicates
        combined_items = list(set(pantry_list + predefined_item))

        # print(combined_items)


        # Decode base64 image
        image_data = base64.b64decode(request.image_data)
        image = Image.open(io.BytesIO(image_data))

        # Resize image
        image = image.resize((224, 224))

        similarity_result = Compare(image=image, text=combined_items)

        # Print out all similarities for each item
        # for item, score in zip(combined_items, similarity_result):
        #     print(f"{item}: {score}")

        is_confident, top_k_items, top_k_scores = get_confidence_topk(similarity_result, combined_items)

        # Find the index of the highest similarity score
        max_index = np.argmax(similarity_result)
        max_similarity = similarity_result[max_index]


        if is_confident:
            return {
                'item': top_k_items[0],
                "message": 1,
                'similarity_score': float(max_similarity),
                'top_k_items': top_k_items,
                'top_k_scores': top_k_scores.tolist()
            }
        else:
            return {
                "item": top_k_items[0],
                "message": 0,
                "similarity_score": float(max_similarity),
                'top_k_items': top_k_items,
                'top_k_scores': top_k_scores.tolist()
            }

    except ValueError as e:
        logger.error(f"Invalid input: {str(e)}")
        raise HTTPException(status_code=400, detail="Invalid input")
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        raise HTTPException(status_code=500, detail="An error occurred while processing the image")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
