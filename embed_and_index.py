import os
import weaviate
from deepface import DeepFace
from tqdm import tqdm

DATA_DIR = 'data'
WEAVIATE_URL = 'http://localhost:8080'
CLASS_NAME = 'FaceImage'

# Connect to Weaviate
client = weaviate.Client(WEAVIATE_URL)

# Create schema if not exists
if not client.schema.exists(CLASS_NAME):
    schema = {
        "classes": [
            {
                "class": CLASS_NAME,
                "vectorizer": "none",
                "properties": [
                    {"name": "image_path", "dataType": ["string"]},
                    {"name": "label", "dataType": ["string"]},
                ],
            }
        ]
    }
    client.schema.create(schema)

def get_all_images(data_dir):
    for label in os.listdir(data_dir):
        label_path = os.path.join(data_dir, label)
        if not os.path.isdir(label_path):
            continue
        for fname in os.listdir(label_path):
            if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                yield label, os.path.join(label_path, fname)

def embed_image(img_path):
    # DeepFace returns a list of dicts, one per face. We use the first face.
    try:
        embedding_objs = DeepFace.represent(img_path=img_path, model_name='ArcFace', enforce_detection=False)
        if embedding_objs:
            return embedding_objs[0]["embedding"]
    except Exception as e:
        print(f"Error embedding {img_path}: {e}")
    return None

def main():
    for label, img_path in tqdm(list(get_all_images(DATA_DIR))):
        embedding = embed_image(img_path)
        if embedding is None:
            continue
        # Store in Weaviate
        data_obj = {
            "image_path": img_path,
            "label": label,
        }
        client.data_object.create(
            data_obj,
            CLASS_NAME,
            vector=embedding
        )

if __name__ == "__main__":
    main() 