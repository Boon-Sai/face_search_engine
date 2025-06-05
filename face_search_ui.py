import gradio as gr
import weaviate
from deepface import DeepFace
import numpy as np
from PIL import Image
import logging
import cv2

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

WEAVIATE_URL = 'http://localhost:8080'
CLASS_NAME = 'FaceImage'

client = weaviate.Client(WEAVIATE_URL)

def preprocess_face(img):
    """Preprocess the image for better face detection and alignment."""
    try:
        # Convert PIL Image to OpenCV format
        img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        
        # Detect faces using MTCNN (more accurate than default detector)
        face_objs = DeepFace.extract_faces(
            img_path=img_cv,
            detector_backend='mtcnn',
            enforce_detection=True,
            align=True
        )
        
        if not face_objs:
            logger.warning("No face detected in the image")
            return None
            
        # Get the first detected face
        face_img = face_objs[0]['face']
        return face_img
        
    except Exception as e:
        logger.error(f"Error in face preprocessing: {str(e)}")
        return None

def get_embedding(img):
    """Extract face embedding with improved preprocessing."""
    try:
        # Preprocess the face
        face_img = preprocess_face(img)
        if face_img is None:
            return None
            
        # Get embedding using ArcFace model
        embedding_objs = DeepFace.represent(
            img_path=face_img,
            model_name='ArcFace',
            enforce_detection=False,
            detector_backend='skip'  # Skip detection since we already did it
        )
        
        if embedding_objs:
            return embedding_objs[0]["embedding"]
        return None
        
    except Exception as e:
        logger.error(f"Error in embedding extraction: {str(e)}")
        return None

def search_faces(query_img, k):
    """Search for similar faces with improved error handling."""
    try:
        embedding = get_embedding(query_img)
        if embedding is None:
            return [], "No face detected or face detection failed."
            
        # Query Weaviate with improved parameters
        result = client.query.get(CLASS_NAME, ["image_path", "label"])\
            .with_near_vector({
                "vector": embedding,
                "certainty": 0.7  # Minimum similarity threshold
            })\
            .with_limit(k)\
            .do()
            
        matches = result['data']['Get'][CLASS_NAME]
        if not matches:
            return [], "No matches found above similarity threshold."
            
        # Prepare results
        images = []
        captions = []
        for match in matches:
            img_path = match["image_path"]
            label = match["label"]
            try:
                img = Image.open(img_path)
                images.append(img)
                captions.append(f"{label}\n{img_path}")
            except Exception as e:
                logger.error(f"Error loading image {img_path}: {str(e)}")
                continue
                
        return images, captions
        
    except Exception as e:
        logger.error(f"Error in face search: {str(e)}")
        return [], f"Error during search: {str(e)}"

def gradio_interface(img, k):
    """Gradio interface with improved error handling."""
    if img is None:
        return None, "Please upload an image."
    if k < 1:
        return None, "Please enter a positive number for k."
        
    images, captions = search_faces(img, k)
    if not images:
        return None, captions if isinstance(captions, str) else "No results found."
    return images, captions

# Custom CSS for better styling
custom_css = """
.slider-container {
    padding: 15px;
    background: #f5f5f5;
    border-radius: 8px;
}
.slider-container label {
    font-weight: bold;
    margin-bottom: 8px;
    display: block;
}
"""

with gr.Blocks(css=custom_css) as demo:
    gr.Markdown("""
    # Face Image Search Engine
    Upload a face image and get visually similar faces from the database.
    """)
    
    with gr.Row():
        with gr.Column():
            inp_img = gr.Image(type="pil", label="Query Image")
            
            with gr.Group(elem_classes="slider-container"):
                gr.Markdown("**Number of Results (k)**")
                inp_k = gr.Slider(
                    minimum=1,
                    maximum=20,
                    value=5,
                    step=1,
                    label="",
                    interactive=True
                )
            
            btn = gr.Button("Search", variant="primary")
        
        with gr.Column():
            out_gallery = gr.Gallery(
                label="Results",
                columns=3,
                height="auto",
                show_label=True,
                object_fit="cover"
            )
            out_captions = gr.Textbox(
                label="Image Details",
                interactive=False
            )
    
    btn.click(
        fn=gradio_interface,
        inputs=[inp_img, inp_k],
        outputs=[out_gallery, out_captions]
    )

demo.launch(share=True)