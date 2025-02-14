from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
import sys
import os
from typing import List
import io
import base64
from PIL import Image, ImageDraw

# Add the src directory to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.preprocessing.image_enhancement import ImageEnhancer
from src.training.object_detection import UnderwaterObjectDetector

app = FastAPI(title="Underwater Image Detection API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize image enhancer and object detector
enhancer = ImageEnhancer()

@app.post("/enhance")
async def enhance_image(
    file: UploadFile = File(...),
    apply_denoise: bool = True,
    apply_color_correction: bool = True,
    apply_clahe: bool = True
):
    """
    Enhance an underwater image using various techniques.
    """
    # Read image file
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Enhance image
    enhanced_image = enhancer.enhance_image(
        image,
        apply_denoise=apply_denoise,
        apply_color_correction=apply_color_correction,
        apply_clahe=apply_clahe
    )

    # Convert OpenCV image to PIL
    enhanced_image_rgb = cv2.cvtColor(enhanced_image, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(enhanced_image_rgb)
    
    # Save to bytes
    img_byte_arr = io.BytesIO()
    pil_image.save(img_byte_arr, format='PNG')
    img_byte_arr = img_byte_arr.getvalue()
    
    return {
        "message": "Image enhanced successfully",
        "enhanced_image": img_byte_arr
    }

@app.post("/detect")
async def detect_objects(file: UploadFile = File(...)):
    try:
        print("Starting detection process...")
        
        # Read image file
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image file")
            
        # Ensure image is not too large
        max_size = 1024
        height, width = image.shape[:2]
        if max(height, width) > max_size:
            scale = max_size / max(height, width)
            new_width = int(width * scale)
            new_height = int(height * scale)
            image = cv2.resize(image, (new_width, new_height))
        
        print(f"Image loaded successfully: shape={image.shape}, dtype={image.dtype}")
            
        # Initialize detector
        try:
            detector = UnderwaterObjectDetector()
        except Exception as e:
            print(f"Error initializing detector: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Failed to initialize detector: {str(e)}")
        
        # Perform detection
        print("Running object detection...")
        try:
            annotated_image, detections = detector.detect_objects(image)
            
            if len(detections) == 0:
                print("No objects detected in the image")
            else:
                print(f"Detected {len(detections)} objects")
                for det in detections:
                    print(f"Found {det['class']} with confidence {det['confidence']:.2f}")
            
            # Convert image to RGB for encoding
            annotated_rgb = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
            
            # Use PIL for reliable encoding
            pil_image = Image.fromarray(annotated_rgb)
            
            # Save to bytes with optimal settings
            img_byte_arr = io.BytesIO()
            pil_image.save(img_byte_arr, format='JPEG', quality=95, optimize=True)
            img_str = base64.b64encode(img_byte_arr.getvalue()).decode()
            
            print("Successfully encoded result image")
            
            return {
                "status": "success",
                "message": f"Detected {len(detections)} objects",
                "detections": detections,
                "annotated_image": f"data:image/jpeg;base64,{img_str}"
            }
            
        except Exception as detect_error:
            print(f"Detection error: {str(detect_error)}")
            # If detection fails, return original image with error
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_image)
            
            # Draw error message
            draw = ImageDraw.Draw(pil_image)
            text = "Detection failed"
            draw.text((10, 10), text, fill='red')
            
            # Save to bytes
            img_byte_arr = io.BytesIO()
            pil_image.save(img_byte_arr, format='JPEG', quality=95, optimize=True)
            img_str = base64.b64encode(img_byte_arr.getvalue()).decode()
            
            return {
                "status": "error",
                "message": f"Detection failed: {str(detect_error)}",
                "detections": [],
                "annotated_image": f"data:image/jpeg;base64,{img_str}"
            }
            
    except Exception as e:
        error_msg = str(e)
        print(f"Error in detection process: {error_msg}")
        raise HTTPException(
            status_code=500,
            detail=f"Detection error: {error_msg}"
        )

@app.get("/model/metrics")
async def get_model_metrics():
    """
    Get current model's performance metrics.
    """
    try:
        detector = UnderwaterObjectDetector()
        metrics = detector.evaluate_model("data/val")
        return {
            "message": "Model metrics retrieved successfully",
            "metrics": metrics
        }
    except Exception as e:
        return {
            "message": f"Error retrieving metrics: {str(e)}",
            "metrics": None
        }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
