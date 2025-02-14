import os
import sys

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(project_root)

import streamlit as st
import cv2
import numpy as np
import sys
import os
from PIL import Image
import io
import requests
from src.preprocessing.image_enhancement import ImageEnhancer

def process_image(image_bytes):
    """Process image bytes to numpy array safely"""
    try:
        # Convert to numpy array
        nparr = np.frombuffer(image_bytes, np.uint8)
        
        # Decode image
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            st.error("Failed to decode image. Please try another image.")
            return None
            
        # Ensure correct size
        max_size = 1024
        height, width = img.shape[:2]
        if max(height, width) > max_size:
            scale = max_size / max(height, width)
            new_width = int(width * scale)
            new_height = int(height * scale)
            img = cv2.resize(img, (new_width, new_height))
            
        return img
    except Exception as e:
        st.error(f"Error processing image: {str(e)}")
        return None

def detect_objects(image):
    """Send image to backend for detection"""
    try:
        # Convert image to bytes
        success, buffer = cv2.imencode('.jpg', image, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        if not success:
            st.error("Failed to encode image for detection")
            return None
            
        # Create file-like object
        image_bytes = io.BytesIO(buffer)
        
        # Send to backend
        files = {'file': ('image.jpg', image_bytes, 'image/jpeg')}
        response = requests.post('http://localhost:8000/detect', files=files)
        
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Detection failed: {response.text}")
            return None
    except Exception as e:
        st.error(f"Error during detection: {str(e)}")
        return None

def main():
    st.set_page_config(
        page_title="Underwater Image Detection & Enhancement",
        page_icon="🌊",
        layout="wide"
    )

    st.title("🌊 Underwater Image Detection & Enhancement")
    st.write("Upload an underwater image to enhance it and detect objects.")

    # Initialize image enhancer
    enhancer = ImageEnhancer()

    # Sidebar options
    st.sidebar.title("Settings")
    task = st.sidebar.radio("Select Task", ["Object Detection", "Image Enhancement", "Both"])

    # File uploader
    uploaded_file = st.file_uploader("Choose an image...", type=['jpg', 'jpeg', 'png'])

    if uploaded_file is not None:
        # Process uploaded image
        image = process_image(uploaded_file.read())
        
        if image is not None:
            # Create columns for display
            if task == "Both":
                col1, col2, col3 = st.columns(3)
            else:
                col1, col2 = st.columns(2)

            # Show original image
            with col1:
                st.subheader("Original Image")
                st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

            # Handle enhancement
            if task in ["Image Enhancement", "Both"]:
                with st.sidebar:
                    st.subheader("Enhancement Settings")
                    clahe_clip = st.slider("CLAHE Clip Limit", 1.0, 5.0, 2.0)
                    denoise_h = st.slider("Denoise Strength", 5.0, 15.0, 10.0)
                
                enhanced = image
                with st.spinner("Enhancing image..."):
                    # Apply enhancements
                    enhanced = enhancer.apply_clahe(enhanced, clip_limit=clahe_clip)
                    enhanced = enhancer.color_correction(enhanced)
                    enhanced = enhancer.denoise_image(enhanced, h=denoise_h)
                
                # Show enhanced image
                with col2 if task == "Image Enhancement" else col2:
                    st.subheader("Enhanced Image")
                    st.image(cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB))

            # Handle detection
            if task in ["Object Detection", "Both"]:
                # Use enhanced image if available
                detect_image = enhanced if task == "Both" else image
                
                with st.spinner("Detecting objects..."):
                    result = detect_objects(detect_image)
                    
                if result is not None:
                    col = col2 if task == "Object Detection" else col3
                    with col:
                        st.subheader("Detection Result")
                        if "annotated_image" in result:
                            st.image(result["annotated_image"])
                            if "detections" in result:
                                st.write(f"Found {len(result['detections'])} objects")
                                for det in result['detections']:
                                    st.write(f"- {det['class']}: {det['confidence']:.2f}")
                        else:
                            st.error("No detection result received")

if __name__ == "__main__":
    main()
