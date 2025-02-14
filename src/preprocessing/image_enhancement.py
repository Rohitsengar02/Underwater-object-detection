import cv2
import numpy as np
from typing import Tuple, Optional

class ImageEnhancer:
    @staticmethod
    def enhance_underwater_image(image):
        """Enhance underwater image for better detection"""
        try:
            # Convert to LAB color space
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            
            # Apply CLAHE to L channel
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
            l = clahe.apply(l)
            
            # Increase contrast of a and b channels
            a = cv2.convertScaleAbs(a, alpha=1.3, beta=0)
            b = cv2.convertScaleAbs(b, alpha=1.3, beta=0)
            
            # Merge channels
            lab = cv2.merge((l, a, b))
            enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
            
            # Color correction
            bgr_planes = cv2.split(enhanced)
            result_planes = []
            
            for plane in bgr_planes:
                dilated = cv2.dilate(plane, np.ones((7,7), np.uint8))
                bg_img = cv2.medianBlur(dilated, 21)
                diff_img = 255 - cv2.absdiff(plane, bg_img)
                norm_img = cv2.normalize(diff_img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
                result_planes.append(norm_img)
            
            result = cv2.merge(result_planes)
            
            return result
            
        except Exception as e:
            print(f"Error enhancing image: {str(e)}")
            return image
