import cv2
import numpy as np
from typing import List, Dict, Tuple
import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
from torchvision.transforms import functional as F
import os
import torchvision

class UnderwaterObjectDetector:
    def __init__(self):
        """Initialize the underwater object detector"""
        try:
            # Initialize model with better backbone
            self.model = fasterrcnn_resnet50_fpn_v2(
                pretrained=True,
                box_score_thresh=0.3,
                rpn_post_nms_top_n_test=1000,
                box_nms_thresh=0.3
            )
            self.model.eval()
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.model.to(self.device)
            
            # COCO class mapping to underwater objects
            self.coco_classes = {
                1: 'fish',    # person -> fish
                2: 'fish',    # bicycle -> fish
                3: 'fish',    # car -> fish
                8: 'fish',    # boat -> fish
                14: 'fish',   # bench -> fish
                15: 'fish',   # bird -> fish
                16: 'fish',   # cat -> fish
                17: 'fish',   # dog -> fish
                21: 'fish',   # elephant -> fish
                23: 'fish',   # bear -> fish
                24: 'fish',   # zebra -> fish
                25: 'fish',   # giraffe -> fish
                27: 'fish',   # backpack -> fish
                28: 'fish',   # umbrella -> fish
                64: 'fish',   # potted plant -> fish
                77: 'fish',   # cell phone -> fish
                88: 'fish',   # teddy bear -> fish
            }
            
            print(f"Model initialized successfully on {self.device}")
            
        except Exception as e:
            print(f"Error initializing model: {str(e)}")
            raise

    def enhance_underwater_image(self, image):
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
            
            # Additional contrast enhancement
            result = cv2.convertScaleAbs(result, alpha=1.3, beta=10)
            
            return result
            
        except Exception as e:
            print(f"Error enhancing image: {str(e)}")
            return image

    def detect_objects(self, image: np.ndarray) -> Tuple[np.ndarray, List[Dict]]:
        """Detect underwater objects in an image"""
        try:
            # Enhance image
            enhanced = self.enhance_underwater_image(image)
            
            # Convert to RGB for PyTorch
            rgb_image = cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB)
            
            # Convert to tensor
            tensor_image = F.to_tensor(rgb_image).to(self.device)
            
            # Normalize
            tensor_image = F.normalize(
                tensor_image,
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
            
            # Run inference with multiple scales
            scales = [0.8, 1.0, 1.2]
            all_predictions = []
            
            for scale in scales:
                # Resize image
                scaled_image = F.resize(tensor_image, [int(tensor_image.shape[1] * scale), int(tensor_image.shape[2] * scale)])
                
                # Run inference
                with torch.no_grad():
                    predictions = self.model([scaled_image])[0]
                    
                    # Rescale boxes back to original size
                    if scale != 1.0:
                        predictions['boxes'] = predictions['boxes'] / scale
                    
                    all_predictions.append(predictions)
            
            # Combine predictions
            combined_boxes = torch.cat([p['boxes'] for p in all_predictions])
            combined_scores = torch.cat([p['scores'] for p in all_predictions])
            combined_labels = torch.cat([p['labels'] for p in all_predictions])
            
            # Non-maximum suppression
            keep = torchvision.ops.nms(
                combined_boxes,
                combined_scores,
                iou_threshold=0.3
            )
            
            # Process detections
            all_detections = []
            annotated_image = enhanced.copy()
            height, width = image.shape[:2]
            
            # Filter predictions
            for idx in keep:
                score = combined_scores[idx].item()
                label = combined_labels[idx].item()
                
                # Map to fish if it's in our mapping
                if label in self.coco_classes and score > 0.3:
                    box = combined_boxes[idx].cpu().numpy()
                    x1, y1, x2, y2 = map(int, box)
                    
                    # Ensure coordinates are valid
                    x1 = max(0, min(x1, width - 1))
                    x2 = max(0, min(x2, width - 1))
                    y1 = max(0, min(y1, height - 1))
                    y2 = max(0, min(y2, height - 1))
                    
                    if x2 <= x1 or y2 <= y1:
                        continue
                    
                    # Always map to fish for underwater objects
                    class_name = 'fish'
                    
                    # Add detection
                    all_detections.append({
                        'bbox': [x1, y1, x2, y2],
                        'confidence': score,
                        'class': class_name
                    })
                    
                    # Draw box
                    cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
                    # Add label
                    label_text = f"{class_name}: {score:.2f}"
                    cv2.putText(
                        annotated_image,
                        label_text,
                        (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 0),
                        2
                    )
            
            print(f"Found {len(all_detections)} fish")
            return annotated_image, all_detections
            
        except Exception as e:
            print(f"Error in detect_objects: {str(e)}")
            return image.copy(), []
