import cv2
import numpy as np
from pathlib import Path
import yaml
import shutil
from typing import List, Dict, Optional
import random
from .image_enhancement import ImageEnhancer

class DataPreparation:
    def __init__(self, data_dir: str):
        """
        Initialize data preparation for underwater object detection.
        Args:
            data_dir: Root directory for dataset
        """
        self.data_dir = Path(data_dir)
        self.enhancer = ImageEnhancer()
        
        # Create necessary directories
        self.images_dir = self.data_dir / 'images'
        self.labels_dir = self.data_dir / 'labels'
        self.enhanced_dir = self.data_dir / 'enhanced'
        
        for dir_path in [self.images_dir, self.labels_dir, self.enhanced_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

    def prepare_dataset(self, 
                       train_ratio: float = 0.7,
                       val_ratio: float = 0.2,
                       test_ratio: float = 0.1,
                       enhance_images: bool = True) -> None:
        """
        Prepare dataset by splitting into train/val/test sets and enhancing images.
        """
        # Create split directories
        splits = ['train', 'val', 'test']
        for split in splits:
            (self.images_dir / split).mkdir(exist_ok=True)
            (self.labels_dir / split).mkdir(exist_ok=True)
            if enhance_images:
                (self.enhanced_dir / split).mkdir(exist_ok=True)

        # Get all image files
        image_files = list(self.images_dir.glob('*.jpg')) + list(self.images_dir.glob('*.png'))
        random.shuffle(image_files)

        # Calculate split sizes
        n_images = len(image_files)
        n_train = int(n_images * train_ratio)
        n_val = int(n_images * val_ratio)

        # Split dataset
        train_files = image_files[:n_train]
        val_files = image_files[n_train:n_train + n_val]
        test_files = image_files[n_train + n_val:]

        # Move files to respective directories
        for files, split in zip([train_files, val_files, test_files], splits):
            for img_path in files:
                # Move image
                shutil.move(str(img_path), str(self.images_dir / split / img_path.name))
                
                # Move corresponding label if exists
                label_path = self.labels_dir / f"{img_path.stem}.txt"
                if label_path.exists():
                    shutil.move(str(label_path), str(self.labels_dir / split / label_path.name))

                # Enhance image if required
                if enhance_images:
                    self._enhance_and_save_image(
                        self.images_dir / split / img_path.name,
                        self.enhanced_dir / split / img_path.name
                    )

    def create_data_yaml(self, 
                        classes: List[str],
                        output_path: Optional[str] = None) -> str:
        """
        Create data.yaml file for YOLOv8 training.
        Args:
            classes: List of class names
            output_path: Path to save data.yaml file
        Returns:
            Path to created data.yaml file
        """
        data_dict = {
            'train': str(self.images_dir / 'train'),
            'val': str(self.images_dir / 'val'),
            'test': str(self.images_dir / 'test'),
            'nc': len(classes),
            'names': classes
        }

        if output_path is None:
            output_path = self.data_dir / 'data.yaml'

        with open(output_path, 'w') as f:
            yaml.dump(data_dict, f, sort_keys=False)

        return str(output_path)

    def _enhance_and_save_image(self, input_path: Path, output_path: Path) -> None:
        """
        Enhance an image and save it to the specified path.
        """
        # Read image
        image = cv2.imread(str(input_path))
        if image is None:
            return

        # Enhance image
        enhanced = self.enhancer.enhance_image(image)

        # Save enhanced image
        cv2.imwrite(str(output_path), enhanced)

    def convert_to_yolo_format(self, 
                             annotations: List[Dict],
                             image_size: tuple) -> str:
        """
        Convert bounding box annotations to YOLO format.
        Args:
            annotations: List of annotation dictionaries with bbox and class
            image_size: Tuple of (width, height)
        Returns:
            YOLO format annotation string
        """
        yolo_annotations = []
        img_w, img_h = image_size

        for ann in annotations:
            # Get bbox coordinates and normalize
            x1, y1, x2, y2 = ann['bbox']
            x_center = ((x1 + x2) / 2) / img_w
            y_center = ((y1 + y2) / 2) / img_h
            width = (x2 - x1) / img_w
            height = (y2 - y1) / img_h

            # Class index
            class_idx = ann['class']

            # Create YOLO format string
            yolo_ann = f"{class_idx} {x_center} {y_center} {width} {height}"
            yolo_annotations.append(yolo_ann)

        return '\n'.join(yolo_annotations)
