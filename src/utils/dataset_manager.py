import os
import shutil
import json
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import cv2
import numpy as np
from tqdm import tqdm
import yaml
import pandas as pd

class DatasetManager:
    def __init__(self, base_dir: str):
        """
        Initialize dataset manager.
        Args:
            base_dir: Base directory for dataset management
        """
        self.base_dir = Path(base_dir)
        self.datasets_dir = self.base_dir / 'datasets'
        self.annotations_dir = self.base_dir / 'annotations'
        self.stats_dir = self.base_dir / 'stats'
        
        # Create necessary directories
        for dir_path in [self.datasets_dir, self.annotations_dir, self.stats_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

    def import_dataset(self, 
                      dataset_path: str,
                      dataset_name: str,
                      format_type: str = 'yolo') -> Dict:
        """
        Import a dataset from a directory.
        Args:
            dataset_path: Path to dataset
            dataset_name: Name for the imported dataset
            format_type: Format of annotations ('yolo', 'coco', etc.)
        Returns:
            Dictionary with import statistics
        """
        dataset_dir = self.datasets_dir / dataset_name
        dataset_dir.mkdir(exist_ok=True)
        
        # Copy dataset files
        stats = {
            'images': 0,
            'annotations': 0,
            'classes': set()
        }
        
        # Copy images
        images_dir = Path(dataset_path) / 'images'
        target_images_dir = dataset_dir / 'images'
        target_images_dir.mkdir(exist_ok=True)
        
        for img_file in tqdm(list(images_dir.glob('*')), desc="Importing images"):
            if img_file.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                shutil.copy2(img_file, target_images_dir)
                stats['images'] += 1
        
        # Copy and process annotations
        if format_type == 'yolo':
            self._import_yolo_annotations(
                Path(dataset_path) / 'labels',
                dataset_dir / 'labels',
                stats
            )
        
        # Save dataset metadata
        metadata = {
            'name': dataset_name,
            'format': format_type,
            'statistics': {
                'image_count': stats['images'],
                'annotation_count': stats['annotations'],
                'classes': list(stats['classes'])
            }
        }
        
        with open(dataset_dir / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=4)
            
        return metadata

    def _import_yolo_annotations(self, 
                               src_dir: Path,
                               dst_dir: Path,
                               stats: Dict) -> None:
        """Import YOLO format annotations."""
        dst_dir.mkdir(exist_ok=True)
        
        for ann_file in tqdm(list(src_dir.glob('*.txt')), desc="Importing annotations"):
            shutil.copy2(ann_file, dst_dir)
            
            # Read classes from annotations
            with open(ann_file, 'r') as f:
                for line in f:
                    class_id = int(line.split()[0])
                    stats['classes'].add(class_id)
                    stats['annotations'] += 1

    def generate_dataset_stats(self, dataset_name: str) -> Dict:
        """
        Generate statistics for a dataset.
        Args:
            dataset_name: Name of the dataset
        Returns:
            Dictionary containing dataset statistics
        """
        dataset_dir = self.datasets_dir / dataset_name
        stats = {
            'image_stats': self._analyze_images(dataset_dir / 'images'),
            'annotation_stats': self._analyze_annotations(dataset_dir / 'labels'),
            'class_distribution': self._get_class_distribution(dataset_dir / 'labels')
        }
        
        # Save statistics
        stats_file = self.stats_dir / f"{dataset_name}_stats.json"
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=4)
            
        return stats

    def _analyze_images(self, images_dir: Path) -> Dict:
        """Analyze image properties."""
        sizes = []
        aspects = []
        formats = {}
        
        for img_file in tqdm(list(images_dir.glob('*')), desc="Analyzing images"):
            if img_file.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                img = cv2.imread(str(img_file))
                if img is not None:
                    h, w = img.shape[:2]
                    sizes.append((w, h))
                    aspects.append(w/h)
                    formats[img_file.suffix.lower()] = formats.get(img_file.suffix.lower(), 0) + 1
        
        return {
            'resolution_stats': {
                'min_width': min(w for w, h in sizes),
                'max_width': max(w for w, h in sizes),
                'min_height': min(h for w, h in sizes),
                'max_height': max(h for w, h in sizes),
                'avg_width': sum(w for w, h in sizes) / len(sizes),
                'avg_height': sum(h for w, h in sizes) / len(sizes)
            },
            'aspect_ratio_stats': {
                'min': min(aspects),
                'max': max(aspects),
                'avg': sum(aspects) / len(aspects)
            },
            'formats': formats
        }

    def _analyze_annotations(self, labels_dir: Path) -> Dict:
        """Analyze annotation properties."""
        box_sizes = []
        boxes_per_image = []
        
        for label_file in tqdm(list(labels_dir.glob('*.txt')), desc="Analyzing annotations"):
            boxes = 0
            with open(label_file, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        _, _, _, w, h = map(float, parts)
                        box_sizes.append((w, h))
                        boxes += 1
            boxes_per_image.append(boxes)
        
        return {
            'box_stats': {
                'min_width': min(w for w, h in box_sizes) if box_sizes else 0,
                'max_width': max(w for w, h in box_sizes) if box_sizes else 0,
                'min_height': min(h for w, h in box_sizes) if box_sizes else 0,
                'max_height': max(h for w, h in box_sizes) if box_sizes else 0,
                'avg_width': sum(w for w, h in box_sizes) / len(box_sizes) if box_sizes else 0,
                'avg_height': sum(h for w, h in box_sizes) / len(box_sizes) if box_sizes else 0
            },
            'boxes_per_image': {
                'min': min(boxes_per_image),
                'max': max(boxes_per_image),
                'avg': sum(boxes_per_image) / len(boxes_per_image)
            }
        }

    def _get_class_distribution(self, labels_dir: Path) -> Dict:
        """Get distribution of classes in dataset."""
        class_counts = {}
        
        for label_file in tqdm(list(labels_dir.glob('*.txt')), desc="Analyzing class distribution"):
            with open(label_file, 'r') as f:
                for line in f:
                    class_id = int(line.split()[0])
                    class_counts[class_id] = class_counts.get(class_id, 0) + 1
                    
        return class_counts

    def create_data_yaml(self, 
                        dataset_name: str,
                        class_names: List[str],
                        output_path: Optional[str] = None) -> str:
        """
        Create YAML configuration file for training.
        Args:
            dataset_name: Name of the dataset
            class_names: List of class names
            output_path: Optional path for yaml file
        Returns:
            Path to created yaml file
        """
        dataset_dir = self.datasets_dir / dataset_name
        
        data_dict = {
            'path': str(dataset_dir),
            'train': str(dataset_dir / 'images' / 'train'),
            'val': str(dataset_dir / 'images' / 'val'),
            'test': str(dataset_dir / 'images' / 'test'),
            'names': {i: name for i, name in enumerate(class_names)}
        }
        
        if output_path is None:
            output_path = dataset_dir / 'data.yaml'
            
        with open(output_path, 'w') as f:
            yaml.dump(data_dict, f, sort_keys=False)
            
        return str(output_path)

    def export_dataset_report(self, dataset_name: str) -> str:
        """
        Generate a detailed report of the dataset.
        Args:
            dataset_name: Name of the dataset
        Returns:
            Path to the generated report
        """
        stats = self.generate_dataset_stats(dataset_name)
        
        # Create report directory
        report_dir = self.stats_dir / f"{dataset_name}_report"
        report_dir.mkdir(exist_ok=True)
        
        # Generate plots
        self._generate_dataset_plots(stats, report_dir)
        
        # Create HTML report
        report_path = report_dir / 'report.html'
        self._generate_html_report(stats, dataset_name, report_path)
        
        return str(report_path)

    def _generate_dataset_plots(self, stats: Dict, report_dir: Path) -> None:
        """Generate visualization plots for dataset statistics."""
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        # Class distribution plot
        plt.figure(figsize=(10, 6))
        class_dist = pd.Series(stats['class_distribution'])
        sns.barplot(x=class_dist.index, y=class_dist.values)
        plt.title('Class Distribution')
        plt.xlabel('Class ID')
        plt.ylabel('Count')
        plt.savefig(report_dir / 'class_distribution.png')
        plt.close()
        
        # Aspect ratio distribution
        plt.figure(figsize=(10, 6))
        aspects = stats['image_stats']['aspect_ratio_stats']
        plt.hist(aspects, bins=50)
        plt.title('Aspect Ratio Distribution')
        plt.xlabel('Aspect Ratio')
        plt.ylabel('Count')
        plt.savefig(report_dir / 'aspect_ratios.png')
        plt.close()

    def _generate_html_report(self, stats: Dict, dataset_name: str, report_path: Path) -> None:
        """Generate HTML report with dataset statistics."""
        html_content = f"""
        <html>
            <head>
                <title>Dataset Report - {dataset_name}</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 20px; }}
                    h1 {{ color: #2c3e50; }}
                    .section {{ margin: 20px 0; padding: 20px; border: 1px solid #ddd; }}
                    .stat-box {{ background: #f8f9fa; padding: 10px; margin: 5px; }}
                </style>
            </head>
            <body>
                <h1>Dataset Report - {dataset_name}</h1>
                
                <div class="section">
                    <h2>Image Statistics</h2>
                    <div class="stat-box">
                        <h3>Resolution Stats</h3>
                        <p>Min Width: {stats['image_stats']['resolution_stats']['min_width']}</p>
                        <p>Max Width: {stats['image_stats']['resolution_stats']['max_width']}</p>
                        <p>Min Height: {stats['image_stats']['resolution_stats']['min_height']}</p>
                        <p>Max Height: {stats['image_stats']['resolution_stats']['max_height']}</p>
                    </div>
                </div>
                
                <div class="section">
                    <h2>Annotation Statistics</h2>
                    <div class="stat-box">
                        <h3>Bounding Box Stats</h3>
                        <p>Average boxes per image: {stats['annotation_stats']['boxes_per_image']['avg']:.2f}</p>
                        <p>Max boxes in an image: {stats['annotation_stats']['boxes_per_image']['max']}</p>
                    </div>
                </div>
                
                <div class="section">
                    <h2>Class Distribution</h2>
                    <img src="class_distribution.png" alt="Class Distribution">
                </div>
                
                <div class="section">
                    <h2>Aspect Ratio Distribution</h2>
                    <img src="aspect_ratios.png" alt="Aspect Ratio Distribution">
                </div>
            </body>
        </html>
        """
        
        with open(report_path, 'w') as f:
            f.write(html_content)
