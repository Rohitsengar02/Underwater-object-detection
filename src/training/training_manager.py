import os
from pathlib import Path
import json
from datetime import datetime
from typing import Dict, Optional, List
import torch
from ultralytics import YOLO
import mlflow
from tqdm import tqdm
import numpy as np
import threading
import queue
from ..utils.visualization import TrainingVisualizer

class TrainingMonitor:
    def __init__(self):
        self.metrics_queue = queue.Queue()
        self.is_training = False
        self.current_metrics = {}
        self.history = {
            'train': {},
            'val': {}
        }

    def update_metrics(self, metrics: Dict):
        """Update current metrics and add to history."""
        self.current_metrics = metrics
        
        # Update history
        for key, value in metrics.items():
            phase = 'train' if 'train' in key else 'val'
            metric_name = key.split('/')[-1]
            
            if metric_name not in self.history[phase]:
                self.history[phase][metric_name] = []
            
            self.history[phase][metric_name].append(value)
        
        # Add to queue for real-time monitoring
        self.metrics_queue.put(metrics)

    def get_current_metrics(self) -> Dict:
        """Get the most recent metrics."""
        return self.current_metrics

    def get_history(self) -> Dict:
        """Get complete training history."""
        return self.history

    def start_training(self):
        """Mark the start of training."""
        self.is_training = True
        self.current_metrics = {}
        self.history = {'train': {}, 'val': {}}

    def end_training(self):
        """Mark the end of training."""
        self.is_training = False

class TrainingManager:
    def __init__(self, base_dir: str):
        """
        Initialize training manager with monitoring and visualization.
        Args:
            base_dir: Base directory for training management
        """
        self.base_dir = Path(base_dir)
        self.models_dir = self.base_dir / 'models'
        self.runs_dir = self.base_dir / 'runs'
        self.configs_dir = self.base_dir / 'configs'
        
        # Create necessary directories
        for dir_path in [self.models_dir, self.runs_dir, self.configs_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
            
        # Initialize MLflow
        mlflow.set_tracking_uri(f"file://{str(self.runs_dir)}")
        mlflow.set_experiment("underwater_detection")

        # Initialize monitor and visualizer
        self.monitor = TrainingMonitor()
        self.visualizer = TrainingVisualizer(str(self.base_dir))

    def create_training_config(self,
                             data_yaml_path: str,
                             model_name: str = 'yolov8n.pt',
                             epochs: int = 100,
                             batch_size: int = 16,
                             img_size: int = 640,
                             **kwargs) -> str:
        """Create a training configuration file."""
        config = {
            'model_name': model_name,
            'data_yaml': data_yaml_path,
            'training_params': {
                'epochs': epochs,
                'batch_size': batch_size,
                'img_size': img_size,
                'optimizer': kwargs.get('optimizer', 'Adam'),
                'lr0': kwargs.get('lr0', 0.01),
                'lrf': kwargs.get('lrf', 0.01),
                'momentum': kwargs.get('momentum', 0.937),
                'weight_decay': kwargs.get('weight_decay', 0.0005),
                'warmup_epochs': kwargs.get('warmup_epochs', 3),
                'warmup_momentum': kwargs.get('warmup_momentum', 0.8),
                'warmup_bias_lr': kwargs.get('warmup_bias_lr', 0.1),
                'box': kwargs.get('box', 7.5),
                'cls': kwargs.get('cls', 0.5),
                'dfl': kwargs.get('dfl', 1.5),
                'fl_gamma': kwargs.get('fl_gamma', 0.0),
                'label_smoothing': kwargs.get('label_smoothing', 0.0),
                'nbs': kwargs.get('nbs', 64),
                'hsv_h': kwargs.get('hsv_h', 0.015),
                'hsv_s': kwargs.get('hsv_s', 0.7),
                'hsv_v': kwargs.get('hsv_v', 0.4),
                'degrees': kwargs.get('degrees', 0.0),
                'translate': kwargs.get('translate', 0.1),
                'scale': kwargs.get('scale', 0.5),
                'shear': kwargs.get('shear', 0.0),
                'perspective': kwargs.get('perspective', 0.0),
                'flipud': kwargs.get('flipud', 0.0),
                'fliplr': kwargs.get('fliplr', 0.5),
                'mosaic': kwargs.get('mosaic', 1.0),
                'mixup': kwargs.get('mixup', 0.0),
                'copy_paste': kwargs.get('copy_paste', 0.0),
            },
            'device': 'cuda' if torch.cuda.is_available() else 'cpu',
            'created_at': datetime.now().isoformat()
        }
        
        config_path = self.configs_dir / f"train_config_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=4)
            
        return str(config_path)

    def train_model(self, config_path: str) -> Dict:
        """
        Train a model with real-time monitoring and visualization.
        Args:
            config_path: Path to training configuration file
        Returns:
            Dictionary containing training results
        """
        # Load config
        with open(config_path, 'r') as f:
            config = json.load(f)
            
        # Start MLflow run
        with mlflow.start_run() as run:
            # Log parameters
            mlflow.log_params(config['training_params'])
            mlflow.log_param('model_name', config['model_name'])
            mlflow.log_param('data_yaml', config['data_yaml'])
            
            # Start training monitor
            self.monitor.start_training()
            
            # Initialize model with callbacks
            model = YOLO(config['model_name'])
            
            # Custom callback for metric monitoring
            def on_train_epoch_end(trainer):
                metrics = trainer.metrics
                self.monitor.update_metrics(metrics)
                
                # Create and save visualizations
                if trainer.epoch % 10 == 0:  # Create visualizations every 10 epochs
                    history = self.monitor.get_history()
                    self.visualizer.create_training_dashboard(
                        history['train'],
                        save_path=str(self.runs_dir / run.info.run_id / f"epoch_{trainer.epoch}")
                    )
            
            # Train model
            results = model.train(
                data=config['data_yaml'],
                callbacks=[on_train_epoch_end],
                **config['training_params']
            )
            
            # End training monitor
            self.monitor.end_training()
            
            # Get final metrics
            metrics = {
                'mAP50': float(results.results_dict['metrics/mAP50(B)']),
                'mAP50-95': float(results.results_dict['metrics/mAP50-95(B)']),
                'precision': float(results.results_dict['metrics/precision(B)']),
                'recall': float(results.results_dict['metrics/recall(B)']),
                'final_loss': float(results.results_dict['train/box_loss'])
            }
            
            # Log metrics to MLflow
            mlflow.log_metrics(metrics)
            
            # Save model
            model_path = self.models_dir / f"model_{run.info.run_id}.pt"
            model.save(str(model_path))
            mlflow.log_artifact(str(model_path))
            
            # Create and save final training report
            history = self.monitor.get_history()
            report_path = self.visualizer.create_training_report(
                run_id=run.info.run_id,
                metrics=history['train'],
                final_metrics=metrics
            )
            mlflow.log_artifact(report_path)
            
            return {
                'run_id': run.info.run_id,
                'metrics': metrics,
                'model_path': str(model_path),
                'report_path': report_path
            }

    def get_training_progress(self) -> Dict:
        """Get current training progress and metrics."""
        return {
            'is_training': self.monitor.is_training,
            'current_metrics': self.monitor.get_current_metrics(),
            'history': self.monitor.get_history()
        }

    def evaluate_model(self, 
                      model_path: str,
                      data_yaml: str,
                      batch_size: int = 16) -> Dict:
        """Evaluate a trained model with visualization."""
        model = YOLO(model_path)
        
        # Run validation
        results = model.val(
            data=data_yaml,
            batch=batch_size
        )
        
        metrics = {
            'mAP50': float(results.results_dict['metrics/mAP50(B)']),
            'mAP50-95': float(results.results_dict['metrics/mAP50-95(B)']),
            'precision': float(results.results_dict['metrics/precision(B)']),
            'recall': float(results.results_dict['metrics/recall(B)'])
        }
        
        # Create evaluation visualizations
        eval_dir = self.runs_dir / 'evaluations' / datetime.now().strftime('%Y%m%d_%H%M%S')
        eval_dir.mkdir(parents=True, exist_ok=True)
        
        self.visualizer.create_performance_summary(
            metrics,
            save_path=str(eval_dir)
        )
        
        return metrics

    def get_training_history(self) -> List[Dict]:
        """Get history of all training runs with visualizations."""
        client = mlflow.tracking.MlflowClient()
        runs = client.search_runs(
            experiment_ids=[client.get_experiment_by_name("underwater_detection").experiment_id]
        )
        
        history = []
        for run in runs:
            history.append({
                'run_id': run.info.run_id,
                'status': run.info.status,
                'start_time': run.info.start_time,
                'end_time': run.info.end_time,
                'metrics': run.data.metrics,
                'parameters': run.data.params,
                'report_path': next(
                    (a.path for a in client.list_artifacts(run.info.run_id) if a.path.endswith('report.html')),
                    None
                )
            })
            
        return history

    def export_model(self, 
                    model_path: str,
                    format: str = 'onnx',
                    **kwargs) -> str:
        """
        Export a trained model to different formats.
        Args:
            model_path: Path to trained model
            format: Export format (onnx, torchscript, etc.)
            **kwargs: Additional export parameters
        Returns:
            Path to exported model
        """
        model = YOLO(model_path)
        exported_path = model.export(format=format, **kwargs)
        return str(exported_path)

    def get_model_info(self, model_path: str) -> Dict:
        """
        Get information about a trained model.
        Args:
            model_path: Path to model file
        Returns:
            Dictionary containing model information
        """
        model = YOLO(model_path)
        
        return {
            'model_type': model.type,
            'task': model.task,
            'num_classes': model.model.nc,
            'input_size': model.model.args['imgsz'],
            'parameters': sum(p.numel() for p in model.model.parameters()),
            'file_size': os.path.getsize(model_path) / (1024 * 1024)  # Size in MB
        }
