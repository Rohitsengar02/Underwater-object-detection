import optuna
from optuna.trial import Trial
import mlflow
from pathlib import Path
import json
from typing import Dict, Optional, List
import yaml
from .training_manager import TrainingManager
import numpy as np
from datetime import datetime

class HyperparameterTuner:
    def __init__(self, base_dir: str):
        """
        Initialize hyperparameter tuner.
        Args:
            base_dir: Base directory for tuning experiments
        """
        self.base_dir = Path(base_dir)
        self.tuning_dir = self.base_dir / 'hyperparameter_tuning'
        self.tuning_dir.mkdir(parents=True, exist_ok=True)
        
        self.training_manager = TrainingManager(base_dir)
        
    def create_tuning_config(self,
                           data_yaml_path: str,
                           n_trials: int = 20,
                           timeout: Optional[int] = None,
                           parameter_space: Optional[Dict] = None) -> str:
        """
        Create a configuration for hyperparameter tuning.
        Args:
            data_yaml_path: Path to data.yaml file
            n_trials: Number of trials to run
            timeout: Timeout in seconds
            parameter_space: Custom parameter space definition
        Returns:
            Path to tuning configuration file
        """
        if parameter_space is None:
            parameter_space = {
                'learning_rate': {
                    'type': 'float',
                    'low': 1e-5,
                    'high': 1e-1,
                    'log': True
                },
                'batch_size': {
                    'type': 'categorical',
                    'choices': [8, 16, 32, 64]
                },
                'img_size': {
                    'type': 'categorical',
                    'choices': [416, 512, 640, 768]
                },
                'mosaic': {
                    'type': 'float',
                    'low': 0.0,
                    'high': 1.0
                },
                'mixup': {
                    'type': 'float',
                    'low': 0.0,
                    'high': 1.0
                },
                'warmup_epochs': {
                    'type': 'int',
                    'low': 1,
                    'high': 5
                }
            }
        
        config = {
            'data_yaml': data_yaml_path,
            'n_trials': n_trials,
            'timeout': timeout,
            'parameter_space': parameter_space,
            'created_at': datetime.now().isoformat()
        }
        
        config_path = self.tuning_dir / f"tuning_config_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=4)
            
        return str(config_path)
    
    def _objective(self, trial: Trial, data_yaml: str, epochs: int = 30) -> float:
        """
        Objective function for Optuna optimization.
        Args:
            trial: Optuna trial object
            data_yaml: Path to data.yaml file
            epochs: Number of epochs for each trial
        Returns:
            Validation mAP score
        """
        # Generate hyperparameters for this trial
        params = {
            'lr0': trial.suggest_float('learning_rate', 1e-5, 1e-1, log=True),
            'batch': trial.suggest_categorical('batch_size', [8, 16, 32, 64]),
            'imgsz': trial.suggest_categorical('img_size', [416, 512, 640, 768]),
            'mosaic': trial.suggest_float('mosaic', 0.0, 1.0),
            'mixup': trial.suggest_float('mixup', 0.0, 1.0),
            'warmup_epochs': trial.suggest_int('warmup_epochs', 1, 5),
            'epochs': epochs
        }
        
        # Create training configuration
        config_path = self.training_manager.create_training_config(
            data_yaml_path=data_yaml,
            **params
        )
        
        # Train model with these parameters
        try:
            results = self.training_manager.train_model(config_path)
            map50 = results['metrics']['mAP50']
            
            # Log results to MLflow
            mlflow.log_metrics({
                'trial_mAP50': map50,
                'trial_number': trial.number
            })
            
            return map50
        
        except Exception as e:
            print(f"Trial failed: {str(e)}")
            return 0.0
    
    def run_tuning(self, config_path: str) -> Dict:
        """
        Run hyperparameter tuning experiment.
        Args:
            config_path: Path to tuning configuration file
        Returns:
            Dictionary containing best parameters and results
        """
        # Load configuration
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Create study
        study_name = f"hparam_tuning_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        study = optuna.create_study(
            study_name=study_name,
            direction='maximize',
            sampler=optuna.samplers.TPESampler()
        )
        
        # Start MLflow run for the tuning experiment
        with mlflow.start_run(run_name=study_name) as run:
            mlflow.log_param('n_trials', config['n_trials'])
            mlflow.log_param('data_yaml', config['data_yaml'])
            
            # Run optimization
            study.optimize(
                lambda trial: self._objective(trial, config['data_yaml']),
                n_trials=config['n_trials'],
                timeout=config['timeout']
            )
            
            # Log best parameters and results
            mlflow.log_params(study.best_params)
            mlflow.log_metric('best_mAP50', study.best_value)
            
            # Save study results
            results = {
                'best_params': study.best_params,
                'best_value': study.best_value,
                'n_trials': len(study.trials),
                'study_name': study_name,
                'run_id': run.info.run_id
            }
            
            results_path = self.tuning_dir / f"{study_name}_results.json"
            with open(results_path, 'w') as f:
                json.dump(results, f, indent=4)
            
            # Generate optimization history plot
            self._plot_optimization_history(study, results_path.parent / f"{study_name}_history.png")
            
            return results
    
    def _plot_optimization_history(self, study: optuna.Study, save_path: Path) -> None:
        """
        Plot optimization history.
        Args:
            study: Optuna study object
            save_path: Path to save the plot
        """
        import matplotlib.pyplot as plt
        
        # Get history of values
        values = [t.value for t in study.trials]
        best_values = np.maximum.accumulate(values)
        
        plt.figure(figsize=(10, 6))
        plt.plot(values, label='Trial Value')
        plt.plot(best_values, label='Best Value')
        plt.xlabel('Trial Number')
        plt.ylabel('mAP50')
        plt.title('Hyperparameter Optimization History')
        plt.legend()
        plt.grid(True)
        plt.savefig(save_path)
        plt.close()
    
    def get_tuning_history(self) -> List[Dict]:
        """
        Get history of all tuning experiments.
        Returns:
            List of dictionaries containing tuning results
        """
        history = []
        for results_file in self.tuning_dir.glob('*_results.json'):
            with open(results_file, 'r') as f:
                history.append(json.load(f))
        return history
    
    def apply_best_parameters(self, study_name: str, data_yaml: str) -> str:
        """
        Create a training configuration using the best parameters from a study.
        Args:
            study_name: Name of the study to use
            data_yaml: Path to data.yaml file
        Returns:
            Path to created configuration file
        """
        results_path = self.tuning_dir / f"{study_name}_results.json"
        with open(results_path, 'r') as f:
            results = json.load(f)
        
        config_path = self.training_manager.create_training_config(
            data_yaml_path=data_yaml,
            **results['best_params']
        )
        
        return config_path
