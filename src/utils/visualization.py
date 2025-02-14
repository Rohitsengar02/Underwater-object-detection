import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import List, Dict, Optional, Tuple
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import json
import pandas as pd
from datetime import datetime

class TrainingVisualizer:
    def __init__(self, base_dir: str):
        """
        Initialize training visualizer.
        Args:
            base_dir: Base directory for visualization outputs
        """
        self.base_dir = Path(base_dir)
        self.viz_dir = self.base_dir / 'visualizations'
        self.viz_dir.mkdir(parents=True, exist_ok=True)
        
        # Set style for matplotlib
        plt.style.use('seaborn')
        
    def create_training_dashboard(self, 
                                metrics: Dict[str, List[float]],
                                save_path: Optional[str] = None) -> Dict:
        """
        Create an interactive training dashboard.
        Args:
            metrics: Dictionary of metric names and their values
            save_path: Optional path to save the dashboard
        Returns:
            Dictionary containing plotly figures
        """
        figures = {}
        
        # Create metric traces
        for metric_name, values in metrics.items():
            fig = go.Figure()
            
            # Add value trace
            fig.add_trace(go.Scatter(
                y=values,
                mode='lines+markers',
                name=metric_name,
                line=dict(width=2),
                marker=dict(size=6)
            ))
            
            # Add moving average
            window = min(10, len(values))
            if window > 1:
                moving_avg = np.convolve(values, np.ones(window)/window, mode='valid')
                fig.add_trace(go.Scatter(
                    y=moving_avg,
                    mode='lines',
                    name=f'{metric_name} (MA)',
                    line=dict(width=2, dash='dash')
                ))
            
            fig.update_layout(
                title=f'{metric_name} Over Time',
                xaxis_title='Epoch',
                yaxis_title=metric_name,
                hovermode='x unified'
            )
            
            figures[metric_name] = fig
            
        if save_path:
            # Save figures as HTML
            dashboard_path = Path(save_path)
            for metric_name, fig in figures.items():
                fig.write_html(str(dashboard_path / f"{metric_name.lower()}_plot.html"))
        
        return figures
    
    def plot_learning_curves(self,
                           train_metrics: Dict[str, List[float]],
                           val_metrics: Dict[str, List[float]],
                           save_path: Optional[str] = None) -> Dict:
        """
        Plot learning curves comparing training and validation metrics.
        Args:
            train_metrics: Dictionary of training metrics
            val_metrics: Dictionary of validation metrics
            save_path: Optional path to save the plots
        Returns:
            Dictionary containing plotly figures
        """
        figures = {}
        
        for metric_name in train_metrics.keys():
            if metric_name in val_metrics:
                fig = go.Figure()
                
                # Add training trace
                fig.add_trace(go.Scatter(
                    y=train_metrics[metric_name],
                    mode='lines+markers',
                    name=f'Train {metric_name}',
                    line=dict(width=2)
                ))
                
                # Add validation trace
                fig.add_trace(go.Scatter(
                    y=val_metrics[metric_name],
                    mode='lines+markers',
                    name=f'Val {metric_name}',
                    line=dict(width=2, dash='dash')
                ))
                
                fig.update_layout(
                    title=f'{metric_name} Learning Curve',
                    xaxis_title='Epoch',
                    yaxis_title=metric_name,
                    hovermode='x unified'
                )
                
                figures[metric_name] = fig
                
        if save_path:
            # Save figures as HTML
            curves_path = Path(save_path)
            for metric_name, fig in figures.items():
                fig.write_html(str(curves_path / f"{metric_name.lower()}_learning_curve.html"))
        
        return figures
    
    def plot_batch_metrics(self,
                          batch_metrics: Dict[str, List[float]],
                          epoch: int,
                          save_path: Optional[str] = None) -> Dict:
        """
        Plot metrics at batch level for a specific epoch.
        Args:
            batch_metrics: Dictionary of batch-level metrics
            epoch: Current epoch number
            save_path: Optional path to save the plots
        Returns:
            Dictionary containing plotly figures
        """
        figures = {}
        
        for metric_name, values in batch_metrics.items():
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                y=values,
                mode='lines',
                name=metric_name,
                line=dict(width=2)
            ))
            
            fig.update_layout(
                title=f'{metric_name} per Batch (Epoch {epoch})',
                xaxis_title='Batch',
                yaxis_title=metric_name,
                hovermode='x unified'
            )
            
            figures[metric_name] = fig
            
        if save_path:
            # Save figures as HTML
            batch_path = Path(save_path) / f"epoch_{epoch}"
            batch_path.mkdir(exist_ok=True)
            for metric_name, fig in figures.items():
                fig.write_html(str(batch_path / f"{metric_name.lower()}_batch_plot.html"))
        
        return figures
    
    def create_performance_summary(self,
                                 metrics: Dict[str, float],
                                 save_path: Optional[str] = None) -> go.Figure:
        """
        Create a performance summary visualization.
        Args:
            metrics: Dictionary of metric names and their final values
            save_path: Optional path to save the summary
        Returns:
            Plotly figure object
        """
        fig = go.Figure()
        
        # Create gauge charts for each metric
        for i, (metric_name, value) in enumerate(metrics.items()):
            fig.add_trace(go.Indicator(
                mode="gauge+number",
                value=value,
                domain={'row': 0, 'column': i},
                title={'text': metric_name},
                gauge={
                    'axis': {'range': [0, 1]},
                    'steps': [
                        {'range': [0, 0.5], 'color': "lightgray"},
                        {'range': [0.5, 0.8], 'color': "gray"},
                        {'range': [0.8, 1], 'color': "darkgray"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': value
                    }
                }
            ))
        
        fig.update_layout(
            grid={'rows': 1, 'columns': len(metrics)},
            title="Model Performance Summary"
        )
        
        if save_path:
            fig.write_html(str(Path(save_path) / "performance_summary.html"))
        
        return fig
    
    def plot_confusion_matrix(self,
                            confusion_matrix: np.ndarray,
                            class_names: List[str],
                            save_path: Optional[str] = None) -> go.Figure:
        """
        Create an interactive confusion matrix plot.
        Args:
            confusion_matrix: Numpy array of confusion matrix
            class_names: List of class names
            save_path: Optional path to save the plot
        Returns:
            Plotly figure object
        """
        fig = px.imshow(
            confusion_matrix,
            labels=dict(x="Predicted", y="True", color="Count"),
            x=class_names,
            y=class_names,
            aspect="auto",
            title="Confusion Matrix"
        )
        
        # Add text annotations
        for i in range(len(class_names)):
            for j in range(len(class_names)):
                fig.add_annotation(
                    x=j,
                    y=i,
                    text=str(confusion_matrix[i, j]),
                    showarrow=False,
                    font=dict(color="white" if confusion_matrix[i, j] > confusion_matrix.max()/2 else "black")
                )
        
        if save_path:
            fig.write_html(str(Path(save_path) / "confusion_matrix.html"))
        
        return fig
    
    def create_training_report(self,
                             run_id: str,
                             metrics: Dict[str, List[float]],
                             final_metrics: Dict[str, float],
                             confusion_matrix: Optional[np.ndarray] = None,
                             class_names: Optional[List[str]] = None) -> str:
        """
        Create a comprehensive training report.
        Args:
            run_id: Training run ID
            metrics: Dictionary of training metrics over time
            final_metrics: Dictionary of final metric values
            confusion_matrix: Optional confusion matrix
            class_names: Optional list of class names
        Returns:
            Path to generated report
        """
        report_dir = self.viz_dir / f"report_{run_id}"
        report_dir.mkdir(exist_ok=True)
        
        # Create visualizations
        dashboard_figures = self.create_training_dashboard(metrics, str(report_dir))
        performance_summary = self.create_performance_summary(final_metrics, str(report_dir))
        
        if confusion_matrix is not None and class_names is not None:
            confusion_fig = self.plot_confusion_matrix(
                confusion_matrix, class_names, str(report_dir)
            )
        
        # Create HTML report
        report_content = f"""
        <html>
            <head>
                <title>Training Report - {run_id}</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 20px; }}
                    .section {{ margin: 20px 0; padding: 20px; border: 1px solid #ddd; }}
                    .metric-value {{ font-size: 1.2em; font-weight: bold; }}
                </style>
            </head>
            <body>
                <h1>Training Report - {run_id}</h1>
                <div class="section">
                    <h2>Training Summary</h2>
                    <p>Run ID: {run_id}</p>
                    <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                </div>
                
                <div class="section">
                    <h2>Final Metrics</h2>
                    {''.join(f'<p>{k}: <span class="metric-value">{v:.4f}</span></p>' for k, v in final_metrics.items())}
                </div>
                
                <div class="section">
                    <h2>Training Curves</h2>
                    {''.join(f'<iframe src="{k.lower()}_plot.html" width="100%" height="500px"></iframe>' for k in metrics.keys())}
                </div>
                
                <div class="section">
                    <h2>Performance Summary</h2>
                    <iframe src="performance_summary.html" width="100%" height="500px"></iframe>
                </div>
                
                {'<div class="section"><h2>Confusion Matrix</h2><iframe src="confusion_matrix.html" width="100%" height="500px"></iframe></div>' if confusion_matrix is not None else ''}
            </body>
        </html>
        """
        
        report_path = report_dir / "report.html"
        with open(report_path, 'w') as f:
            f.write(report_content)
        
        return str(report_path)
