import streamlit as st
import sys
import os
from pathlib import Path
import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# Add the src directory to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))
from src.training.training_manager import TrainingManager
from src.utils.dataset_manager import DatasetManager

def main():
    st.set_page_config(
        page_title="Model Training",
        page_icon="🚀",
        layout="wide"
    )

    st.title("🚀 Model Training and Management")

    # Initialize managers
    base_dir = Path(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))
    training_manager = TrainingManager(str(base_dir))
    dataset_manager = DatasetManager(str(base_dir))

    # Sidebar navigation
    page = st.sidebar.radio(
        "Navigation",
        ["Dataset Management", "Training Configuration", "Training History", "Model Evaluation"]
    )

    if page == "Dataset Management":
        show_dataset_management(dataset_manager)
    elif page == "Training Configuration":
        show_training_configuration(training_manager)
    elif page == "Training History":
        show_training_history(training_manager)
    else:
        show_model_evaluation(training_manager)

def show_dataset_management(dataset_manager):
    st.header("Dataset Management")

    # Dataset import
    st.subheader("Import Dataset")
    dataset_path = st.text_input("Dataset Path")
    dataset_name = st.text_input("Dataset Name")
    format_type = st.selectbox("Format Type", ["yolo", "coco"])

    if st.button("Import Dataset"):
        with st.spinner("Importing dataset..."):
            try:
                metadata = dataset_manager.import_dataset(dataset_path, dataset_name, format_type)
                st.success(f"Dataset imported successfully! Found {metadata['statistics']['image_count']} images.")
            except Exception as e:
                st.error(f"Error importing dataset: {str(e)}")

    # Dataset statistics
    st.subheader("Dataset Statistics")
    datasets = [d.name for d in dataset_manager.datasets_dir.iterdir() if d.is_dir()]
    if datasets:
        selected_dataset = st.selectbox("Select Dataset", datasets)
        if st.button("Generate Statistics"):
            with st.spinner("Generating dataset statistics..."):
                try:
                    stats = dataset_manager.generate_dataset_stats(selected_dataset)
                    
                    # Display image statistics
                    st.write("📊 Image Statistics")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write("Resolution Stats:")
                        st.write(stats['image_stats']['resolution_stats'])
                    with col2:
                        st.write("Aspect Ratio Stats:")
                        st.write(stats['image_stats']['aspect_ratio_stats'])

                    # Display annotation statistics
                    st.write("📑 Annotation Statistics")
                    st.write(stats['annotation_stats'])

                    # Plot class distribution
                    st.write("📈 Class Distribution")
                    class_dist = pd.DataFrame(
                        list(stats['class_distribution'].items()),
                        columns=['Class', 'Count']
                    )
                    fig = px.bar(class_dist, x='Class', y='Count')
                    st.plotly_chart(fig)

                except Exception as e:
                    st.error(f"Error generating statistics: {str(e)}")
    else:
        st.info("No datasets found. Please import a dataset first.")

def show_training_configuration(training_manager):
    st.header("Training Configuration")

    # Model selection
    st.subheader("Model Selection")
    model_name = st.selectbox(
        "Base Model",
        ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8l.pt", "yolov8x.pt"]
    )

    # Training parameters
    st.subheader("Training Parameters")
    col1, col2 = st.columns(2)

    with col1:
        epochs = st.number_input("Epochs", min_value=1, value=100)
        batch_size = st.number_input("Batch Size", min_value=1, value=16)
        img_size = st.number_input("Image Size", min_value=32, value=640, step=32)
        learning_rate = st.number_input("Learning Rate", min_value=0.0001, value=0.01, format="%.4f")

    with col2:
        patience = st.number_input("Early Stopping Patience", min_value=1, value=50)
        mosaic = st.slider("Mosaic Augmentation", min_value=0.0, max_value=1.0, value=1.0)
        mixup = st.slider("Mixup Augmentation", min_value=0.0, max_value=1.0, value=0.0)
        copy_paste = st.slider("Copy-Paste Augmentation", min_value=0.0, max_value=1.0, value=0.0)

    # Data configuration
    st.subheader("Data Configuration")
    data_yaml = st.text_input("Path to data.yaml")

    if st.button("Create Configuration"):
        try:
            config_path = training_manager.create_training_config(
                data_yaml_path=data_yaml,
                model_name=model_name,
                epochs=epochs,
                batch_size=batch_size,
                img_size=img_size,
                lr0=learning_rate,
                patience=patience,
                mosaic=mosaic,
                mixup=mixup,
                copy_paste=copy_paste
            )
            st.success(f"Configuration created successfully at: {config_path}")

            # Start training option
            if st.button("Start Training"):
                with st.spinner("Training in progress..."):
                    try:
                        results = training_manager.train_model(config_path)
                        st.success("Training completed successfully!")
                        st.write("📊 Training Results:")
                        st.json(results)
                    except Exception as e:
                        st.error(f"Error during training: {str(e)}")

        except Exception as e:
            st.error(f"Error creating configuration: {str(e)}")

def show_training_history(training_manager):
    st.header("Training History")

    try:
        history = training_manager.get_training_history()
        if history:
            # Convert history to DataFrame
            df = pd.DataFrame(history)
            df['start_time'] = pd.to_datetime(df['start_time'], unit='ms')
            df['end_time'] = pd.to_datetime(df['end_time'], unit='ms')
            df['duration'] = (df['end_time'] - df['start_time']).dt.total_seconds() / 60  # minutes

            # Display metrics over time
            st.subheader("Training Metrics Over Time")
            metrics = ['mAP50', 'mAP50-95', 'precision', 'recall']
            metric = st.selectbox("Select Metric", metrics)

            fig = px.line(df, x='start_time', y=f"metrics.{metric}", title=f"{metric} Over Time")
            st.plotly_chart(fig)

            # Display training runs table
            st.subheader("Training Runs")
            st.dataframe(df[['run_id', 'status', 'start_time', 'duration', f"metrics.{metric}"]])

        else:
            st.info("No training history found.")

    except Exception as e:
        st.error(f"Error loading training history: {str(e)}")

def show_model_evaluation(training_manager):
    st.header("Model Evaluation")

    # Model selection
    model_path = st.text_input("Model Path")
    data_yaml = st.text_input("Evaluation Data YAML")

    if st.button("Evaluate Model"):
        with st.spinner("Evaluating model..."):
            try:
                metrics = training_manager.evaluate_model(model_path, data_yaml)
                
                # Display metrics
                st.subheader("Evaluation Metrics")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("mAP50", f"{metrics['mAP50']:.3f}")
                    st.metric("mAP50-95", f"{metrics['mAP50-95']:.3f}")
                
                with col2:
                    st.metric("Precision", f"{metrics['precision']:.3f}")
                    st.metric("Recall", f"{metrics['recall']:.3f}")

                # Model information
                st.subheader("Model Information")
                model_info = training_manager.get_model_info(model_path)
                st.json(model_info)

            except Exception as e:
                st.error(f"Error evaluating model: {str(e)}")

    # Model export
    st.subheader("Export Model")
    export_format = st.selectbox("Export Format", ["onnx", "torchscript", "tflite"])
    
    if st.button("Export Model"):
        with st.spinner("Exporting model..."):
            try:
                exported_path = training_manager.export_model(model_path, format=export_format)
                st.success(f"Model exported successfully to: {exported_path}")
            except Exception as e:
                st.error(f"Error exporting model: {str(e)}")

if __name__ == "__main__":
    main()
