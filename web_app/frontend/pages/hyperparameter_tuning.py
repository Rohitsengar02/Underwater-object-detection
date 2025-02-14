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
from src.training.hyperparameter_tuner import HyperparameterTuner

def main():
    st.set_page_config(
        page_title="Hyperparameter Tuning",
        page_icon="🎯",
        layout="wide"
    )

    st.title("🎯 Hyperparameter Tuning")

    # Initialize tuner
    base_dir = Path(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))
    tuner = HyperparameterTuner(str(base_dir))

    # Sidebar navigation
    page = st.sidebar.radio(
        "Navigation",
        ["Configure Tuning", "Tuning History", "Apply Best Parameters"]
    )

    if page == "Configure Tuning":
        show_tuning_configuration(tuner)
    elif page == "Tuning History":
        show_tuning_history(tuner)
    else:
        show_apply_best_parameters(tuner)

def show_tuning_configuration(tuner):
    st.header("Configure Hyperparameter Tuning")

    # Basic configuration
    st.subheader("Basic Configuration")
    data_yaml = st.text_input("Data YAML Path")
    n_trials = st.number_input("Number of Trials", min_value=1, value=20)
    timeout = st.number_input("Timeout (seconds, 0 for no timeout)", min_value=0, value=0)
    timeout = None if timeout == 0 else timeout

    # Parameter space configuration
    st.subheader("Parameter Space Configuration")
    st.write("Configure the range of values to explore for each hyperparameter:")

    col1, col2 = st.columns(2)

    with col1:
        # Learning rate range
        st.write("Learning Rate Range")
        lr_min = st.number_input("Min Learning Rate", min_value=1e-6, max_value=1.0, value=1e-5, format="%.6f")
        lr_max = st.number_input("Max Learning Rate", min_value=1e-6, max_value=1.0, value=1e-1, format="%.6f")

        # Batch size options
        st.write("Batch Size Options")
        batch_sizes = st.multiselect(
            "Select batch sizes to try",
            options=[8, 16, 32, 64, 128],
            default=[16, 32, 64]
        )

    with col2:
        # Image size options
        st.write("Image Size Options")
        img_sizes = st.multiselect(
            "Select image sizes to try",
            options=[416, 512, 640, 768],
            default=[640]
        )

        # Augmentation ranges
        st.write("Augmentation Ranges")
        mosaic_range = st.slider("Mosaic Range", 0.0, 1.0, (0.0, 1.0))
        mixup_range = st.slider("Mixup Range", 0.0, 1.0, (0.0, 1.0))

    # Create parameter space
    parameter_space = {
        'learning_rate': {
            'type': 'float',
            'low': lr_min,
            'high': lr_max,
            'log': True
        },
        'batch_size': {
            'type': 'categorical',
            'choices': batch_sizes
        },
        'img_size': {
            'type': 'categorical',
            'choices': img_sizes
        },
        'mosaic': {
            'type': 'float',
            'low': mosaic_range[0],
            'high': mosaic_range[1]
        },
        'mixup': {
            'type': 'float',
            'low': mixup_range[0],
            'high': mixup_range[1]
        }
    }

    if st.button("Start Tuning"):
        if not data_yaml:
            st.error("Please provide the path to data.yaml")
            return

        try:
            with st.spinner("Creating tuning configuration..."):
                config_path = tuner.create_tuning_config(
                    data_yaml_path=data_yaml,
                    n_trials=n_trials,
                    timeout=timeout,
                    parameter_space=parameter_space
                )
                
                st.success(f"Configuration created at: {config_path}")
                
                if st.button("Run Tuning"):
                    with st.spinner("Running hyperparameter tuning..."):
                        results = tuner.run_tuning(config_path)
                        
                        st.success("Tuning completed successfully!")
                        st.subheader("Best Parameters:")
                        st.json(results['best_params'])
                        
                        st.subheader("Best Performance:")
                        st.metric("mAP50", f"{results['best_value']:.4f}")
                        
                        # Show optimization history plot
                        st.subheader("Optimization History")
                        history_path = Path(config_path).parent / f"{results['study_name']}_history.png"
                        if history_path.exists():
                            st.image(str(history_path))

        except Exception as e:
            st.error(f"Error during tuning: {str(e)}")

def show_tuning_history(tuner):
    st.header("Tuning History")

    try:
        history = tuner.get_tuning_history()
        if history:
            # Convert history to DataFrame
            df = pd.DataFrame(history)
            
            # Display summary metrics
            st.subheader("Summary Statistics")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total Experiments", len(history))
            with col2:
                st.metric("Best mAP50", f"{max(h['best_value'] for h in history):.4f}")
            with col3:
                st.metric("Average mAP50", f"{sum(h['best_value'] for h in history)/len(history):.4f}")

            # Plot performance distribution
            st.subheader("Performance Distribution")
            fig = px.histogram(df, x='best_value', nbins=20, title="Distribution of Best mAP50 Scores")
            st.plotly_chart(fig)

            # Display detailed results
            st.subheader("Detailed Results")
            for i, result in enumerate(history):
                with st.expander(f"Experiment {i+1} - mAP50: {result['best_value']:.4f}"):
                    st.json(result)

        else:
            st.info("No tuning history found.")

    except Exception as e:
        st.error(f"Error loading tuning history: {str(e)}")

def show_apply_best_parameters(tuner):
    st.header("Apply Best Parameters")

    # Get available studies
    try:
        history = tuner.get_tuning_history()
        if not history:
            st.info("No tuning history available.")
            return

        # Select study
        study_names = [h['study_name'] for h in history]
        selected_study = st.selectbox("Select Study", study_names)

        # Input for new data.yaml
        data_yaml = st.text_input("Data YAML Path for New Training")

        if st.button("Apply Parameters"):
            if not data_yaml:
                st.error("Please provide the path to data.yaml")
                return

            try:
                config_path = tuner.apply_best_parameters(selected_study, data_yaml)
                st.success(f"Created new training configuration with best parameters at: {config_path}")
                
                # Show best parameters
                with open(config_path, 'r') as f:
                    config = json.load(f)
                st.subheader("Training Configuration")
                st.json(config)

            except Exception as e:
                st.error(f"Error applying parameters: {str(e)}")

    except Exception as e:
        st.error(f"Error loading studies: {str(e)}")

if __name__ == "__main__":
    main()
