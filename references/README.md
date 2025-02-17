# Wind Power Forecasting Project

This project helps you build and evaluate machine learning models to forecast wind power generation in Orkney using weather data. It includes data pipeline setup, model training, and experiment tracking using MLFlow.

## Project Structure

- `Assignment_1_Template_MLFlow_Project.py`: Main script for MLFlow project execution. You should edit this file for submission.
- `Assignment_1-Template.ipynb`: Jupyter notebook with detailed code explanations. Just to play around, idally not intended for submission, as it is not compatible with MLFlow projects.
- `python_env.yaml`: Python environment configuration
- `requirements.txt`: Project dependencies
- `MLproject`: MLFlow project configuration

## Setup

1. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Project Overview

This project demonstrates how to:

1. Connect to InfluxDB to retrieve wind and power generation data
2. Process and analyze time series data using pandas
3. Build ML pipelines with scikit-learn
4. Track experiments using MLFlow
5. Deploy models for production use

## Data Sources

The project uses two main data sources from InfluxDB:
- Wind forecast data (speed and direction)
- Power generation data from Orkney

## Model Training

The project includes:
- Time series cross-validation
- Feature engineering pipeline
- Model evaluation metrics (MAE, MSE, R²)
- Experiment tracking with MLFlow

## MLFlow Integration

MLFlow is used for:
- Experiment tracking
- Model versioning
- Parameter logging
- Metric visualization
- Model deployment

## Usage

1. Start the MLFlow server:
   ```bash
   mlflow server
   ```

2. Run the main script:
   ```bash
   python Assignment_1_Template_MLFlow_Project.py
   ```

3. View experiments at http://localhost:5000

## Requirements

See `requirements.txt` for full list of dependencies. Key requirements:
- Python 3.11+
- MLFlow
- scikit-learn
- pandas
- InfluxDB client
