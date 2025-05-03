# Solar Power Output Forecasting

This project forecasts solar power generation (DC and AC power) using historical weather sensor data and generation data. It leverages a modern machine learning and MLOps workflow, including data preprocessing, feature engineering, model training, evaluation, and experiment tracking with MLflow.

## 📈 What This Project Does

- Cleans and merges generation and weather datasets from solar power plants
- Builds predictive features based on temporal and weather-related inputs
- Trains machine learning models to forecast power output
- Tracks model performance metrics with MLflow
- Generates plots for visual evaluation
- Includes anomaly detection for monitoring unexpected behavior

## 💡 How This Can Help

- Improve energy forecasting for grid integration
- Enable proactive maintenance by identifying anomalies
- Provide baseline predictions for smart solar applications
- Demonstrate a full ML lifecycle in an energy-focused MLOps project

## 🚀 How to Use

1. **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

2. **Run the full ML pipeline**:
    ```bash
    make prepare-data PLANT_ID=1
    make build-features PLANT_ID=1
    make split-data PLANT_ID=1
    make train-model PLANT_ID=1
    make predict PLANT_ID=1
    make plot PLANT_ID=1
    ```

3. **Start the MLflow UI**:
    ```bash
    mlflow ui
    ```

4. **Explore results**:
    - Trained model: `models/model_plant1.pkl`
    - Prediction plots: `reports/figures/plot_plant1.png`
    - MLflow dashboard at `http://localhost:5002`

## 📊 Dataset Source

The data used in this project is from the [Solar Power Generation Data](https://www.kaggle.com/datasets/anikannal/solar-power-generation-data) provided by the Indian Ministry of Power via Kaggle. It includes weather sensor data and power generation metrics from two solar power plants over 34 days.

---

This project is ideal for learning how to apply MLOps principles to time-series forecasting and energy analytics.
