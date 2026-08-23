# Customer Satisfaction Prediction — MLOps with ZenML and MLflow

## Overview

This project implements an end-to-end Machine Learning and MLOps workflow for predicting customer satisfaction for a future purchase.

The project uses the **Brazilian E-Commerce Public Dataset by Olist**, which contains information about approximately 100,000 orders made between 2016 and 2018. The dataset includes information related to customers, orders, products, payments, freight, delivery, and customer reviews.

The objective is to use historical customer and order information to predict the review score or satisfaction level associated with a future purchase.

The project focuses not only on developing a machine learning model, but also on building a reproducible and production-oriented ML workflow using **ZenML** and **MLflow**.

---

## Project Status

> **Status: Completed Core MLOps Pipeline**

The current implementation includes:

- [x] Data ingestion and preprocessing
- [x] Feature preparation
- [x] Model training
- [x] MLflow experiment tracking
- [x] Model evaluation
- [x] MLflow model registry
- [x] Champion model alias
- [x] ZenML pipeline orchestration
- [x] Deployment validation
- [x] MLflow model deployment
- [x] Local model serving
- [x] Prediction pipeline
- [x] PSI-based data drift monitoring
- [x] Drift report generation
- [x] MLflow logging of drift metrics
- [x] Streamlit application integration

### Current Drift Monitoring Result

The deployed monitoring pipeline successfully compares incoming inference data against the reference profile associated with the **champion model**.

Example monitoring result from the current pipeline:

| Feature | PSI | Status |
|---|---:|---|
| payment_sequential | 0.000000 | LOW |
| payment_installments | 2.518670 | HIGH |
| payment_value | 0.082873 | LOW |
| price | 0.088715 | LOW |
| freight_value | 0.449461 | HIGH |
| product_name_lenght | 0.047780 | LOW |
| product_description_lenght | 0.059682 | LOW |
| product_photos_qty | 1.241205 | HIGH |
| product_weight_g | 0.100460 | MEDIUM |
| product_length_cm | 0.103283 | MEDIUM |
| product_height_cm | 0.206259 | MEDIUM |
| product_width_cm | 0.109061 | MEDIUM |

The current test run detected significant drift in:

- `payment_installments`
- `freight_value`
- `product_photos_qty`

This demonstrates that the project is capable of detecting changes in incoming production data before they silently affect model behavior.

---

## Problem Statement

For a given customer's historical information, the objective is to predict the satisfaction score for their next purchase.

Customer satisfaction can be influenced by several factors, including:

- Order status
- Product price
- Payment information
- Freight cost
- Delivery performance
- Customer location
- Product characteristics
- Historical customer behavior

The model uses relevant features derived from these factors to predict the expected customer satisfaction score.

In a real-world e-commerce environment, such a system could be used to identify customers who may be dissatisfied and enable businesses to take proactive measures.

---

## Objectives

The main objectives of this project are:

- Load and process the Olist e-commerce dataset.
- Perform data cleaning and preprocessing.
- Engineer features suitable for machine learning.
- Train a customer satisfaction prediction model.
- Track experiments, parameters, and metrics using MLflow.
- Evaluate model performance.
- Implement reproducible ML pipelines using ZenML.
- Validate newly trained models before deployment.
- Register and version models using MLflow.
- Maintain a champion model using MLflow aliases.
- Deploy models using the MLflow model deployer.
- Monitor incoming data for distribution drift.
- Log drift metrics and reports to MLflow.
- Provide predictions through a Streamlit application.
- Establish a foundation for continuous model training and deployment.

---

# System Architecture

```text
                         OLIST DATASET
                              |
                              v
                       DATA INGESTION
                              |
                              v
                       DATA CLEANING
                              |
                              v
                    FEATURE ENGINEERING
                              |
                              v
                       MODEL TRAINING
                              |
                              v
                  MLFLOW EXPERIMENT TRACKING
                              |
                              v
                       MODEL EVALUATION
                              |
                              v
                     DEPLOYMENT VALIDATION
                              |
                    +---------+---------+
                    |                   |
                  PASS                 FAIL
                    |                   |
                    v                   v
             MODEL REGISTRATION     REJECT MODEL
                    |
                    v
             CHAMPION MODEL
                    |
                    v
             MLFLOW DEPLOYMENT
                    |
                    v
             MODEL SERVING API
                    |
          +---------+---------+
          |                   |
          v                   v
   DRIFT MONITORING      STREAMLIT APP
          |                   |
          v                   v
      PSI REPORT         PREDICTION
          |                   |
          +---------+---------+
                    |
                    v
          CUSTOMER SATISFACTION
              PREDICTION
```

---

# Technology Stack

| Technology | Purpose |
|---|---|
| Python | Machine learning and pipeline development |
| Pandas | Data processing and manipulation |
| NumPy | Numerical operations |
| Scikit-learn | Machine learning and evaluation |
| ZenML | ML pipeline orchestration |
| MLflow | Experiment tracking, model registry and deployment |
| Streamlit | Prediction application |
| Git | Version control |
| GitHub | Source code management |
| SQLite | Local MLflow backend |
| PSI | Data drift monitoring |

---

# Project Workflow

## 1. Data Ingestion

The raw Olist dataset is loaded and converted into a structured format for further processing.

```text
Raw Dataset
     |
     v
Data Ingestion
     |
     v
DataFrame
```

---

## 2. Data Cleaning

The raw data contains multiple tables and fields that require preprocessing.

The cleaning stage includes:

- Handling missing values
- Removing unnecessary columns
- Selecting relevant features
- Converting data types where required
- Preparing data for model training
- Ensuring consistent feature formats

---

## 3. Feature Engineering

Information from orders, customers, products, payments, freight, and delivery is transformed into features that can be used by the machine learning model.

The current inference pipeline works with features including:

```text
payment_sequential
payment_installments
payment_value
price
freight_value
product_name_lenght
product_description_lenght
product_photos_qty
product_weight_g
product_length_cm
product_height_cm
product_width_cm
```

The objective is to convert business-related information into meaningful numerical features for customer satisfaction prediction.

---

## 4. Model Training

The processed dataset is used to train the machine learning model.

MLflow is integrated into the training process to track:

- Model parameters
- Training metrics
- Evaluation metrics
- Model artifacts
- Training runs
- Model versions

This allows different experiments and model configurations to be compared systematically.

---

## 5. Model Evaluation

The trained model is evaluated using appropriate performance metrics.

Depending on the final problem formulation, evaluation metrics may include:

- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- Mean Absolute Error (MAE)
- R² Score

The evaluation results are used to determine whether the model meets the criteria required for deployment.

---

# ZenML Pipeline

ZenML is used to organize the machine learning workflow into reproducible pipeline steps.

A typical training pipeline consists of:

```text
ingest_data
     |
     v
clean_data
     |
     v
feature_engineering
     |
     v
train_model
     |
     v
evaluate_model
```

Each step performs a specific task and produces artifacts that can be consumed by subsequent steps.

---

# Continuous Deployment Pipeline

The continuous deployment pipeline extends the training workflow by adding model validation and deployment.

```text
Data Ingestion
      |
      v
Data Cleaning
      |
      v
Feature Engineering
      |
      v
Model Training
      |
      v
Model Evaluation
      |
      v
Deployment Trigger
      |
      v
Model Validation
      |
      +----------------+
      |                |
     PASS             FAIL
      |                |
      v                v
Deploy Model       Reject Model
      |
      v
MLflow Model Service
      |
      v
Prediction
```

The newly trained model is evaluated before deployment.

If the model satisfies the configured performance criteria, it can replace the currently deployed model. Otherwise, the existing model remains in use.

---

# Deployment Trigger

The deployment trigger acts as a quality gate for the machine learning pipeline.

```text
New Model
    |
    v
Model Evaluation
    |
    v
Performance Threshold
    |
    +-------------------+
    |                   |
   PASS                FAIL
    |                   |
    v                   v
Deploy Model        Reject Model
```

This prevents a model with inferior performance from automatically replacing an existing production model.

---

# MLflow Integration

MLflow is used for experiment tracking and model management.

For each training run, MLflow can track:

```text
Parameters
     |
     v
Metrics
     |
     v
Artifacts
     |
     v
Trained Model
     |
     v
Registered Model
```

MLflow provides:

- Experiment tracking
- Parameter logging
- Metric logging
- Artifact management
- Model versioning
- Model registry
- Model aliases
- Model deployment

The current project uses the registered model:

```text
Customer-Satisfaction-Model
```

with the deployment alias:

```text
champion
```

The drift monitoring pipeline retrieves the model version associated with this alias and uses its reference profile for monitoring.

---

# ZenML and MLflow

ZenML and MLflow serve different but complementary purposes.

## ZenML

ZenML is responsible for:

- Pipeline orchestration
- Pipeline step management
- Pipeline execution
- Artifact management
- MLOps integration
- Reproducible workflow execution

## MLflow

MLflow is responsible for:

- Experiment tracking
- Parameter logging
- Metric tracking
- Artifact management
- Model registry
- Model versioning
- Model aliases
- Model deployment

The overall workflow can therefore be represented as:

```text
                         ZENML
                           |
                           v
                  Pipeline Orchestration
                           |
             +-------------+-------------+
             |             |             |
             v             v             v
           Data         Training     Evaluation
             |             |             |
             +-------------+-------------+
                           |
                           v
                         MLFLOW
                           |
                           v
                  Experiment Tracking
                           |
                           v
                    Model Registry
                           |
                           v
                     Model Version
                           |
                           v
                       Champion
                           |
                           v
                    Model Deployment
```

---

# Data Drift Monitoring

A dedicated `drift_monitor` step has been implemented using **Population Stability Index (PSI)**.

The monitoring workflow is:

```text
Incoming Inference Data
          |
          v
Prepare Current Data
          |
          v
Load Champion Model
          |
          v
Download Reference Profile
          |
          v
Reconstruct Reference Distribution
          |
          v
Calculate PSI
          |
          v
Classify Drift
          |
          +-----------------------------+
          |             |               |
          v             v               v
         LOW          MEDIUM           HIGH
          |             |               |
          +-------------+---------------+
                        |
                        v
                 Drift Report
                        |
                        v
                  MLflow Logging
```

## PSI Thresholds

| PSI Value | Classification | Meaning |
|---:|---|---|
| `< 0.10` | LOW | Little or no significant drift |
| `0.10 – < 0.25` | MEDIUM | Moderate drift |
| `>= 0.25` | HIGH | Significant drift |

The monitor:

- Loads the champion model from MLflow.
- Retrieves its reference profile.
- Compares reference and current feature distributions.
- Calculates PSI for every feature.
- Classifies features into LOW, MEDIUM, or HIGH drift.
- Detects missing features.
- Handles empty or invalid feature data.
- Generates a structured JSON drift report.
- Logs feature-level PSI values to MLflow.
- Logs aggregate drift metrics to MLflow.

The generated report is stored as:

```text
monitoring/drift_report.json
```

---

# Current Drift Monitoring Output

A successful monitoring run currently reports:

```text
Reference features : 12
Current features   : 12

payment_sequential                  PSI=0.000000 | LOW
payment_installments                PSI=2.518670 | HIGH
payment_value                       PSI=0.082873 | LOW
price                               PSI=0.088715 | LOW
freight_value                       PSI=0.449461 | HIGH
product_name_lenght                 PSI=0.047780 | LOW
product_description_lenght          PSI=0.059682 | LOW
product_photos_qty                  PSI=1.241205 | HIGH
product_weight_g                    PSI=0.100460 | MEDIUM
product_length_cm                   PSI=0.103283 | MEDIUM
product_height_cm                   PSI=0.206259 | MEDIUM
product_width_cm                    PSI=0.109061 | MEDIUM
```

Summary:

```text
Mean PSI       : 0.417287
Maximum PSI    : 2.518670
Low Drift      : 5
Medium Drift   : 4
High Drift     : 3
Missing        : 0
Errors         : 0
No Data        : 0
Drift Detected : YES
```

High-drift features:

```text
payment_installments
freight_value
product_photos_qty
```

This confirms that the drift monitoring component is functioning successfully.

---

# Streamlit Application

A Streamlit application provides an interface for interacting with the deployed model.

The application accepts relevant order and product information and sends the input to the deployed model.

```text
User Input
    |
    v
Streamlit Application
    |
    v
MLflow Model Service
    |
    v
Prediction
    |
    v
Customer Satisfaction Score
```

The application is designed to use the latest model deployed by the pipeline.

Run the application using:

```bash
streamlit run streamlit_app.py
```

---

# Model Serving

The project currently uses the ZenML MLflow model deployer.

The local prediction service is exposed through an endpoint similar to:

```text
http://127.0.0.1:8002/invocations
```

The prediction pipeline sends data using the MLflow `dataframe_split` format.

Example:

```text
Payload rows    : 100
Payload columns : 12
```

The model successfully returns customer satisfaction predictions for the incoming records.

---

# Project Structure

```text
CUSTOMER-SATISFACTION-MLOPS/
│
├── data/
│   └── README.md
│
├── steps/
│   ├── ingest_data.py
│   ├── clean_data.py
│   ├── train_model.py
│   ├── evaluation.py
│   └── drift_monitor.py
│
├── pipelines/
│   ├── training_pipeline.py
│   └── deployment_pipeline.py
│
├── monitoring/
│   └── drift_report.json
│
├── run_pipeline.py
├── run_deployment.py
├── streamlit_app.py
├── requirements.txt
├── README.md
└── .gitignore
```

The project structure may evolve as additional MLOps components are introduced.

---

# Installation

## 1. Clone the Repository

```bash
git clone https://github.com/ansh-rohilla/Customer-Satisfaction-MLops.git
cd Customer-Satisfaction-MLops
```

## 2. Create a Virtual Environment

```bash
python -m venv .venv
```

### macOS / Linux

```bash
source .venv/bin/activate
```

### Windows

```powershell
.venv\Scripts\activate
```

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

# ZenML Configuration

Install ZenML:

```bash
pip install zenml
```

For the ZenML server and dashboard:

```bash
pip install "zenml[server]"
```

Start the ZenML server:

```bash
zenml up
```

Install the MLflow integration:

```bash
zenml integration install mlflow -y
```

Register the MLflow experiment tracker:

```bash
zenml experiment-tracker register mlflow_tracker --flavor=mlflow
```

Register the MLflow model deployer:

```bash
zenml model-deployer register mlflow --flavor=mlflow
```

Create and activate the ZenML stack:

```bash
zenml stack register mlflow_stack \
    -a default \
    -o default \
    -d mlflow \
    -e mlflow_tracker \
    --set
```

Verify the active stack:

```bash
zenml stack describe
```

The expected stack contains:

```text
orchestrator: default
artifact_store: default
model_deployer: mlflow
experiment_tracker: mlflow_tracker
```

---

# MLflow Configuration

The project uses MLflow for experiment tracking and model management.

The local MLflow backend is configured through the project's `.env` file.

Example:

```env
MLFLOW_TRACKING_URI=sqlite:////Users/<username>/Library/Application Support/zenml/local_stores/<zenml-store-id>/mlflow.db
```

The exact path is machine-specific and should not be hard-coded when sharing the repository.

The current project verifies that:

```text
Customer-Satisfaction-Model
```

exists in the MLflow Model Registry and that the:

```text
champion
```

alias points to a valid model version.

---

# Running the Training Pipeline

Run the standard training pipeline:

```bash
python run_pipeline.py
```

The training pipeline performs operations such as:

```text
Data Ingestion
      |
      v
Data Cleaning
      |
      v
Feature Engineering
      |
      v
Model Training
      |
      v
Model Evaluation
      |
      v
MLflow Tracking
      |
      v
Model Registration
```

---

# Running the Deployment Pipeline

Execute the deployment workflow using:

```bash
python run_deployment.py
```

The deployment pipeline performs:

1. Data ingestion.
2. Data cleaning and preparation.
3. Model training.
4. Model evaluation.
5. Deployment validation.
6. Model registration.
7. Model deployment.
8. Drift monitoring.
9. Prediction.

To run prediction mode:

```bash
python run_deployment.py --config predict
```

A successful prediction run should show:

```text
Step drift_monitor has finished
Step predictor has started
Step predictor has finished
Pipeline run has finished
```

---

# MLflow Experiment Tracking

MLflow can be used to inspect and compare different training and monitoring runs.

Tracked information may include:

- Hyperparameters
- Training metrics
- Evaluation metrics
- Model artifacts
- Model versions
- Feature-level PSI
- Mean PSI
- Maximum PSI
- Drift counts
- Monitoring reports

Start the MLflow UI using the tracking database configured for the project:

```bash
mlflow ui --backend-store-uri '<MLFLOW_TRACKING_URI>'
```

For example:

```bash
mlflow ui --backend-store-uri 'sqlite:////Users/<username>/Library/Application Support/zenml/local_stores/<store-id>/mlflow.db'
```

Then open the MLflow UI in the browser.

---

# Important MLflow Model Registry Concepts

The project uses the following model registry structure:

```text
Customer-Satisfaction-Model
             |
             +---- Version 1
             |
             +---- Version 2
             |
             +---- Version 3
             |
             +---- ...
             |
             +---- Version 9
                    |
                    v
                 champion
```

The `champion` alias identifies the model version currently used for monitoring and deployment.

This makes the monitoring pipeline independent of a hard-coded model version.

---

# Data Management

The original Olist dataset is not committed to the repository because of its size.

The raw dataset should remain in the local `data/` directory and be excluded through `.gitignore`.

Example:

```gitignore
data/*.csv
data/*.zip
.venv/
__pycache__/
mlruns/
.zen/
.env
.DS_Store
```

The repository contains the code and configuration required to process the dataset.

---

# Dataset

This project uses the **Brazilian E-Commerce Public Dataset by Olist**.

The dataset contains approximately 100,000 orders from Brazilian e-commerce marketplaces between 2016 and 2018.

The dataset includes information about:

- Customers
- Orders
- Products
- Payments
- Reviews
- Sellers
- Freight
- Delivery
- Geographic locations

The dataset is publicly available through Kaggle.

---

# MLOps Concepts Demonstrated

This project demonstrates the following concepts:

- Machine learning pipelines
- Pipeline orchestration
- Data preprocessing
- Feature engineering
- Experiment tracking
- Model evaluation
- Model versioning
- Model registry
- Model aliases
- Model deployment
- Continuous deployment
- Reproducibility
- Model quality gates
- Data drift detection
- PSI-based monitoring
- Monitoring artifact generation
- ML lifecycle management

---

# End-to-End MLOps Workflow

A conventional machine learning workflow can be represented as:

```text
Dataset
   |
   v
Train Model
   |
   v
Evaluate Model
   |
   v
Done
```

The MLOps workflow implemented in this project extends the process:

```text
Dataset
   |
   v
Data Ingestion
   |
   v
Data Cleaning
   |
   v
Feature Engineering
   |
   v
Model Training
   |
   v
MLflow Experiment Tracking
   |
   v
Model Evaluation
   |
   v
Deployment Validation
   |
   v
Model Registration
   |
   v
Champion Model
   |
   v
Model Deployment
   |
   +--------------------+
   |                    |
   v                    v
Prediction          Drift Monitoring
   |                    |
   v                    v
Streamlit App      PSI Report
   |                    |
   +----------+---------+
              |
              v
     Production Monitoring
```

---

# Why ZenML + MLflow?

## Why ZenML?

ZenML provides a structured way to build and execute reproducible ML pipelines.

It helps separate the machine learning workflow into reusable steps such as:

```text
Ingestion
Cleaning
Training
Evaluation
Deployment
Monitoring
```

This makes the workflow easier to maintain and extend.

## Why MLflow?

MLflow provides the model lifecycle components required to track and manage machine learning models.

It provides:

```text
Experiment Tracking
        |
        v
Metric Tracking
        |
        v
Artifact Tracking
        |
        v
Model Registry
        |
        v
Model Versioning
        |
        v
Model Deployment
```

Together, ZenML and MLflow provide a practical MLOps architecture for this project.

---

# Future Improvements

The core project is functional, but several production-level improvements can be added:

- [ ] Docker-based deployment
- [ ] Automated unit and integration testing
- [ ] GitHub Actions CI/CD
- [ ] Automated data validation
- [ ] Automated model retraining
- [ ] Automated drift-triggered retraining
- [ ] Model performance monitoring
- [ ] Cloud deployment
- [ ] Kubernetes-based deployment
- [ ] Production database integration
- [ ] API gateway
- [ ] Authentication and authorization
- [ ] Centralized logging
- [ ] Alerting for high drift
- [ ] Scheduled monitoring
- [ ] Cloud-based MLflow server
- [ ] Production-grade artifact storage

These improvements are optional extensions for moving the project from a local MLOps demonstration toward a production-grade system.

---

# Learning Outcomes

This project demonstrates how a traditional machine learning workflow can be transformed into a reproducible MLOps system.

Through this project, the following practical concepts are demonstrated:

- Building reusable ML pipelines
- Using ZenML for pipeline orchestration
- Using MLflow for experiment tracking
- Managing models with the MLflow Model Registry
- Using model aliases such as `champion`
- Deploying models through MLflow
- Creating a local model-serving workflow
- Validating models before deployment
- Monitoring incoming data distributions
- Implementing PSI-based data drift detection
- Logging monitoring results to MLflow
- Integrating prediction and monitoring into one workflow

---

# Example Production Scenario

A simplified real-world scenario for this project is:

```text
Customer Places Order
        |
        v
Order Information
        |
        v
Feature Engineering
        |
        v
Customer Satisfaction Model
        |
        v
Predicted Satisfaction Score
        |
        +-----------------------+
        |                       |
        v                       v
High Satisfaction        Low Satisfaction
        |                       |
        v                       v
Normal Workflow          Proactive Intervention
```

At the same time, incoming production data can be monitored:

```text
Production Data
      |
      v
Drift Monitor
      |
      v
PSI Calculation
      |
      v
Drift Classification
      |
      +----------------------------+
      |            |               |
     LOW         MEDIUM           HIGH
      |            |               |
      v            v               v
  Continue      Monitor       Investigate /
  Normally      Closely       Retrain Model
```

This provides the foundation for a complete machine learning lifecycle.

---

# Repository

GitHub Repository:

**https://github.com/ansh-rohilla/Customer-Satisfaction-MLops**

---

# Author

**Ansh Rohilla**

B.Tech Computer Science Engineering  
Artificial Intelligence and Edge Computing

GitHub:  
https://github.com/ansh-rohilla

---

# Acknowledgements

- **Olist** for the Brazilian E-Commerce Public Dataset.
- **ZenML** for the MLOps pipeline orchestration framework.
- **MLflow** for experiment tracking, model management, and deployment.
- **Scikit-learn** for machine learning utilities.
- **Streamlit** for the prediction interface.

---

# Conclusion

The Customer Satisfaction Prediction project demonstrates an end-to-end MLOps workflow that goes beyond simply training a machine learning model.

The system integrates:

```text
Data
 |
 v
Preprocessing
 |
 v
Feature Engineering
 |
 v
Model Training
 |
 v
MLflow Tracking
 |
 v
Model Evaluation
 |
 v
Model Registry
 |
 v
Deployment Validation
 |
 v
Model Deployment
 |
 v
Prediction
 |
 v
Data Drift Monitoring
```

The current implementation successfully performs model deployment, prediction, MLflow model registry management, champion model selection, and PSI-based data drift monitoring.

The project therefore provides a practical foundation for building, deploying, and monitoring machine learning systems using modern MLOps practices.
