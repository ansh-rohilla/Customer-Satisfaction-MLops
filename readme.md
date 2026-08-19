# Customer Satisfaction Prediction — MLOps with ZenML and MLflow

## Overview

This project implements an end-to-end Machine Learning and MLOps workflow for predicting customer satisfaction for a future purchase.

The project uses the **Brazilian E-Commerce Public Dataset by Olist**, which contains information about approximately 100,000 orders made between 2016 and 2018. The dataset includes information related to customers, orders, products, payments, freight, delivery, and customer reviews.

The objective is to use historical customer and order information to predict the review score or satisfaction level associated with a future purchase.

The project focuses not only on developing a machine learning model but also on building a reproducible and production-oriented ML workflow using **ZenML** and **MLflow**.

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
- Implement a reproducible ML pipeline using ZenML.
- Validate newly trained models before deployment.
- Deploy models using the MLflow model deployer.
- Provide predictions through a Streamlit application.
- Establish a foundation for continuous model training and deployment.

---

## System Architecture

```text
Olist Dataset
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
      +----------------------+
      |                      |
   Pass                     Fail
      |                      |
      v                      v
Model Deployment       Reject Model
      |
      v
MLflow Model Service
      |
      v
Streamlit Application
      |
      v
Customer Satisfaction Prediction
```

---

## Technology Stack

| Technology | Purpose |
|---|---|
| Python | Machine learning and pipeline development |
| Pandas | Data processing and manipulation |
| NumPy | Numerical operations |
| Scikit-learn | Machine learning and evaluation |
| ZenML | ML pipeline orchestration |
| MLflow | Experiment tracking and model management |
| Streamlit | Prediction application |
| Git | Version control |
| GitHub | Source code management |

---

## Project Workflow

### 1. Data Ingestion

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

### 2. Data Cleaning

The raw data contains multiple tables and fields that require preprocessing.

The cleaning stage includes:

- Handling missing values
- Removing unnecessary columns
- Selecting relevant features
- Converting data types where required
- Preparing the data for model training

### 3. Feature Engineering

Information from orders, customers, products, payments, freight, and delivery is transformed into features that can be used by the machine learning model.

The objective is to convert business-related information into meaningful numerical and categorical features.

### 4. Model Training

The processed dataset is used to train the machine learning model.

MLflow is integrated into the training process to track:

- Model parameters
- Training metrics
- Evaluation metrics
- Model artifacts
- Training runs

This allows different experiments and model configurations to be compared systematically.

### 5. Model Evaluation

The trained model is evaluated using appropriate performance metrics.

Depending on the final problem formulation, evaluation metrics may include:

- Accuracy
- Precision
- Recall
- F1-score
- Mean Squared Error
- Mean Absolute Error

The evaluation results are used to determine whether the model meets the criteria required for deployment.

---

## ZenML Pipeline

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

## Continuous Deployment Pipeline

The continuous deployment pipeline extends the training pipeline by adding model validation and deployment steps.

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
Model Deployment
```

The newly trained model is evaluated before deployment.

If the model satisfies the configured performance criteria, it can replace the currently deployed model. Otherwise, the existing model remains in use.

---

## Deployment Trigger

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
    +----------------+
    |                |
   Pass             Fail
    |                |
    v                v
Deploy Model     Reject Model
```

This prevents a model with inferior performance from automatically replacing an existing production model.

---

## MLflow Integration

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
```

This provides a centralized way to compare experiments and identify the best-performing model.

MLflow is also used as the model deployment component within the ZenML stack.

---

## ZenML and MLflow

ZenML and MLflow serve different purposes within the project.

### ZenML

ZenML is responsible for:

- Pipeline orchestration
- Pipeline step management
- Pipeline execution
- Artifact management
- Integration with MLOps tools

### MLflow

MLflow is responsible for:

- Experiment tracking
- Parameter logging
- Metric tracking
- Model artifact management
- Model management
- Model deployment

The overall workflow can therefore be represented as:

```text
                    ZenML
                      |
              Pipeline Orchestration
                      |
        +-------------+-------------+
        |             |             |
        v             v             v
      Data         Training     Evaluation
                      |
                      v
                   MLflow
                      |
              Experiment Tracking
                      |
                      v
               Model Management
                      |
                      v
                Model Deployment
```

---

## Streamlit Application

A Streamlit application provides an interface for interacting with the deployed model.

The application accepts relevant order and product information and sends the input to the deployed model.

```text
User Input
    |
    v
Streamlit Application
    |
    v
Deployed Model
    |
    v
Prediction
    |
    v
Customer Satisfaction Score
```

The application is designed to use the latest model deployed by the pipeline.

---

## Project Structure

```text
CUSTOMER-SATISFACTION-MLOPS/
|
├── data/
|   └── README.md
|
├── steps/
|   ├── ingest_data.py
|   ├── clean_data.py
|   ├── train_model.py
|   └── evaluation.py
|
├── pipelines/
|   ├── training_pipeline.py
|   └── deployment_pipeline.py
|
├── run_pipeline.py
├── run_deployment.py
├── streamlit_app.py
├── requirements.txt
├── README.md
└── .gitignore
```

The project structure may evolve as additional MLOps components are introduced.

---

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/ansh-rohilla/Customer-Satisfaction-MLops.git
cd Customer-Satisfaction-MLops
```

### 2. Create a Virtual Environment

```bash
python -m venv .venv
```

Activate the environment on macOS/Linux:

```bash
source .venv/bin/activate
```

On Windows:

```bash
.venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## ZenML Configuration

Install ZenML:

```bash
pip install zenml
```

To use the ZenML server and dashboard:

```bash
pip install "zenml[server]"
```

Start the ZenML server:

```bash
zenml up
```

---

## MLflow Configuration

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

---

## Running the Training Pipeline

Run the standard training pipeline:

```bash
python run_pipeline.py
```

The training pipeline performs the following operations:

```text
Data Ingestion
      |
      v
Data Cleaning
      |
      v
Model Training
      |
      v
Model Evaluation
      |
      v
MLflow Tracking
```

---

## Running the Deployment Pipeline

To execute the continuous deployment workflow:

```bash
python run_deployment.py
```

The pipeline:

1. Ingests the data.
2. Cleans and processes the data.
3. Trains the model.
4. Evaluates the model.
5. Checks the deployment criteria.
6. Deploys the model if the criteria are satisfied.

---

## Running the Streamlit Application

Start the Streamlit application using:

```bash
streamlit run streamlit_app.py
```

The application provides an interface for submitting input features and obtaining customer satisfaction predictions from the deployed model.

---

## MLflow Experiment Tracking

MLflow can be used to inspect and compare different training runs.

Tracked information may include:

- Hyperparameters
- Evaluation metrics
- Model artifacts
- Training runs
- Model versions

This enables systematic experimentation and improves the reproducibility of model development.

---

## Data Management

The original Olist dataset is not committed to the repository because of its size.

The raw dataset should remain in the local `data/` directory and be excluded through `.gitignore`.

Example:

```gitignore
data/
*.csv
*.zip
.venv/
__pycache__/
mlruns/
.zen/
.env
.DS_Store
```

The repository contains the code and configuration required to process the dataset.

---

## Future Improvements

The project can be extended with additional MLOps capabilities:

- Docker-based deployment
- Automated testing
- GitHub Actions CI/CD
- Data validation
- Data drift detection
- Model drift monitoring
- Automated model retraining
- Cloud deployment
- Model versioning
- Kubernetes-based deployment
- API-based model serving
- Hyperparameter optimization
- Production database integration

---

## MLOps Concepts Demonstrated

This project demonstrates the following concepts:

- Machine learning pipelines
- Pipeline orchestration
- Experiment tracking
- Model evaluation
- Model versioning
- Model deployment
- Continuous deployment
- Reproducibility
- Model quality gates
- ML lifecycle management

---

## Dataset

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

## Learning Outcomes

The project demonstrates how a traditional machine learning workflow can be transformed into a reproducible MLOps system.

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

The MLOps workflow implemented in this project extends this process:

```text
Dataset
    |
    v
Data Pipeline
    |
    v
Feature Engineering
    |
    v
Model Training
    |
    v
Experiment Tracking
    |
    v
Model Evaluation
    |
    v
Deployment Validation
    |
    v
Model Deployment
    |
    v
Prediction Application
```

The project therefore provides practical experience with integrating machine learning development, experiment tracking, pipeline orchestration, and model deployment.

---

## Author

**Ansh Rohilla**

B.Tech Computer Science Engineering  
AI and Edge Computing

GitHub: https://github.com/ansh-rohilla