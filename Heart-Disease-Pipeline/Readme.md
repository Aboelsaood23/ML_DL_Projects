# 🩺 End-to-End Heart Disease Classification Pipeline

An elegant, end-to-end machine learning and data engineering ecosystem designed to predict the presence or absence of heart disease. This project bridges rigorous data preprocessing, multi-model selection, automated data validation, and multi-container deployment into a seamless production-ready architecture.

---

# ⚠️ Crucial Project Discovery: Target Label Mapping

During exploration, a key correction was made to the target label interpretations to match clinical reality:

* **Label 0 ➔ Presence of the heart condition**
* **Label 1 ➔ Absence of the heart condition**

---

# 🛠️ Project Architecture & Core Steps

## 1️⃣ One-Hot Encoding & Feature Engineering

To safeguard against data leakage and guarantee deterministic data behavior during live inference, a dedicated preprocessing engine was built:

### ✅ One-Hot Encoding

Applied clean encoding to categorical variables, converting non-numeric data fields into model-ready features.

### ✅ Feature Scaling

Scaled continuous metrics to equalize feature magnitude across different mathematical architectures.

### ✅ Inference Synchronization

All transformations are encapsulated inside a master preprocessing file. This guarantees that live inference data flowing from the UI is scaled and encoded in the exact same way the models were originally trained on.

---

## 2️⃣ Seven-Model Tournament, Cross-Validation & Grid Search

Instead of banking on a single classifier, a robust model-evaluation script was developed to find the absolute best predictive performance:

### ✅ 7 Classification Models

Simultaneously trained and tested seven diverse machine learning algorithms.

### ✅ Cross-Validation

Applied cross-validation during the training loop to ensure model stability and prevent overfitting across data subsets.

### ✅ Hyperparameter Tuning

Utilized systematic Grid Search to sweep through parameter grids, squeezing out maximum performance for target evaluation metrics.

---

## 3️⃣ Asynchronous Backend API & Pydantic Validation

The backend serves as a high-performance, decoupled inference gateway:

### ✅ FastAPI Backend

Engineered a fast, production-grade API server to host the multi-model pipeline.

### ✅ Data Validation with Pydantic

Built strict runtime data validation schemas using Pydantic. Any incorrect, missing, or malformed data payload sent to the endpoint is trapped instantly at the gateway before hitting the models.

---

## 4️⃣ Interactive Frontend Dashboard

A smooth web interface was built to allow direct interaction with the underlying model suite:

### ✅ Streamlit Frontend

Developed a user-facing dashboard where users can input patient clinical metrics and receive instant multi-model predictions.

> 🤖 **Front-End Development Note:**
> As a Data Scientist completely focused on core ML engineering, data pipelines, and mathematical foundations, I don't have any prior experience with frontend technologies. This entire Streamlit interface was successfully built, styled, and connected to the API completely using AI tools.

---

## 5️⃣ Container Isolation & Multi-Container DevOps

To eliminate any system dependency conflicts, the entire architecture was fully modularized and containerized:

### ✅ Dockerfiles

Constructed isolated, optimized Docker images for both the FastAPI backend and the Streamlit UI.

### ✅ Docker Compose Orchestration

Configured a `docker-compose.yml` file to stitch the frontend, backend, and internal networks into a single, cohesive ecosystem.

> 🤖 **DevOps Note:**
> Setting up network links, port forwarding, and multi-container coordination in Docker Compose was also completed with the help and guidance of AI tools, drastically speeding up the infrastructure setup.

---

# 🎨 Tech Stack Matrix

| Core Domain              | Technologies & Libraries                 |
| ------------------------ | ---------------------------------------- |
| 🐍 Core Engine           | Python                                   |
| 📊 ML & Engineering      | Pandas, NumPy, Scikit-Learn              |
| ⚡ Backend & Validation   | FastAPI, Pydantic                        |
| 🎨 User Interface        | Streamlit *(Powered by AI)*              |
| 🐳 DevOps Infrastructure | Docker, Docker Compose *(Powered by AI)* |
