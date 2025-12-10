Hybrid Medical–Financial Risk Scoring System
AI Pipeline for Automated Diabetic Retinopathy Screening + Insurance Underwriting
📌 Overview

This project presents a hybrid AI system that automates medical-risk evaluation from retinal fundus images and integrates it with structured financial data to generate a unified underwriting score. With diabetic retinopathy highly prevalent in India and insurance approvals for chronic conditions often slow and restrictive, the system provides a scalable and objective approach to medical–financial risk assessment.

All components—data processing, model training, evaluation, fusion, and visualization—are contained in a single, end-to-end notebook.

📓 What the Notebook Does

Loads and preprocesses retinal images

Trains an EfficientNet-B0 model to classify diabetic retinopathy severity

Generates Grad-CAM visual explanations

Processes demographic and financial data

Trains an XGBoost/Gradient Boosting financial risk model

Combines medical + financial predictions into a single underwriting score

Plots risk distributions, feature importance, and portfolio insights

🧠 Pipeline Architecture

Medical Model: EfficientNet-B0 deep learning classifier

Financial Model: Gradient Boosting / XGBoost

Fusion Layer: Weighted integration of medical severity probability + financial attributes

Outputs: Unified underwriting score + interpretability visuals

📂 Repository Structure
│── Hybrid_Underwriting_System.ipynb   # Full project notebook
│── README.md

🛠️ Tech Stack

TensorFlow/Keras (EfficientNet-B0)

Scikit-Learn, XGBoost

OpenCV, Pandas, NumPy

Matplotlib (visualizations)

Grad-CAM (model explainability)

📊 Results

Automated prediction of diabetic retinopathy severity

Financial risk segmentation based on demographic + behavioral attributes

Unified underwriting score suitable for eligibility filtering and premium calibration

Clear visual outputs for reporting and analysis
