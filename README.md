# Earthquake in Nepal: Building Damage Prediction


Data

Download the dataset from:
<[dataset link](https://www.kaggle.com/datasets/domingosnhamussusa/earthquake-damage-in-nepal)>

Place it in:

data/raw/earthquake_data.csv

## Project Overview

This project focuses on a **binary classification problem** to predict the likelihood of a building suffering **severe damage due to an earthquake in Nepal**.

Using data from the **2015 Nepal (Gorkha) earthquake**, the goal is to analyze structural and environmental features of buildings and build a machine learning model that can estimate damage risk.

---

## Objective

To develop a machine learning model that predicts whether a building is likely to experience **severe structural damage** during an earthquake.

This can support:
- Disaster risk assessment  
- Emergency planning  
- Safer infrastructure design  

---

## Project Structure
├── README.md <- Top-level project documentation
│
├── data
│ ├── raw <- Original, immutable dataset
│ └── processed <- Cleaned and feature-engineered data
│
├── models <- Trained models and serialized outputs
│
├── notebooks <- Jupyter notebooks for EDA and experimentation
│
├── reference <- Supporting documentation and domain knowledge
│
├── reports <- Generated analysis reports (HTML, PDF, etc.)
│ └── figures <- Visualizations and plots used in reports
│
├── requirements.txt <- Python dependencies for reproducibility
│
└── src <- Source code (data processing, training, evaluation, utility modules)
---

## Workflow

1. Data collection and exploration (EDA)
2. Data cleaning and preprocessing
3. Feature engineering and selection
4. Model training (binary classification)
5. Model evaluation and validation
6. Communicating results

---

## Dataset

The Nepal Earthquake Damage Dataset contains information about buildings affected by the 2015 Nepal earthquake.

Source:
https://eq2015.npc.gov.np/#/download

Features include building age, height, foundation type, roof type, land surface condition, and other structural characteristics.

Target:
- damage_grade

## Expected Outcome

A trained classification model capable of predicting whether a building will suffer **severe damage**, helping improve earthquake preparedness and risk mitigation strategies.

---
