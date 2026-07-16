# Sleep Health Prediction Model

## Project Overview

This project investigates the relationship between sleep health and daily outcomes using machine learning techniques. A common preprocessing pipeline is applied to a sleep health dataset, followed by three independent prediction tasks that represent different machine learning problem types: multiclass classification, regression, and binary classification.

### Dataset

The dataset is a synthetic sleep health dataset published on Kaggle, containing 100,000 records and 32 columns. It is calibrated against peer-reviewed sources including the CDC National Health Interview Survey, Sleep Foundation population studies, and the Framingham Heart Study, so while individual values are synthetic, the statistical relationships mirror real-world patterns.

After removing the row identifier, 28 features remain across three groups: 18 numeric continuous features (e.g. age, BMI, sleep duration, sleep quality score, REM percentage, stress score), 3 binary flags (exercise that day, sleep aid used, shift work), and 7 categorical features (gender, occupation, country, chronotype, mental health condition, season, day type). There are no missing values and no duplicate rows.

### Target A: Sleep Disorder Risk Prediction

**Target Variable:** `sleep_disorder_risk`

This task focuses on predicting an individual's sleep disorder risk level across four categories:

* Healthy
* Mild
* Moderate
* Severe

**Models Evaluated:**

* Logistic Regression
* Random Forest
* AdaBoost

### Target B: Cognitive Performance Prediction

**Target Variable:** `cognitive_performance_score`

This regression task predicts next-day cognitive performance as a continuous numerical score.

**Models Evaluated:**

* Linear Regression
* Random Forest Regressor
* Gradient Boosting Regressor

### Target C: Restfulness Prediction

**Target Variable:** `felt_rested`

This binary classification task predicts whether an individual feels rested upon waking.

**Models Evaluated:**

* Logistic Regression
* Random Forest Classifier
* XGBoost Classifier

### Study Structure

The three prediction tasks represent different levels of complexity and require different evaluation strategies. While all tasks share the same data preparation and feature engineering workflow, each pipeline is trained and evaluated independently using metrics appropriate to its problem type.

* **Sections 1–6:** Shared data preprocessing and exploratory analysis
* **Section 7:** Pipeline A – Sleep Disorder Risk Classification
* **Section 8:** Pipeline B – Cognitive Performance Regression
* **Section 9:** Pipeline C – Restfulness Classification
* **Sections 10–12:** Comparative analysis, discussion, limitations, and future work

---

## Getting Started

### Requirements

* Python 3.x
* pandas, numpy
* scikit-learn (Logistic Regression, Random Forest, AdaBoost, Gradient Boosting, PCA, StandardScaler)
* xgboost
* matplotlib, seaborn

### Running the Project

1. Clone the repository:
   ```bash
   git clone git@github.com:themarkolusanya1/Sleep_health_prediction_model.git
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Open and run `data_science_sleep_health_final(new) (1).py` [or specify the notebook/script entry point].

---

## Team Members

| No. | Team Member            |
| --- | ----------------------- |
| 1   | John Edikan              |
| 2   | Mawumenyo Atsu Nyamadi   |
| 3   | Makinde Mark Olusanya    |
