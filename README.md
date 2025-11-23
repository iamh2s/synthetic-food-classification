Synthetic Food Classification Project

This project focuses on **food classification and nutrition analysis** using synthetic datasets.  
It includes data cleaning, imputation, feature engineering, and benchmarking multiple ML models.

---

 Project Structure
- `synthetic_food_dataset_clean.csv` → Cleaned dataset
- `synthetic_food_dataset_imbalanced.csv` → Imbalanced dataset for testing
- `synthetic_food_dataset_report.html` → Data profiling and analysis report
- `notebooks/` → Jupyter notebooks for experiments
- `src/` → Source code for preprocessing, modeling, and evaluation

---

Features
- Automated data cleaning and imputation (numeric, categorical, boolean, text)
- Outlier detection and handling
- Feature encoding and scaling
- Model benchmarking (Logistic Regression, Random Forest, XGBoost, etc.)
- Evaluation metrics: Accuracy, Precision, Recall, F1-score, Confusion Matrix

---
Installation
Clone the repository and set up a virtual environment:
```bash
git clone https://github.com/yourname/synthetic-food-classification.git
cd synthetic-food-classification
python -m venv venv
source venv/bin/activate   # or venv\Scripts\activate on Windows
pip install -r requirements.txt
