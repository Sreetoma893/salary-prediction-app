# 💼 Employee Salary Prediction

A machine learning web application that predicts employee salary based on
experience, education level, job role, location, and industry.

**B.Tech Final Year Project (2022–2026)**
Sreetoma Ghosh · B.Tech in Information Technology
Department of Engineering and Technological Studies (DETS), University of Kalyani

This project demonstrates a complete ML workflow: data generation/
preprocessing, exploratory data analysis, multi-model comparison, and
deployment via an interactive Streamlit web app.

---

## 📌 Features

- **Exploratory Data Analysis (EDA)** — salary distributions, correlations,
  and trends across experience, education, location, and industry.
- **Model Comparison** — Linear Regression, Random Forest, and XGBoost
  are trained and evaluated; the best-performing model is automatically
  selected based on R² score.
- **Interactive Web App** — Streamlit-based UI supporting:
  - Single prediction (fill a form, get an instant salary estimate)
  - Batch prediction (upload a CSV, download predictions)
- **Feature Importance Analysis** — understand which factors most affect
  predicted salary.

---

## 🗂️ Project Structure

```
salary_prediction_project/
├── data/
│   └── salary_data.csv              # Training dataset
├── notebooks/
│   └── EDA_and_Model_Comparison.ipynb   # Full analysis & model comparison
├── models/
│   ├── salary_model.pkl             # Saved best model
│   ├── model_columns.pkl            # Feature column order (for inference)
│   └── model_name.pkl               # Name of the selected best model
├── generate_dataset.py              # Synthetic dataset generator
├── train_model.py                   # Training script (CLI)
├── app.py                           # Streamlit web app
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 📊 Dataset

The dataset (`data/salary_data.csv`) contains **1,500 synthetic records**
with the following columns:

| Column            | Description                                             |
|-------------------|----------------------------------------------------------|
| `experience`      | Years of professional experience                          |
| `education_level` | High School / Bachelors / Masters / PhD                   |
| `job_role`        | Software Engineer, Data Scientist, Analyst, Manager, etc.  |
| `location`        | City (Bangalore, Delhi, Mumbai, etc.)                      |
| `industry`        | IT, Finance, Healthcare, Education, Retail, Manufacturing  |
| `salary`          | Target variable — annual salary (INR)                     |

> **Note:** Salaries follow realistic patterns (experience growth curve,
> education/location/industry multipliers, plus random noise) rather than
> being purely random, so the models learn meaningful relationships.
> You can replace `data/salary_data.csv` with a real-world dataset that
> has the same column names to retrain on actual data.

---

## 🧠 Model Comparison Results

| Model              | MAE      | RMSE     | R² Score |
|--------------------|----------|----------|----------|
| Linear Regression  | ~96,000  | ~129,000 | ~0.91    |
| Random Forest      | ~136,000 | ~180,000 | ~0.83    |
| **XGBoost**        | **~86,000** | **~116,000** | **~0.93** |

*(Exact numbers may vary slightly on re-run due to random sampling.
See `notebooks/EDA_and_Model_Comparison.ipynb` for the full results.)*

**XGBoost** was selected as the production model.

---

## 🚀 Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/<your-username>/salary-prediction-app.git
cd salary-prediction-app
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Generate the dataset (optional — already included)
```bash
python generate_dataset.py
```

### 4. Train the models
```bash
python train_model.py
```
This compares Linear Regression, Random Forest, and XGBoost, then saves
the best model to `models/`.

### 5. Run the web app
```bash
streamlit run app.py
```
Open the URL shown in the terminal (typically `http://localhost:8501`).

---

## ☁️ Running on Google Colab

If you don't want to set up a local environment, you can run this entirely
on Google Colab using `ngrok` for a public link. See `colab_notebook.md`
for step-by-step instructions, or:

```python
!pip install -r requirements.txt pyngrok -q
!python generate_dataset.py
!python train_model.py

from pyngrok import ngrok
import os, time

ngrok.kill()
os.system("streamlit run app.py &")
time.sleep(6)
print(ngrok.connect(8501))
```

⚠️ You'll need your own free ngrok authtoken from
[dashboard.ngrok.com](https://dashboard.ngrok.com/tunnels/authtokens) —
never commit your authtoken to GitHub.

---

## 🛠️ Tech Stack

- **Language:** Python 3.10+
- **ML Libraries:** scikit-learn, XGBoost
- **Data Processing:** pandas, NumPy
- **Visualization:** matplotlib, seaborn
- **Web App:** Streamlit
- **Model Persistence:** joblib

---

## 📈 Future Improvements

- Train on a real-world, larger dataset (e.g. from Kaggle or industry data)
- Add more features: skills/certifications, company size, remote vs onsite
- Hyperparameter tuning (GridSearchCV / Optuna)
- Try additional models (LightGBM, CatBoost, neural networks)
- Add authentication and a database to log predictions
- Deploy permanently on Streamlit Community Cloud / Render / Railway

---

## 📄 License

This project is open-source and available for academic and educational use.

---

## 🙋 Author

**Sreetoma Ghosh**
B.Tech in Information Technology (2022 – 2026)
Department of Engineering and Technological Studies (DETS), University of Kalyani
