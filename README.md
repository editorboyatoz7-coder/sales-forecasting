# 📈 Sales Forecasting System

An end-to-end **Sales Forecasting** project built for an internship assignment. It covers data generation, exploratory data analysis (EDA), feature engineering, machine learning model training, evaluation, future forecasting, and an interactive Streamlit dashboard.

---

## 🚀 Features

- Synthetic daily sales dataset generator (trend + seasonality + promotions + holidays)
- Feature engineering: date features, lag features, rolling statistics
- Multiple forecasting models: **Linear Regression**, **Random Forest**, **XGBoost**
- Model evaluation with **MAE**, **RMSE**, **MAPE**, **R²**
- Recursive future forecasting (30/60/90 days)
- Interactive **Streamlit dashboard** for visualization
- Jupyter notebook for EDA & experimentation

---

## 🗂️ Project Structure

```
sales-forecasting/
├── app.py                     # Streamlit dashboard
├── requirements.txt
├── README.md
├── data/
│   └── sales_data.csv         # Generated dataset
├── notebooks/
│   └── sales_forecasting_eda.ipynb
├── src/
│   ├── __init__.py
│   ├── generate_data.py       # Creates synthetic dataset
│   ├── preprocessing.py       # Feature engineering
│   ├── evaluation.py          # Evaluation metrics
│   ├── train_models.py        # Train & compare models
│   └── forecast_future.py     # Generate future predictions
└── outputs/                    # Model artifacts & results (generated)
```

---

## 🛠️ Setup & Installation

### 1. Clone the repository
```bash
git clone https://github.com/<your-username>/sales-forecasting.git
cd sales-forecasting
```

### 2. Create a virtual environment (recommended)
```bash
python -m venv venv
source venv/bin/activate      # On Windows: venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

---

## ▶️ How to Run

### Step 1 — Generate the dataset
```bash
python src/generate_data.py
```
This creates `data/sales_data.csv` with 3 years of synthetic daily sales data.

### Step 2 — Train and compare models
```bash
python -m src.train_models
```
This trains Linear Regression, Random Forest, and XGBoost, saves a comparison table to `outputs/model_comparison.csv`, and saves the best model to `outputs/best_model.pkl`.

### Step 3 — Forecast future sales
```bash
python -m src.forecast_future
```
This generates a 30-day forecast and saves it to `outputs/future_forecast.csv`.

### Step 4 — Launch the dashboard
```bash
streamlit run app.py
```
Open the link shown in your terminal (usually `http://localhost:8501`).

### (Optional) Explore the notebook
```bash
jupyter notebook notebooks/sales_forecasting_eda.ipynb
```

---

## 📊 Methodology

1. **Data Preprocessing** — handle dates, missing values, encode features
2. **EDA** — trend, weekly/monthly seasonality, promotion impact
3. **Feature Engineering** — lag features (1, 7, 14, 30 days), rolling mean/std
4. **Modeling** — Linear Regression, Random Forest, XGBoost
5. **Evaluation** — MAE, RMSE, MAPE, R²
6. **Forecasting** — recursive multi-step forecasting for future dates
7. **Visualization** — Streamlit dashboard with interactive charts

---

## 📈 Evaluation Metrics

| Metric | Description |
|--------|-------------|
| MAE    | Mean Absolute Error |
| RMSE   | Root Mean Squared Error |
| MAPE   | Mean Absolute Percentage Error |
| R²     | Coefficient of Determination |

---

## 🔮 Future Enhancements

- Incorporate real-world datasets (Walmart, Rossmann, or company data)
- Add LSTM/GRU deep learning models
- Add Facebook Prophet for trend/seasonality decomposition
- Multi-store / multi-product forecasting
- Deploy dashboard to Streamlit Cloud / Render / Heroku

---

## 📄 License

This project is open-source and free to use for educational purposes.

---

## 🙋 Author

Internship Project — Sales Forecasting System
