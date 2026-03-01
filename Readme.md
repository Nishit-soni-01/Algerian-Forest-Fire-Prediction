# Algerian Forest Fire Risk Prediction

## 📌 Project Overview
This project focuses on predicting the **Fire Weather Index (FWI)**, a critical indicator of forest fire risk, using meteorological data from two regions in Algeria: **Bejaia** and **Sidi Bel-abbes**.

The project follows an end-to-end data science workflow, including data cleaning, exploratory data analysis (EDA), feature engineering, and the deployment of a regularized machine learning model.

## 📊 Dataset Information
The dataset consists of 244 instances (122 per region) collected between June 2012 and September 2012.
- **Weather Attributes:** Temperature, Relative Humidity (RH), Wind Speed (Ws), and Rain.
- **FWI System Components:** FFMC (Fine Fuel Moisture Code), DMC (Duff Moisture Code), DC (Drought Code), ISI (Initial Spread Index), BUI (Buildup Index).
- **Target Variable:** Fire Weather Index (FWI).

## 🛠️ Tech Stack
- **Language:** Python 3.x
- **Libraries:** Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn
- **Modeling:** Ridge Regression (Regularized Linear Model)
- **Serialization:** Pickle (for model and scaler persistence)

## 📁 Project Structure
- `Algerian_forest_fires_dataset.csv`: The original raw dataset.
- `Algerian_forest_fires_cleaned_dataset.csv`: The processed data after cleaning and encoding.
- `Feature Engineering and EDA.ipynb`: Notebook containing data cleaning, visualization, and correlation analysis.
- `model Training.ipynb`: Notebook covering feature scaling, model training, and evaluation.
- `ridge.pkl`: The trained Ridge Regression model.
- `scaler.pkl`: The StandardScaler object for data normalization.

## 🚀 Key Features
1. **End-to-End Data Cleaning:** - Handled missing values and standardized column names.
    - Converted categorical labels into numerical formats for modeling.
    - Added a `Region` feature to differentiate datasets from different geographic areas.
2. **Exploratory Data Analysis (EDA):**
    - Identified that August and September have the highest frequency of forest fires.
    - Conducted correlation analysis to identify key drivers of fire risk.
3. **Machine Learning Pipeline:**
    - Implemented **Ridge Regression** to prevent overfitting and manage multicollinearity.
    - Used **StandardScaler** to ensure all meteorological features are on the same scale.

## 📈 Results & Insights
- High temperatures (above 30°C) and low humidity (below 50%) are strong predictors of fire incidents.
- Regularization through Ridge Regression helped achieve a more stable and generalizable model compared to standard Linear Regression.

## 💻 How to Run
1. **Clone the repository:**
   ```bash
   git clone [https://github.com/yourusername/Algerian-Forest-Fire-Prediction.git](https://github.com/yourusername/Algerian-Forest-Fire-Prediction.git)