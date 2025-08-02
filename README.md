# 🛳 Titanic Survival Prediction – Random Forest Classifier

## 📌 Project Overview
This project predicts whether passengers survived the Titanic disaster using a **Random Forest Classifier**. It demonstrates an **end-to-end ML workflow**, including:
- Data cleaning
- Feature engineering
- Model training and evaluation
- Visualization of results and feature importance

---

## 🚀 Technologies Used
- **Language:** Python 3.10+
- **Libraries:**  
  - `pandas`, `numpy` – Data handling  
  - `matplotlib`, `seaborn` – Visualization  
  - `scikit-learn` – Machine Learning models & evaluation

---

## 📂 Project Structure
```
ml-titanic-project/
│
├── data/
│   ├── train.csv              # Dataset from Kaggle
│   ├── test.csv 
├── notebooks/
│   ├── titanic_random_forest.ipynb   # Main analysis and model notebook
│
├── requirements.txt           # Python dependencies
└── README.md                  # Project documentation
```

---

## 📊 Dataset
- **Source:** [Kaggle Titanic Dataset](https://www.kaggle.com/competitions/titanic/data)
- **Target variable:** `Survived` (0 = Did not survive, 1 = Survived)
- **Features used:** Pclass, Sex, Age, SibSp, Parch, Fare, Embarked

---

## 🔧 Steps Performed
1. **Data Cleaning**
   - Dropped non-predictive columns (`PassengerId`, `Cabin`, `Name`, `Ticket`)
   - Filled missing `Age` values with median
   - Filled missing `Embarked` values with most frequent value (`S`)
   - Converted categorical variables to numeric (One-Hot Encoding)

2. **Model Training**
   - Random Forest Classifier with:
     - `n_estimators = 100`
     - `class_weight = balanced`
     - `random_state = 42`
   - Train-test split (80/20)

3. **Evaluation**
   - Achieved **~83% accuracy** on the test set
   - Visualized confusion matrix
   - Ranked top 10 feature importances

---

## 📈 Results
- **Accuracy:** ~83%  
- **Top predictive features:** `Fare`, `Sex_male`, `Age`, `Pclass`  
- **Model correctly predicted most non-survivors, slightly under-predicted survivors due to class imbalance.**

---

### Confusion Matrix Example:
|            | Predicted No | Predicted Yes |
|------------|-------------|---------------|
| **Actual No**  | 92          | 13            |
| **Actual Yes** | 17          | 57            |

---

## ▶️ How to Run
1. Clone this repository:
   ```bash
   git clone https://github.com/ykim3001/ml-titanic-project.git
   cd ml-titanic-project
   ```
2. Install dependencies:
   ```bash
   pip3 install -r requirements.txt
   ```
3. Open the notebook:
   ```bash
   jupyter notebook notebooks/titanic_random_forest.ipynb
   ```

---

## 📌 Next Steps
- Try different ML models (Logistic Regression, XGBoost, Neural Networks)
- Perform hyperparameter tuning with GridSearchCV
- Deploy this model with FastAPI and Docker

---

## 📜 Acknowledgments
- [Kaggle Titanic Competition](https://www.kaggle.com/competitions/titanic)
- Scikit-learn Documentation
