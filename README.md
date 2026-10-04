# Student Performance Prediction 🎓

## 📌 About the Project

This project uses Machine Learning to predict student academic performance based on selected student-related features. It aims to analyze student data and provide predictions using classification models.

## 🎯 Objectives

* Analyze student performance data.
* Train Machine Learning models.
* Predict student performance.
* Evaluate and compare model performance.

## 🛠️ Technologies & Libraries

* Python
* Pandas
* NumPy
* Scikit-learn
* Matplotlib
* Seaborn
* FastAPI
* fastapi
* uvicorn
* httpx
* joblib


## 🤖 Machine Learning Models

* Logistic Regression
* DecisionTreeClassifier

## 📊 Evaluation Metrics

* Accuracy
* Precision
* Recall
* F1-Score

## 🚀 How to Run

1. Clone the repository:

   ```bash
   git clone YOUR_REPOSITORY_LINK
   ```

2. Install the required libraries:

   ```bash
   pip install -r requirements.txt
   ```

3. Run the application:

   ```bash
   uvicorn main:app --reload
   ```
## 📊 Model Evaluation

The performance of the Machine Learning models was evaluated using the following metrics:

* **Accuracy:** Measures the overall percentage of correct predictions.
* **Precision:** Measures the proportion of positive predictions that are actually correct.
* **Recall:** Measures the proportion of actual positive cases correctly identified.
* **F1-Score:** Balances Precision and Recall.

### 📈 Model Comparison

| Model               | Accuracy | Precision | Recall | F1-Score |
| ------------------- | -------: | --------: | -----: | -------: |
| Logistic Regression |   90.00% |    94.50% | 93.64% |   94.06% |
| Random Forest       |   88.46% |    93.58% | 92.73% |   93.15% |

### 🏆 Results

Logistic Regression achieved the best overall performance among the evaluated models, with an accuracy of 90.00% and an F1-Score of 94.06%.

It outperformed Random Forest across all the reported evaluation metrics.


## 📁 Project Structure

```text
Student-Performance-Prediction
│
├── gitignore
├── app.py
├── requirements.txt
├── README.md
└── student_performance_model.pkl
```
## 📚 Future Improvements
- Improve model performance.
- Add more visualizations.

## 👨‍💻 Author

**Omar Mohamed Elshnofy**
Third-Year Computer Engineering Student | Machine Learning Engineer in Progress 🚀
