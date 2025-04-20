
# 💳 Credit Card Fraud Detection Using Machine Learning

This project demonstrates the implementation of machine learning models to detect fraudulent credit card transactions. It tackles a highly imbalanced dataset and uses classification techniques to identify suspicious activities effectively.

## 📂 Project Structure

```
credit-card-fraud-detection/
│
├── data/
│   └── creditcard.csv              # Dataset containing anonymized transaction records
│
├── models/
│   └── logistic_regression_model.pkl
│   └── random_forest_model.pkl
│
├── visuals/
│   └── accuracy_comparison.png     # Bar chart comparing model accuracies
│
├── src/
│   ├── preprocessing.py            # Functions for data loading and cleaning
│   ├── train_models.py             # Script for training Logistic Regression and Random Forest
│   ├── evaluate.py                 # Model evaluation metrics
│   └── predict.py                  # Fraud prediction function
│
├── main.py                         # Entry point to run the pipeline end-to-end
├── requirements.txt                # List of Python dependencies
└── README.md                       # Project documentation
```

## 🧠 Models Used

- **Logistic Regression**
- **Random Forest Classifier**

## 🔍 Dataset

The dataset used is sourced from [Kaggle's Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud). It contains transactions made by European cardholders in September 2013. Features are PCA-transformed for privacy.

- **Rows**: 284,807
- **Fraudulent**: 492
- **Legitimate**: 284,315

## ⚙️ Technologies Used

- Python 3.8+
- Pandas
- NumPy
- Scikit-learn
- Matplotlib / Seaborn

## 🧪 Steps Performed

1. **Data Loading and Cleaning**
   - Missing value analysis
   - Data type inspection

2. **Class Imbalance Handling**
   - Undersampling majority class to balance the dataset

3. **Feature Selection**
   - Used all features except the 'Class' column as inputs

4. **Model Training**
   - Trained Logistic Regression and Random Forest

5. **Evaluation**
   - Accuracy, Precision, Recall, F1-Score
   - Bar chart visualization

6. **Real-Time Prediction**
   - `predict_fraud()` function accepts transaction input and returns 'Fraud' or 'Legit'

## 📊 Performance Summary

| Model               | Accuracy | Precision | Recall | F1-Score |
|---------------------|----------|-----------|--------|----------|
| Logistic Regression | 93.4%    | 91.2%     | 87.5%  | 89.3%    |
| Random Forest       | 97.6%    | 96.8%     | 94.7%  | 95.7%    |

## 🔮 How to Use

1. Clone the repository:

```bash
git clone https://github.com/Usama-Goreja/credit-card-fraud-detection.git
cd credit-card-fraud-detection
```




sample_transaction = [...]  # Add your feature values
result = predict_fraud(sample_transaction)
print("Prediction:", result)
```

## 🚀 Future Improvements

- Use SMOTE or ADASYN for oversampling
- Hyperparameter tuning with Grid Search
- Try deep learning (e.g., LSTM for sequential transaction data)
- Deploy model as a REST API

## 📜 License

This project is licensed under the MIT License.

## 🤝 Contributions

Contributions are welcome! Feel free to open an issue or submit a pull request.

https://www.linkedin.com/in/usamaiqbal2000/


