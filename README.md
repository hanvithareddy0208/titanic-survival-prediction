🚢 Titanic Survival Prediction
A machine learning project that predicts whether a passenger survived the Titanic disaster based on features like age, sex, class, and more. This classic classification problem is based on the Kaggle Titanic dataset.

📂 Project Structure
titanic-survival-prediction
│
├── data                     # Raw and processed data
│   ├── train.csv
│   └── test.csv
│
├── code.py                  # code
│
├── output images            # output images
│
└── README.md                 # Project documentation

📊 Dataset
The dataset is available on Kaggle - Titanic: Machine Learning from Disaster. It includes the following features:
PassengerId
Pclass (Ticket class)
Name
Sex
Age
SibSp (# of siblings / spouses aboard)
Parch (# of parents / children aboard)
Ticket
Fare
Cabin
Embarked (Port of Embarkation)
The target variable is:
Survived (0 = No, 1 = Yes)

TECHNOLOGIES USED :
pycharm 

Installed libraries:
pandas
matplotlib
seaborn
scikit-learn

🧠 Model
The model is typically a classification algorithm such as:
Logistic Regression
Random Forest
XGBoost
Performance is evaluated using accuracy, precision, recall, and F1-score.

✅ Features Used
After preprocessing, the following features were used in the model:
Pclass
Sex
Age (imputed)
SibSp
Parch
Fare
Embarked

📈 Evaluation

Logistic Regression Model:
Accuracy: 0.8045
Precision: 0.7910
Recall: 0.7162
F-score: 0.7518
Cross-validation scores: [0.68965517 0.64655172 0.73275862 0.75862069 0.73275862]
Average cross-validation score: 0.7120689655172414


K-Nearest Neighbors (KNN) Model:
Accuracy: 0.6480
Precision: 0.6341
Recall: 0.3514
F-score: 0.4522
Note: Results may vary based on feature engineering and model used.

📚 References
Kaggle Titanic Competition
Scikit-learn Documentation
Pandas Documentation
