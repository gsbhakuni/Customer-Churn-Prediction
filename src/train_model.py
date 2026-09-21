from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score

def Logistic_Regression(x_train, x_test, y_train, y_test):
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)
    accuracy = accuracy_score(y_test, predictions)
    accuracy = accuracy * 100
    return model, round(accuracy, 2)


def Random_Forest(x_train, x_test, y_train, y_test):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)
    accuracy= accuracy_score(y_test, predictions)
    accuracy = accuracy * 100
    return model, round(accuracy, 2)

def XGBoost(x_train, x_test, y_train, y_test):
    model = XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth = 3, random_state=42)
    model.fit(x_train, y_train)
    predict = model.predict(x_test)
    accuracy = accuracy_score(y_test, predict)
    accuracy = accuracy * 100
    return model, round(accuracy, 2)
