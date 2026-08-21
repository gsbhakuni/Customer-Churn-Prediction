import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

def process_data(df):
    df.drop(columns=['customerID'], inplace=True)

    le = LabelEncoder()
    col = [
        "gender",
        "Partner",
        "Dependents",
        "PhoneService",
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
        "PaperlessBilling",
        "Churn",
    ]
    for i in col:
        df[i] = le.fit_transform(df[i])

    col = ['Contract', 'PaymentMethod', 'InternetService', 'MultipleLines']
    df = pd.get_dummies(df, columns=col, drop_first=True)

    for i in df.columns:
        df[i] = le.fit_transform(df[i])

    y = df['Churn']
    x = df.drop(columns=['Churn'])

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

    sc = StandardScaler()
    x_train = sc.fit_transform(x_train)
    x_test = sc.transform(x_test)

    return x_train, x_test, y_train, y_test