import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

def process_data(df):
    df.drop(columns=['customerID'], inplace=True)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors="coerce")
    df["Churn"] = df["Churn"].map({"No" : 0, "Yes" : 1})

    y = df['Churn']
    x = df.drop("Churn", axis = 1)

    categorical_cols = x.select_dtypes(include="object").columns
    numerical_cols = x.select_dtypes(exclude="object").columns

    oe = ColumnTransformer(
        transformers = [
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
            ("num", "passthrough", numerical_cols)
        ]
    )
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
   
    x_train = oe.fit_transform(x_train[categorical_cols])
    x_test = oe.transform(x_test[categorical_cols])

    sc = StandardScaler()
    x_train = sc.fit_transform(x_train)
    x_test = sc.transform(x_test)

    return x_train, x_test, y_train, y_test