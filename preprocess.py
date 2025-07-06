import pandas as pd
import os
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def load_data_kagglehub(path):
    print("Available files:", os.listdir(path))
    df = pd.read_csv(os.path.join(path, 'creditcard.csv'))
    return df

def preprocess_data(df):
    # Scale 'Amount'
    scaler = StandardScaler()
    df['Amount'] = scaler.fit_transform(df[['Amount']])
    df = df.drop(['Time'], axis=1)

    X = df.drop('Class', axis=1)
    y = df['Class']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    return X_train, X_test, y_train, y_test
