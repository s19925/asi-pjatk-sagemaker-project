from typing import Dict, Tuple
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import wandb
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import r2_score, accuracy_score, roc_auc_score
import pandas as pd
from typing import Any, Dict
import logging
from sklearn.model_selection import train_test_split

sweep_configuration = {
    'method': 'bayes',
    'name': 'sweep',
    'metric': {'goal': 'maximize', 'name': 'roc_auc'},
    'parameters':
        {
            'n_estimators': {'min': 25, 'max': 200},
            'max_depth': {'min': 3, 'max': 10}
        }
}

sweep_id = wandb.sweep(sweep=sweep_configuration, project='PJATK-ASI-PROJECT')


def prepare_data_for_modeling(data: pd.DataFrame) -> pd.DataFrame:
    '''Prepare data for modeling

    Inputs:
    data: a pandas dataframe with customers data

    Outputs:
    data: a pandas dataframe with customers data ready for modeling
        X: a pandas dataframe with customer features
        y: a pandas series with customer labels, encoded as integers (0 = low revenue, 1 = high revenue)

    '''
    # Suppress "a copy of slice from a DataFrame is being made" warning
    pd.options.mode.chained_assignment = None

    # prepare dataset for classification
    features = data.columns[1:-1]
    X = data[features]
    y = data['HeartDisease']

    # identify numeric features in X
    numeric_features = X.select_dtypes(include=[np.number]).columns
    # identify categorical features in X
    categorical_features = X.select_dtypes(exclude=[np.number]).columns

    # Normalize numeric features with MinMaxScaler
    scaler = MinMaxScaler()
    X[numeric_features] = scaler.fit_transform(X[numeric_features])

    # One-hot encode categorical features
    X = pd.get_dummies(X, columns=categorical_features)

    # transform bool in y to 0/1
    y = y.astype(int)

    # create a dataframe with the features and the labels
    data = pd.concat([X, y], axis=1)

    return data


def split_data(data: pd.DataFrame, parameters: Dict) -> Tuple:
    """Splits data into features and targets training and test sets.

    Args:
        data: Data containing features and target.
        parameters: Parameters defined in parameters/data_science.yml.
    Returns:
        Split data.
    """

    features = data.columns[:-1]
    X = data.iloc[:, 0:-1]
    y = data['HeartDisease']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test


def hyperparameters_tuning():
    df = pd.read_csv('data/01_raw/heart.csv')
    df['Sex'] = df['Sex'].astype('category')
    df['ChestPainType'] = df['ChestPainType'].astype('category')
    df['FastingBS'] = df['FastingBS'].astype('category')
    df['RestingECG'] = df['RestingECG'].astype('category')
    df['ExerciseAngina'] = df['ExerciseAngina'].astype('category')
    df['ST_Slope'] = df['ST_Slope'].astype('category')
    df = df.dropna()

    data_prepared = prepare_data_for_modeling(df)

    features = data_prepared.columns[:-1]
    X = data_prepared.iloc[:, 0:-1]
    y = data_prepared['HeartDisease']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, train_size=0.8, random_state=42)

    config_defaults = {
        "n_estimators": 50,
        "max_depth": 3,
    }

    wandb.init(project='PJATK-ASI-PROJECT', config=config_defaults)

    n_estimators = wandb.config.n_estimators
    max_depth = wandb.config.max_depth

    model = train_model(X_train, y_train, n_estimators, max_depth)

    accuracy, roc_auc = evaluate_model(model, X_test, y_test)

    wandb.log({"n_estimators": n_estimators})
    wandb.log({"max_depth": max_depth})
    wandb.log({"accuracy": accuracy})
    wandb.log({"roc_auc": roc_auc})

    return n_estimators, max_depth


def train_model(X_train: pd.DataFrame, y_train: pd.Series, n_estimators, max_depth) -> RandomForestClassifier:
    """Trains model.

    Args:
        max_depth:
        n_estimators:
        X_train: Training data of independent features.
        y_train: Training data for HeartDisease.

    Returns:
        Trained model.
    """
    pd.options.mode.chained_assignment = None
    model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=0)

    model.fit(X_train, y_train)

    return model


def evaluate_model(
        model: RandomForestClassifier, X_test: pd.DataFrame, y_test: pd.Series
):
    """Calculates and logs the coefficient of determination.

    Args:
        model: Trained model.
        X_test: Testing data of independent features.
        y_test: Testing data for price.
    Returns:
        accuracy, roc_auc
    """

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_pred, y_test)
    roc_auc = roc_auc_score(y_pred, y_test)
    logger = logging.getLogger(__name__)
    logger.info("Model has an accuracy of %.3f on test data.", accuracy)
    logger.info("Model has an ROC AUC of %.3f on test data.", roc_auc)

    return accuracy, roc_auc


wandb.agent(sweep_id, function=hyperparameters_tuning, count=4)
