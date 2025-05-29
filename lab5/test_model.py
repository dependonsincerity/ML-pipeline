import pytest
import pandas as pd
import pickle
from sklearn.metrics import mean_squared_error

@pytest.fixture(scope="module")
def model():
    with open("model.pkl", "rb") as f:
        return pickle.load(f)

def test_clean1(model):
    df = pd.read_csv("clean1.csv")
    preds = model.predict(df[['x']])
    mse = mean_squared_error(df['y'], preds)
    assert mse < 1.0

def test_clean2(model):
    df = pd.read_csv("clean2.csv")
    preds = model.predict(df[['x']])
    mse = mean_squared_error(df['y'], preds)
    assert mse < 1.0

def test_clean3(model):
    df = pd.read_csv("clean3.csv")
    preds = model.predict(df[['x']])
    mse = mean_squared_error(df['y'], preds)
    assert mse < 1.0

def test_noisy(model):
    df = pd.read_csv("noisy.csv")
    preds = model.predict(df[['x']])
    mse = mean_squared_error(df['y'], preds)
    assert mse < 1.0
