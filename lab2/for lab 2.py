import os
import wget
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, 'iris.csv')  
MODEL_PATH = os.path.join(BASE_DIR, 'model.pkl')  # Путь для сохранения модели

# Ссылка на данные, если нужно скачать
DATA_URL = 'https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv'

os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

# Скачивание данных
def download_data():
    if not os.path.exists(DATA_PATH):
        print(f"Downloading dataset from {DATA_URL}...")
        wget.download(DATA_URL, DATA_PATH)
        print("\nDataset downloaded successfully!")
    else:
        print("Dataset already exists.")

# Подготовка данных
def prepare_data():
    print("Preparing data...")
    df = pd.read_csv(DATA_PATH)
    X = df.drop(columns=['species'])  # Входные признаки
    y = df['species']               # Целевая переменная
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print("Data prepared successfully!")
    return X_train, X_test, y_train, y_test

# Обучение модели
def train_model(X_train, y_train):
    print("Training model...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    print("Model trained successfully!")
    return model

# Сохранение модели
def save_model(model):
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(model, f)
    print(f"Model saved to {MODEL_PATH}")

# Оценка модели
def evaluate_model(model, X_test, y_test):
    print("Evaluating model...")
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model accuracy: {accuracy:.2f}")

# Основной процесс выполнения
def main():
    download_data()
    X_train, X_test, y_train, y_test = prepare_data()
    model = train_model(X_train, y_train)
    save_model(model)
    evaluate_model(model, X_test, y_test)

if __name__ == "__main__":
    main()
