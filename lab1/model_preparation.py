import os
import pickle
import pandas as pd
from sklearn.linear_model import LinearRegression

# Функция для чтения данных из CSV-файла
def load_data(folder, filename):
    filepath = os.path.join(folder, filename)
    return pd.read_csv(filepath)

# Основная функция для обучения модели
def prepare_model():
    train_folder = 'train'
    train_filename = 'train_temperature_data_scaled.csv'  # Изменено имя файла
    
    # Загрузка данных
    train_data = load_data(train_folder, train_filename)
    
    # Разделение на признаки и целевую переменную
    X_train = train_data[['temperature', 'humidity', 'wind_speed']]  # Используем новые признаки
    y_train = train_data['temperature']  # Целевая переменная - температура (или другой столбец)
    
    # Создание и обучение модели
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Сохранение модели в файл
    model_filepath = "model_temperature.pkl"  # Изменено имя файла модели
    with open(model_filepath, 'wb') as f:
        pickle.dump(model, f)
    
    print(f"Model saved to {model_filepath}")

if __name__ == "__main__":
    prepare_model()
