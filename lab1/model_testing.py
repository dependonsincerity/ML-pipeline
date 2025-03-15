import os
import pickle
import pandas as pd
from sklearn.metrics import mean_squared_error

# Функция для чтения данных из CSV-файла
def load_data(folder, filename):
    filepath = os.path.join(folder, filename)
    return pd.read_csv(filepath)

# Основная функция для тестирования модели
def test_model():
    test_folder = 'test'
    test_filename = 'test_temperature_data_scaled.csv'  # Изменено имя файла
    
    # Загрузка данных
    test_data = load_data(test_folder, test_filename)
    
    # Разделение на признаки и целевую переменную
    X_test = test_data[['temperature', 'humidity', 'wind_speed']]  # Используем новые признаки
    y_test = test_data['temperature']  # Целевая переменная - температура (или другой столбец)
    
    # Загрузка модели из файла
    model_filepath = "./model_temperature.pkl"  # Изменено имя файла модели
    with open(model_filepath, 'rb') as f:
        model = pickle.load(f)
    
    # Предсказание
    y_pred = model.predict(X_test)
    
    # Вычисление метрики (MSE)
    mse = mean_squared_error(y_test, y_pred)
    print(f"Model test MSE is: {mse:.3f}")

if __name__ == "__main__":
    test_model()
