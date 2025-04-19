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
    test_filename = 'test_temperature_data_scaled.csv'
    
    test_data = load_data(test_folder, test_filename)
    
    feature_columns = ['temperature', 'humidity', 'is_cloudy', 'season_spring', 'season_summer', 'season_fall']
    
    X_test = test_data[feature_columns]
    y_test = test_data['temperature']
    
    model_filepath = "./model_temperature.pkl"
    with open(model_filepath, 'rb') as f:
        model = pickle.load(f)
    
    y_pred = model.predict(X_test)
    
    mse = mean_squared_error(y_test, y_pred)
    print(f"Model test MSE is: {mse:.3f}")

if __name__ == "__main__":
    test_model()

