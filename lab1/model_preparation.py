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
    train_filename = 'train_temperature_data_scaled.csv'
    
    train_data = load_data(train_folder, train_filename)
    
    # Обновляем список признаков
    feature_columns = ['temperature', 'humidity', 'is_cloudy', 'season_spring', 'season_summer', 'season_fall']
    
    X_train = train_data[feature_columns]
    y_train = train_data['temperature']
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    model_filepath = "model_temperature.pkl"
    with open(model_filepath, 'wb') as f:
        pickle.dump(model, f)
    
    print(f"Model saved to {model_filepath}")

if __name__ == "__main__":
    prepare_model()