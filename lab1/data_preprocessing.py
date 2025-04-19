import os
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Функция для чтения данных из CSV-файла
def load_data(folder, filename):
    filepath = os.path.join(folder, filename)
    return pd.read_csv(filepath)

# Функция для сохранения данных в CSV-файл
def save_data_to_csv(data, folder, filename):
    filepath = os.path.join(folder, filename)
    data.to_csv(filepath, index=False)
    print(f"Preprocessed data saved to {filepath}")

# Основная функция для предобработки данных
def preprocess_data():
    train_folder = 'train'
    test_folder = 'test'
    
    train_filename = 'train_temperature_data.csv'
    test_filename = 'test_temperature_data.csv'
    
    # Загрузка данных
    train_data = load_data(train_folder, train_filename)
    test_data = load_data(test_folder, test_filename)
    
    # One-hot encoding для season
    train_data = pd.get_dummies(train_data, columns=['season'], drop_first=False)
    test_data = pd.get_dummies(test_data, columns=['season'], drop_first=False)
    
    # Проверим, чтобы все сезоны были во всех наборах (если нет — добавляем)
    for season in ['season_winter', 'season_spring', 'season_summer', 'season_fall']:
        if season not in train_data.columns:
            train_data[season] = 0
        if season not in test_data.columns:
            test_data[season] = 0
    
    # Убедимся, что все season столбцы в одном порядке
    season_columns = ['season_winter', 'season_spring', 'season_summer', 'season_fall']
    train_data = train_data[['day', 'hour', 'temperature', 'humidity', 'is_cloudy'] + season_columns]
    test_data = test_data[['day', 'hour', 'temperature', 'humidity', 'is_cloudy'] + season_columns]
    
    # Масштабирование признаков
    features_to_scale = ['temperature', 'humidity']
    
    scaler = StandardScaler()
    train_data[features_to_scale] = scaler.fit_transform(train_data[features_to_scale])
    test_data[features_to_scale] = scaler.transform(test_data[features_to_scale])
    
    # Преобразуем бинарные признаки к типу int
    train_data['is_cloudy'] = train_data['is_cloudy'].astype(int)
    test_data['is_cloudy'] = test_data['is_cloudy'].astype(int)
    
    for season in season_columns:
        train_data[season] = train_data[season].astype(int)
        test_data[season] = test_data[season].astype(int)
    
    # Сохранение предобработанных данных
    save_data_to_csv(train_data, train_folder, 'train_temperature_data_scaled.csv')
    save_data_to_csv(test_data, test_folder, 'test_temperature_data_scaled.csv')

if __name__ == "__main__":
    preprocess_data()