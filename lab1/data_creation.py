import os
import numpy as np
import pandas as pd

# Создание директорий downloads/train и downloads/test
def create_directories():
    for folder in ['train', 'test']:
        if not os.path.exists(folder):
            os.makedirs(folder)

# Генерация данных о температуре воздуха
def generate_temperature_data(days=100, noise_level=0.1, anomaly_probability=0.05):
    data = []
    for day in range(days):
        for hour in range(24):  # Для каждого часа дня
            base_temperature = 10 + 20 * np.sin(2 * np.pi * day / 365)
            
            # Влияние времени суток
            if 6 <= hour < 12:  # Утро
                time_factor = 0.8
            elif 12 <= hour < 18:  # День
                time_factor = 1.2
            elif 18 <= hour < 24:  # Вечер
                time_factor = 1.5
            else:  # Ночь
                time_factor = 0.5
            
            # Влияние влажности
            humidity = np.random.uniform(0, 100)  # Случайная влажность (%)
            humidity_factor = 1 - (humidity / 100)  # Чем выше влажность, тем ниже температура
            
            # Влияние облачности
            is_cloudy = np.random.rand() < 0.3  # Вероятность облачности 30%
            cloud_factor = 0.9 if is_cloudy else 1.0
            
            # Сезонное изменение температуры
            season_factor = 1.0
            if day < 60:  # Зима
                season_factor = 0.8
            elif 60 <= day < 180:  # Весна
                season_factor = 1.1
            elif 180 <= day < 300:  # Лето
                season_factor = 1.2
            else:  # Осень
                season_factor = 0.9
            
            # Общий уровень температуры
            temperature = base_temperature * time_factor * humidity_factor * cloud_factor * season_factor
            
            # Добавление случайного шума
            noise = np.random.normal(0, noise_level * 5)
            
            # Добавление аномалий
            if np.random.rand() < anomaly_probability:
                anomaly = np.random.choice([-10, 10])  # Аномалия может быть как положительной, так и отрицательной
            else:
                anomaly = 0
            
            temperature += noise + anomaly
            temperature = max(-50, temperature)  # Температура не может быть ниже -50 градусов
            temperature = min(60, temperature)  # Температура не может быть выше 60 градусов
            
            data.append({
                'day': day,
                'hour': hour,
                'temperature': temperature,
                'humidity': humidity,
                'is_cloudy': int(is_cloudy),
                'season': 'winter' if day < 60 else ('spring' if 60 <= day < 180 else ('summer' if 180 <= day < 300 else 'fall'))
            })
    
    return pd.DataFrame(data)

# Сохранение данных в CSV-файлы
def save_data_to_csv(data, folder, filename):
    filepath = os.path.join(folder, filename)
    data.to_csv(filepath, index=False)
    print(f"Data saved to {filepath}")

# Основная функция для создания данных
def main():
    create_directories()
    
    # Генерация обучающего и тестового наборов данных
    train_data = generate_temperature_data(days=80, noise_level=0.1, anomaly_probability=0.05)
    test_data = generate_temperature_data(days=20, noise_level=0.2, anomaly_probability=0.1)
    
    # Сохранение данных
    save_data_to_csv(train_data, 'train', 'train_temperature_data.csv')
    save_data_to_csv(test_data, 'test', 'test_temperature_data.csv')

if __name__ == "__main__":
    main()
