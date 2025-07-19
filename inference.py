import catboost
import pandas as pd
import os
import joblib

import numpy as np
import pickle as pkl


# Темлпейт для обработки входных данных
def prepare_and_load_data(some_data):
    pass


features = prepare_and_load_data('data.csv')

models = []

directory_path = "model"

# Проходим по всем файлам в директории
for filename in os.listdir(directory_path):
    file_path = os.path.join(directory_path, filename)
    file_ext = os.path.splitext(filename)[1].lower()
    
    # Загрузка моделей scikit-learn
    if file_ext in ['.pkl', '.joblib']:
        model = joblib.load(file_path)
        models.append(('sklearn', filename, model))

predictions = []

for model in models:
    predictions.append(model.predict(features))

print(predictions)
