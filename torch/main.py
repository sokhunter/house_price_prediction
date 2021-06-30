import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
import json
import dataset
import algorithm

CONFIG = json.load(open('config.json'))
# Dataset
Dataset = dataset.download_dataset(CONFIG['data']['filename'])
data = dataset.get_data(Dataset)
normalize_data = dataset.Normalizer.transform(data)
data_x, data_y = dataset.split_data_x_y(normalize_data)

# Divicion data train y data test
data_x_train, data_x_test = dataset.split_data_train_test(data_x, CONFIG["data"]["train_split_size"])
data_y_train, data_y_test = dataset.split_data_train_test(data_y, CONFIG["data"]["train_split_size"])
dataset_train = dataset.TimeSeriesDataset(data_x_train, data_y_train)
dataset_test = dataset.TimeSeriesDataset(data_x_test, data_y_test)
train_dataloader = DataLoader(dataset_train, CONFIG["training"]["batch_size"], shuffle=True)
test_dataloader = DataLoader(dataset_test, CONFIG["training"]["batch_size"], shuffle=True)

# Modelo
model = algorithm.Model(config_model=CONFIG["model"], config_training=CONFIG["training"])

# Entrenamiento
for epoch in range(CONFIG["training"]["num_epoch"]):
    loss_train, lr_train = model.run_epoch(train_dataloader, is_training=True)
    loss_test, lr_test = model.run_epoch(test_dataloader)

# Prediccion