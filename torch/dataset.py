import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset
import numpy as np

def download_dataset(filename):
	dataset = pd.read_csv(filename, delimiter=';')
	return dataset

def get_data(dataset):
	data = dataset[list(dataset.columns)]
	return clean_data(data)

def clean_data(data):
	pd.options.mode.chained_assignment = None
	attributes = [
		('sotano', ['Ex', 'Gd', 'TA', 'Fa', 'Na']),
		('calefaccion', ['Floor', 'GasA', 'GasW', 'Wall', 'OthW', 'Grav']),
		('aire_acondicionado', ['N', 'Y'])
	]
	for id, reg in enumerate(data['sotano']):
		if isinstance(reg, float):
			data['sotano'][id] = 'Na'
		else:
			data['sotano'][id] = reg
	for attribute in attributes:
		for idr, reg in enumerate(data[attribute[0]]):
			for ida, val in enumerate(attribute[1]):
				if reg == val:
					data[attribute[0]][idr] = ida
	return data

def split_data_x_y(data):
	data_input = []
	data_output = []
	for reg in data:
		data_input.append(reg[:-1])
		data_output.append(reg[-1])
	return np.array(data_input), np.array(data_output)

def split_data_train_test(data, split_size):
	len_train = int(len(data)*split_size)
	return data[:len_train], data[len_train:]

# Private
__scaler = None

class Normalizer():
	def transform(data):
		global __scaler
		__scaler = MinMaxScaler()
		return __scaler.fit_transform(data)

	# Obtener el Precio
	# __scaler.inverse_transform: solicita un vector de registros
	# -> donde cada registro tiene 16 características (vector de tamaño 16)
	# -> cada caracteristica es un valor de cada columna de nuestro dataset
	def inverse_transform(normalized_price, size = 16):
		register = [0 for i in range(size - 1)]
		register.append(normalized_price)
		return int(round(__scaler.inverse_transform([register])[0][-1]))

class TimeSeriesDataset(Dataset):
	def __init__(self, x, y):
		self.x = x.astype(np.float32)
		self.y = y.astype(np.float32)
		
	def __len__(self):
		return len(self.x)

	def __getitem__(self, idx):
		return (self.x[idx], self.y[idx])