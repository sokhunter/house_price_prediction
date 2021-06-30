import torch.nn as nn
import torch.optim as optim

class Model(nn.Module):
	def __init__(self, config_model, config_training):
		super().__init__()
		# self.hidden_layer_size = config_model['hidden_layer_size']
		self.device=config_training['device']
		
		# Definicion de capas
		# nn.Linear(tamaño del input, cantidad de neuronas)
		self.Linear_1 = nn.Linear(config_model['input_size'], config_model['hidden_layer_size'])
		self.Hidden_Layer_1 = nn.Linear(config_model['hidden_layer_size'], config_model['num_layers']*config_model['hidden_layer_size'])
		self.Hidden_Layer_2 = nn.Linear(config_model['num_layers']*config_model['hidden_layer_size'], config_model['num_layers']*config_model['hidden_layer_size'])
		self.Linear_2 = nn.Linear(config_model['num_layers']*config_model['hidden_layer_size'], config_model['output_size'])

		# Definicion de funcion de activacion
		self.ReLU = nn.ReLU()
		# Tecnica para evitar overfitting
		self.dropout = nn.Dropout(config_model['dropout'])
		# Funcion de costo
		self.coste = nn.MSELoss()
		# Funcion de obtimizacion
		self.optimizer = optim.Adam(self.parameters(), lr=config_training['learning_rate'], betas=(0.9, 0.98), eps=1e-9)
		# Tecnica para que la taza de aprendizaje decaiga a medida que aprende el modelo
		self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=config_training['scheduler_step_size'], gamma=0.1)

		self.to(config_training['device'])

	def forward(self, x):
		batchsize = x.shape[0]
		# Primera capa
		# Función de activación(Capa Lineal 1)
		# -> ReLu(WX + b) = ouput
		out = self.ReLU(self.Linear_1(x))

		# Segunda capa (PRIMERA CAPA OCULTA)
		out = self.ReLU(self.Hidden_Layer_1(out))

		# Tercera capa (SEGUNDA CAPA OCULTA)
		out = self.ReLU(self.Hidden_Layer_2(out))

		# Cuarta capa
		# Para nuestra última capa "Capa Lineal 2" mantendremos el output sin cambios
		predictions = self.Linear_2(out)

		return predictions[:,-1]
	
	def run_epoch(self, dataloader, is_training=False):
		if is_training:
			self.train()
		else:
			self.eval()
		epoch_loss = 0
		for idx, (x, y) in enumerate(dataloader):
			if is_training:
				self.optimizer.zero_grad()
			
			batchsize = x.shape[0]
			x = x.to(self.device)
			y = y.to(self.device)

			out = self(x)
			loss = self.coste(out, y)
			if is_training:
				loss.backward()
				self.optimizer.step()
			
			epoch_loss += (loss.detach().item() / batchsize)
		lr = self.scheduler.get_last_lr()[0]
		return epoch_loss, lr