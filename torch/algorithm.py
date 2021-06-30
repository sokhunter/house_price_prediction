import torch.nn as nn
import torch.optim as optim

class NNModel(nn.Module):
	def __init__(self, config_model, config_training):
		super().__init__()
		self.hidden_layer_size = config_model['hidden_layer_size']
		self.device=config_training['device']

		self.Linear_1 = nn.Linear(config_model['input_size'], config_model['hidden_layer_size'])
		self.ReLU = nn.ReLU()
		self.LSTM = nn.LSTM(input_size=config_model['hidden_layer_size'], hidden_size=config_model['hidden_layer_size'], num_layers=config_model['num_layers'], batch_first=True)
		self.dropout = nn.Dropout(config_model['dropout'])
		self.Linear_2 = nn.Linear(config_model['num_layers']*config_model['hidden_layer_size'], config_model['output_size'])

		self.coste = nn.MSELoss()
		self.optimizer = optim.Adam(self.parameters(), lr=config_training['learning_rate'], betas=(0.9, 0.98), eps=1e-9)
		self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=config_training['scheduler_step_size'], gamma=0.1)

		self.init_weights()
		self.to(config_training['device'])
    
	def init_weights(self):
		for name, param in self.LSTM.named_parameters():
			if 'bias' in name:
				nn.init.constant_(param, 0.0)
			elif 'weight_ih' in name:
				nn.init.kaiming_normal_(param)
			elif 'weight_hh' in name:
				nn.init.orthogonal_(param)

	def forward(self, x):
		batchsize = x.shape[0]
		# Primera capa
		# Función de activación(Capa Lineal 1)
		# -> ReLu(WX + b) = ouput
		out = self.ReLU(self.Linear_1(x))

		# Segunda capa
		# Para la Capa LSTM no aplicaremos función de activación, pero si dropout
		# El output de la "Capa Lineal 1" será nuestra entrada para la "Capa LSTM"
		# -> LSTM(x) = WX + b = ouput
		# -> Dropout(output) = output con olvido
		lstm_out, (h_t, c_t) = self.LSTM(out)
		# reshape output from hidden cell into [batch, features] for `linear_2`
		out = h_t.permute(1, 0, 2).reshape(batchsize, -1)

		# Tercera capa
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