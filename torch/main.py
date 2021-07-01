import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
import torch
import json
import dataset
import algorithm

class Predictus():
    def __init__ (self):
        self.CONFIG = json.load(open('config.json'))
        self.error_range = 0

        self.area = 8635
        self.material = 5
        self.condicion = 5
        self.anio_construccion = 1948
        self.anio_remodelacion = 2001
        self.sotano = 'TA'
        self.calefaccion = 'GasA'
        self.aire_acondicionado = 'Y'
        self.area_piso_1 = 1072
        self.area_construida = 1285
        self.banios_con_ducha = 1
        self.banios_sin_ducha = 0
        self.dormitorios = 2
        self.chimeneas = 0
        self.area_garage = 240
        self.area_piscina = 0

    def ask(self):
        logo = '''
    __________                    .___.__        __                
    \______   \_______   ____   __| _/|__| _____/  |_ __ __  ______
    |     ___/\_  __ \_/ __ \ / __ | |  |/ ___\   __\  |  \/  ___/
    |    |     |  | \/\  ___// /_/ | |  \  \___|  | |  |  /\___ \ 
    |____|     |__|    \___  >____ | |__|\___  >__| |____//____  >
                            \/     \/         \/                \/ '''
        print("**********************************************************************")
        print(logo)
        print("**********************************************************************")
        print("Predictus lo ayudará a valuar el precio de su próximo hogar\n")
        print("Modo de uso: Debera responder todas las preguntas con valores numericos (NO LETRAS) recuerde que todas las areas usadas son en pulgadas\n")
        input('Presione enter para iniciar el cuestionario... \n')

        self.area = input('¿Cúal es el área total? (pulgadas): ')
        self.material = input('Califique del 1 al 10 la calidad y acabado del material: ')
        self.condicion = input('Califique del 1 al 10 la condicion de la casa: ')
        self.anio_construccion = input('¿Cúal es el año de construccion de la casa? ')
        self.anio_remodelacion = input('¿Cúal es el año de construccion de la casa? ')
        self.sotano = input('''Cúal es la altura del sotano segun las siguientes categorias
        Ex	Excelente (100+ pies)	
        Gd	Bueno (90-99 pies)
        TA	Tipico (80-89 pies)
        Fa	Justo (70-79 pies)
        Na	Sin sotano\n''')
        self.calefaccion = input('''Cúal es el tipo de calefacción segun las siguientes categorias
        Floor	Horno de piso
        GasA	Horno de aire caliente forzado a gas
        GasW	Agua caliente a gas o calor a vapor
        Grav	Horno de gravedad	
        OthW	Calor de agua caliente o vapor que no sea gas
        Wall	Horno de pared\n''')
        self.aire_acondicionado = input('¿Tiene aire acondicionado? (Y / N): ')
        self.area_piso_1 = input('¿Cúal es el área construida del primer piso? (pulgadas): ')
        self.area_construida = input('¿Cúal es el área construida todal, incluye todos los pisos?: (pulgadas)')
        self.banios_con_ducha = input('¿Cuantos baños con ducha tiene? : ')
        self.banios_sin_ducha = input('¿Cuantos baños sin ducha tiene?: ')
        self.dormitorios = input('¿Cuantos dormitorios tiene?: ')
        self.chimeneas = input('¿Cuantas chimeneas tiene?: ')
        self.area_garage = input('¿Cúal es el área del garage? (pulgadas): ')
        self.area_piscina = input('¿Cúal es el área de la piscina? (pulgadas): ')
        print('\nPorfavor espere mientras calculamos el valor de su próximo hogar.\nEste proceso puede tomar unos minutos')
        print('Calculando...')
    
    def load_data(self):
        # Dataset
        Dataset = dataset.download_dataset(self.CONFIG['data']['filename'])
        data = dataset.get_data(Dataset)
        normalize_data = dataset.Normalizer.transform(data)
        data_x, data_y = dataset.split_data_x_y(normalize_data)

        # Divicion data train y data test
        data_x_train, data_x_test = dataset.split_data_train_test(data_x, self.CONFIG["data"]["train_split_size"])
        data_y_train, data_y_test = dataset.split_data_train_test(data_y, self.CONFIG["data"]["train_split_size"])
        dataset_train = dataset.TimeSeriesDataset(data_x_train, data_y_train)
        dataset_test = dataset.TimeSeriesDataset(data_x_test, data_y_test)
        self.train_dataloader = DataLoader(dataset_train, self.CONFIG["training"]["batch_size"], shuffle=True)
        self.test_dataloader = DataLoader(dataset_test, self.CONFIG["training"]["batch_size"], shuffle=True)

    def model(self):
        # Modelo
        self.model = algorithm.Model(config_model=self.CONFIG["model"], config_training=self.CONFIG["training"])

    def train(self):
        # Entrenamiento
        for epoch in range(self.CONFIG["training"]["num_epoch"]):
            loss_train, lr_train = self.model.run_epoch(self.train_dataloader, is_training=True)
            loss_test, lr_test = self.model.run_epoch(self.test_dataloader)
    
    def test(self):
        # Verificacion de la prediccion del modelo
        i=0
        for idx, (x, y) in enumerate(self.test_dataloader):
            i+=1
            x = x.to(self.CONFIG["training"]["device"])
            out = self.model(x)
            out_vector = out.detach().numpy()
            y_vector = y.numpy()
            for i in range(len(y_vector)):
                price_predict = dataset.Normalizer.inverse_transform(out_vector[i], self.CONFIG["model"]["input_size"]+1)
                price_real = dataset.Normalizer.inverse_transform(y_vector[i], self.CONFIG["model"]["input_size"]+1)
                self.error_range += abs(1-(price_predict/price_real))
            self.error_range = round(100 * self.error_range / i, 2)
            break

    def prediccion(self):
        # Prediccion
        data = np.array([[self.area, self.material, self.condicion, self.anio_construccion, self.anio_remodelacion, self.sotano, self.calefaccion, self.aire_acondicionado, self.area_piso_1, self.area_construida, self.banios_con_ducha, self.banios_sin_ducha, self.dormitorios, self.chimeneas, self.area_garage, self.area_piscina, 0]])
        data = pd.DataFrame(data, columns=['area', 'material', 'condicion', 'anio_construccion', 'anio_remodelacion', 'sotano', 'calefaccion',  'aire_acondicionado', 'area_construida_piso_1', 'area_construida', 'banios', 'banios_sin_ducha', 'dormitorios', 'chimeneas', 'area_garage', 'area_piscina', 'precio'])
        data = dataset.get_data(data)
        print(data)
        data = dataset.Normalizer.transform(data)
        features, _ = dataset.split_data_x_y(data)
        x = torch.tensor(features).float().to(self.CONFIG["training"]["device"])
        out = self.model(x)
        out_vector = out.detach().numpy()
        price_predict = dataset.Normalizer.inverse_transform(out_vector[0], self.CONFIG["model"]["input_size"]+1)
        return price_predict



form = Predictus()
# form.ask()
form.load_data()
form.model()
form.train()
form.test()
price = form.prediccion()
print("El precio de la casa es de {} con un porcentaje de error de {}".format(price, form.error_range))