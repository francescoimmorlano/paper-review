import os
import numpy as np
import pandas as pd
import time
from datetime import datetime, timedelta
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.optimizers.legacy import Adam
from tensorflow.keras.regularizers import l1_l2
from netCDF4 import Dataset
import csv
from lib import *
from architectures import *
from variables import *

"""
Script for training one Deep Neural Network (DNN) on the simulation of one ESM under one SSP scenario.
Overall, 66 DNNs are trained, each on a different simulation and with the same hyperparameters
and architecture.
"""

columns_history_df = ['train_loss', 'val_loss']
columns_model_hyperparameters_df = ['train_directory_name', 'model', 'scenario', 'date_time', 'elapsed_time', 'epochs', 'batch_size', 'start_year_training', 'end_year_training', 'val_year', 'L1_regularization', 'L2_regularization', 'shuffle', 'scale_input', 'scale_output', 'norm_min', 'norm_max', 'train_percent', 'val_percent', 'opt', 'loss', 'weight_initializer', 'activation', 'CO2eq_climate_model', 'withAerosolForcing', 'use_observations']

ts = datetime.now()
ts_human = ts.strftime('%Y-%m-%d_%H-%M-%S')
print(f'\n\n************************************ First_Training_{ts_human} ************************************')

if demo_download:
    PATH_ANNUAL_SIMULATIONS_DIRECTORY = f'./Demo_download/Data/CMIP6_data/{variable}/Annual_uniform_remapped'
    PATH_FIRST_TRAINING = f'./Demo_download/Experiments/First_Training/First_Training_{ts_human}'
elif demo_no_download:
    PATH_ANNUAL_SIMULATIONS_DIRECTORY = f'./Demo_no_download/Data/CMIP6_data/{variable}/Annual_uniform_remapped'
    PATH_FIRST_TRAINING = f'./Demo_no_download/Experiments/First_Training/First_Training_{ts_human}'
else:
    PATH_ANNUAL_SIMULATIONS_DIRECTORY = f'./Source_data/CMIP6_data/{variable}/Annual_uniform_remapped'
    PATH_FIRST_TRAINING = f'./Experiments/First_Training/First_Training_{ts_human}'

PATH_MODELS = f'{PATH_FIRST_TRAINING}/Models'
PATH_HISTORIES = f'{PATH_FIRST_TRAINING}/Histories'
PATH_PLOTS = f'{PATH_FIRST_TRAINING}/Plots'
PATH_HYPERPARAMETERS = f'{PATH_FIRST_TRAINING}/Hyperparameters'
PATH_PREDICTIONS_YEAR_2095 = f'{PATH_FIRST_TRAINING}/Predictions_on_year_2095'
if not os.path.exists(PATH_FIRST_TRAINING): os.makedirs(PATH_FIRST_TRAINING)
if not os.path.exists(PATH_MODELS): os.mkdir(PATH_MODELS)
if not os.path.exists(PATH_HISTORIES): os.mkdir(PATH_HISTORIES)
if not os.path.exists(PATH_PLOTS): os.mkdir(PATH_PLOTS)
if not os.path.exists(PATH_HYPERPARAMETERS): os.mkdir(PATH_HYPERPARAMETERS)
if not os.path.exists(PATH_PREDICTIONS_YEAR_2095): os.mkdir(PATH_PREDICTIONS_YEAR_2095)

##################### CALLBACKS #####################
save_predictions_on_validation_set = True

n_channels = 1

epochs = 500
batch_size = 8
n_filters = 128

shuffle = (True, 42)

start_year_training = 1850

# Year reserved for validation 
val_year = 2095

first_ssp_year = 2015

# Decide if normalize input and/or output to a range set by the feature_range variable
scale_input = True
scale_output = True
feature_range = (0,1)

# This is set to (1,0) to leave all the samples in the training set and then extract the ones reserved for the validation. Then shuffle is applied
train_val_ratio = (1, 0)

loss = 'mae'

l1_regularization = 0
l2_regularization = 0
regularizer = l1_l2(l1=l1_regularization, l2=l2_regularization)

weight_initializer = 'glorot_uniform'

lr = 1e-4
optim = Adam(learning_rate=lr, beta_1=0.9, beta_2=0.999, epsilon=1e-8)

alpha_constant = 0.2
hidden_layers_activation_function = 'leaky_relu'
output_layer_activation_function = 'sigmoid'

if hidden_layers_activation_function == 'leaky_relu' or hidden_layers_activation_function == 'prelu':
    activation_functions = f'Hidden Layers: {hidden_layers_activation_function} - alpha constant: {alpha_constant}'
else:
    activation_functions = f'Hidden Layers: {hidden_layers_activation_function}'

if output_layer_activation_function == 'leaky_relu' or output_layer_activation_function == 'prelu':
    activation_functions = f'{activation_functions}; Output Layer: {output_layer_activation_function} - {alpha_constant}'
else:
    activation_functions = f'{activation_functions}; Output Layer: {output_layer_activation_function}'

CO2eq_climate_model = 'MCE-v1-2'
withAerosolForcing = True

# Lowest and highest temperature value in the CMIP6 simulations used
y_min = 212.1662
y_max = 317.38766

class PerformancePlotCallback(Callback):
    def __init__(self, val_X, val_y, val_year, model_name, short_scenario, scenario, y_min, y_max):
        self.val_X = val_X
        self.val_y = val_y
        self.val_year = val_year
        self.model_name = model_name
        self.short_scenario = short_scenario
        self.scenario = scenario
        self.y_min = y_min
        self.y_max = y_max
        
    def on_epoch_end(self, epoch, logs={}):
        if ((epoch < 20) or (epoch < 50 and epoch % 5 == 0) or (epoch > 50 and epoch % 50 == 0) or (epoch == epochs-1)):
            val_y_pred = self.model.predict(self.val_X)

            if scale_output:
                val_y_pred_denorm = denormalize_img(val_y_pred[0,:,:,0], feature_range[0], feature_range[1], self.y_min, self.y_max)
                val_y_denorm = denormalize_img(self.val_y[0,:,:,0], feature_range[0], feature_range[1], self.y_min, self.y_max)
            else:
                val_y_pred_denorm = val_y_pred[0,:,:,0]
                val_y_denorm = self.val_y[0,:,:,0]

            PATH_MODEL_SCENARIO_PREDICTIONS = f'{PATH_PREDICTIONS_YEAR_2095}/{variable_short}_{model}_{short_scenario}'
            if not os.path.exists(PATH_MODEL_SCENARIO_PREDICTIONS): os.mkdir(PATH_MODEL_SCENARIO_PREDICTIONS)
            
            PATH_TO_SAVE_PREDICTION = f'{PATH_MODEL_SCENARIO_PREDICTIONS}/{variable_short}_{self.model_name}_{self.short_scenario}_epoch-{epoch}_year-{self.val_year}_{ts_human}_val_set_prediction'

            with open(f'{PATH_TO_SAVE_PREDICTION}.csv',"w+") as my_csv:
                csvWriter = csv.writer(my_csv,delimiter=',')
                csvWriter.writerows(val_y_pred_denorm[:,:])
            
            plot_prediction_mae_map(val_y_denorm, val_y_pred_denorm, self.model_name, self.scenario, epoch, self.val_year, f'{PATH_TO_SAVE_PREDICTION}.png')
                  
simulations_list = os.listdir(PATH_ANNUAL_SIMULATIONS_DIRECTORY)
simulations_list.sort()

# Read CO2e values for each SSP scenario
_, X_ssp245, _ = read_CO2_equivalent('./', 'ssp245', CO2eq_climate_model, withAerosolForcing)
_, X_ssp370, _ = read_CO2_equivalent('./', 'ssp370', CO2eq_climate_model, withAerosolForcing)
_, X_ssp585, _ = read_CO2_equivalent('./', 'ssp585', CO2eq_climate_model, withAerosolForcing)

X_ssp_list = []
X_ssp_list.append(X_ssp245)
X_ssp_list.append(X_ssp370)
X_ssp_list.append(X_ssp585)

return_list = compute_values_for_scaling(X_ssp_list)

X_min_list = return_list[0]
X_max_list = return_list[1]

for idx_model, model in enumerate(models_list):
    for idx_short_scenario, short_scenario in enumerate(short_scenarios_list):
        start_time = time.time()

        scenario = f'SSP{short_scenario[-3]}-{short_scenario[-2]}.{short_scenario[-1]}'
        print(f'{idx_model+1}. Model: {model} - Scenario: {scenario}\n\n')

        PATH_HYPERPARAMETERS_CSV = f'{PATH_HYPERPARAMETERS}/{variable_short}_{model}_{short_scenario}_{ts_human}_models_hyperparameters.csv'
        if not os.path.exists(PATH_HYPERPARAMETERS_CSV):
            pd.DataFrame(columns=columns_model_hyperparameters_df).to_csv(PATH_HYPERPARAMETERS_CSV)
        
        df_hypp = pd.read_csv(PATH_HYPERPARAMETERS_CSV, dtype='str', usecols=columns_model_hyperparameters_df)

        HISTORICAL_SIMULATION_FILENAME = [s for s in simulations_list if (model in s and 'historical' in s)][0]
        SSP_SIMULATION_FILENAME  = [s for s in simulations_list if (model in s and short_scenario in s)][0]
        
        nc_historical_data = Dataset(f'{PATH_ANNUAL_SIMULATIONS_DIRECTORY}/{HISTORICAL_SIMULATION_FILENAME}', mode='r+', format='NETCDF3_CLASSIC')
        nc_ssp_data = Dataset(f'{PATH_ANNUAL_SIMULATIONS_DIRECTORY}/{SSP_SIMULATION_FILENAME}', mode='r+', format='NETCDF3_CLASSIC')

        n_historical_years = nc_historical_data[variable_short].shape[0]
        n_ssp_years = nc_ssp_data[variable_short].shape[0]
        n_lats = nc_historical_data['lat'].shape[0]
        n_lons = nc_historical_data['lon'].shape[0]

        # If the simulation goes up to 2100 
        if n_ssp_years == 86: 
            simulation_array = np.zeros((2100-start_year_training+1, n_lats, n_lons))
            years = np.arange(start_year_training, 2100+1, 1, dtype=int)
            years = years.reshape(2100-start_year_training+1,1,1)
        # If the simulation goes up to 2099
        elif n_ssp_years == 85:
            simulation_array = np.zeros((2099-start_year_training+1, n_lats, n_lons))
            years = np.arange(start_year_training, 2099+1, 1, dtype=int)
            years = years.reshape(2099-start_year_training+1,1,1)
        # If the simulation goes up to 2098
        elif n_ssp_years == 84:
            simulation_array = np.zeros((2098-start_year_training+1, n_lats, n_lons))
            years = np.arange(start_year_training, 2098+1, 1, dtype=int)
            years = years.reshape(2098-start_year_training+1,1,1)
        if start_year_training <= 2014:
            simulation_array[:(2014-start_year_training+1),:,:] = nc_historical_data[variable_short][(start_year_training-1850):,:,:]
            simulation_array[(2014-start_year_training+1):,:,:] = nc_ssp_data[variable_short][:,:,:]
        else:
            simulation_array[:,:,:] = nc_ssp_data[variable_short][(start_year_training-2015):,:,:]

        nc_historical_data.close()
        nc_ssp_data.close()
        
        # If the simulation goes up to 2100 
        if (n_ssp_years == 86):
            X_co2 = X_ssp_list[idx_short_scenario][:].copy()
        # If the simulation goes up to 2099
        elif (n_ssp_years == 85): 
            X_co2 = X_ssp_list[idx_short_scenario][:-1].copy()
        # If the simulation goes up to 2098
        elif (n_ssp_years == 84): 
            X_co2 = X_ssp_list[idx_short_scenario][:-2].copy()
        
        val_year_idx = val_year - start_year_training
        val_year = years[val_year_idx,0,0]
        
        val_X = X_co2[val_year_idx]
        X_co2.pop(val_year_idx)
        
        val_y = simulation_array[val_year_idx,:,:]
        simulation_array = np.delete(simulation_array, val_year_idx, axis=0)
        
        X_co2 = np.array(X_co2)
        val_X = np.array(val_X)

        train, _, _ = sets_setup(X_co2, simulation_array, train_val_ratio, shuffle)

        train_X, train_y = train[0], train[1]

        if scale_input:
            X_min = X_min_list[idx_short_scenario]
            X_max = X_max_list[idx_short_scenario]
            train_X = normalize_img(train_X, feature_range[0], feature_range[1], X_min, X_max).reshape(-1,1)
            val_X = normalize_img(val_X, feature_range[0], feature_range[1], X_min, X_max).reshape(-1,1)
        elif not scale_input:
            train_X = train_X.reshape(-1, 1)
            val_X = val_X.reshape(-1, 1)
        if scale_output:
            train_y = normalize_img(train_y, feature_range[0], feature_range[1], y_min, y_max).reshape(-1, simulation_array.shape[1], simulation_array.shape[2], n_channels)
            val_y = normalize_img(val_y, feature_range[0], feature_range[1], y_min, y_max).reshape(-1,simulation_array.shape[1], simulation_array.shape[2], n_channels)
        else:
            train_y = train_y.reshape(-1, simulation_array.shape[1], simulation_array.shape[2], n_channels)
            val_y = val_y.reshape(-1, simulation_array.shape[1], simulation_array.shape[2], n_channels)

        print("\nTrain X shape: ", train_X.shape)
        print("Train y shape: ", train_y.shape)

        print("Val X shape: ", val_X.shape)
        print("Val y shape: ", val_y.shape)

        print('\n******************************************************')

        NN_model = custom_CNN_transpose(n_filters, weight_initializer, regularizer, hidden_layers_activation_function, output_layer_activation_function, alpha_constant)
        NN_model.compile(loss=loss, optimizer=optim)
        NN_model.summary()

        if (save_predictions_on_validation_set):
            if (not scale_output):
                y_min = 0
                y_max = 0
            save_validation_predictions_callback = PerformancePlotCallback(val_X, val_y, val_year, model, short_scenario, scenario, y_min, y_max)
        else:
            save_validation_predictions_callback = []

        callbacks = [save_validation_predictions_callback]
    
        history = NN_model.fit(train_X,
                                train_y,
                                epochs=epochs,
                                batch_size=batch_size,
                                validation_data=(val_X,val_y),
                                use_multiprocessing=True,
                                callbacks=callbacks)

        NN_model_name = f'{PATH_MODELS}/{variable_short}_{model}_{short_scenario}_{ts_human}_model.tf'
        NN_model.save(NN_model_name)

        pd.DataFrame(np.array([history.history["loss"],
                            history.history["val_loss"]]).T, columns=columns_history_df).to_csv(f'{PATH_HISTORIES}/{variable_short}_{model}_{short_scenario}_{ts_human}_history.csv')

        elapsed = (time.time() - start_time)
        elapsed_time = str(timedelta(seconds=elapsed))

        path_to_save_plot = f'{PATH_PLOTS}/{variable_short}_{model}_{short_scenario}_{ts_human}_trainvalcurve.png'
        plot_train_val_loss_curve(history.history["loss"], history.history["val_loss"], loss, path_to_save_plot)

        df_hypp.loc[len(df_hypp.index)] = [f'First_Training_{ts_human}', model, scenario, ts_human, elapsed_time, epochs, batch_size, start_year_training, '2098', val_year, l1_regularization, l2_regularization, shuffle[0], scale_input, scale_output, feature_range[0], feature_range[1], train_val_ratio[0], train_val_ratio[1], optim.get_config(), loss, weight_initializer, activation_functions, CO2eq_climate_model, withAerosolForcing, 'False']

        df_hypp.to_csv(PATH_HYPERPARAMETERS_CSV)

        val_y_pred = NN_model.predict(val_X)

        if scale_output:
            val_y_pred_denorm = denormalize_img(val_y_pred[0,:,:,0], feature_range[0], feature_range[1], y_min, y_max)
            val_y_denorm = denormalize_img(val_y[0,:,:,0], feature_range[0], feature_range[1], y_min, y_max)
        else:
            val_y_pred_denorm = val_y_pred[0,:,:,0]
            val_y_denorm = val_y[0,:,:,0]

        plot_prediction_mae_map(val_y_denorm, val_y_pred_denorm, model, scenario, epochs, val_year, f'{PATH_PLOTS}/{variable_short}_{model}_{short_scenario}_{ts_human}_pred-on-val_end-epoch.png')

        print('\n\n************************************ PREDICTION ON VALIDATION SET DONE ************************************\n')