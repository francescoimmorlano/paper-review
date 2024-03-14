"""
Author: Francesco Immorlano

Script for Transfer Learning on Observational Data
"""

from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers.legacy import Adam 
from keras.callbacks import Callback
from netCDF4 import Dataset
import numpy as np
import pandas as pd
import os
from datetime import datetime
import matplotlib.pyplot as plt
import time
from datetime import datetime
from datetime import timedelta
import csv
from lib import *
from variables import *

ts = datetime.now()
ts_human = ts.strftime('%Y-%m-%d_%H-%M-%S')
print(f'\n******************************************* Transfer_learning_obs_{ts_human} *******************************************')

loss = 'mae'

##################### CALLBACKS #####################
save_predictions_on_validation_set = True

compute_validation = True

n_BEST_datasets_per_model_scenario = 5

shuffling_dataset = (False, 42)
scale_input = True
scale_output = True

feature_range = (0,1)
n_channels = 1

# First training directory (only the directory name)
FIRST_TRAINING_DIRECTORY = 'First_Training_2024-03-14_09-26-50'

epochs = 2
batch_size = 16

lr = 1e-5
optim = Adam(learning_rate=lr, beta_1=0.9, beta_2=0.999, epsilon=1e-8)

start_year_training = 1979
end_year_training = 2022
start_year_test = end_year_training + 1
end_year_test = 2098
# Years reserved for validation
val_years_list = [2017, 2018, 2019, 2020]

n_training_years = end_year_training - start_year_training + 1
n_test_years = end_year_test - start_year_test + 1

CO2eq_climate_model = 'MCE-v1-2'
withAerosolForcing = True

class PerformancePlotCallback(Callback):
    def __init__(self, val_X, val_y, val_year, model_name, short_scenario, scenario, y_min, y_max, path_to_save):
        self.val_X = val_X
        self.val_y = val_y
        self.val_year = val_year
        self.model_name = model_name
        self.short_scenario = short_scenario
        self.scenario = scenario
        self.y_min = y_min
        self.y_max = y_max
        self.path_to_save = path_to_save
        
    def on_epoch_end(self, epoch, logs={}):
        if ((epoch < 20) or (epoch < 50 and epoch % 5 == 0) or (epoch > 50 and epoch % 50 == 0) or (epoch == epochs-1)):
            val_y_pred = self.model.predict(self.val_X)

            fig = plt.figure(figsize=(20,13))

            if scale_output:
                val_y_pred_denorm = denormalize_img(val_y_pred[0,:,:,0], feature_range[0], feature_range[1], self.y_min, self.y_max)
                val_y_denorm = denormalize_img(self.val_y[0,:,:,0], feature_range[0], feature_range[1], self.y_min, self.y_max)
            else:
                val_y_pred_denorm = val_y_pred[0,:,:,0]
                val_y_denorm = val_y[0,:,:,0]


            PATH_TO_SAVE_PREDICTION = f'{self.path_to_save}/{variable_short}_{model}_{self.short_scenario}_epoch-{epoch}_year-{self.val_year}_{ts_human}_val_set_prediction'


            with open(f'{PATH_TO_SAVE_PREDICTION}.csv',"w+") as my_csv:
                csvWriter = csv.writer(my_csv,delimiter=',')
                csvWriter.writerows(val_y_pred_denorm[:,:])

            plot_prediction_mae_map(val_y_denorm, val_y_pred_denorm, self.model_name, self.scenario, epoch, self.val_year, f'{PATH_TO_SAVE_PREDICTION}.png')

if compute_validation:
    columns_history_df = ['train_loss', 'val_loss']
else:
    columns_history_df = ['train_loss']

columns_model_hyperparameters_df = ['transf_learn_directory', 'first_train_directory', 'end_year_training', 'model', 'scenario', 'date_time', 'elapsed_loop_time', 'elapsed_train_time', 'epochs', 'batch_size', 'learning_rate', 'shuffle', 'scale_input', 'scale_output', 'norm_min', 'norm_max', 'y_min', 'y_max', 'CO2eq_climate_model', 'withAerosolForcing', 'use_observations']

if demo_download:
    ROOT_EXPERIMENTS = './Demo_download'
    ROOT_DATA = './Demo_download/Data'
elif demo_no_download:
    ROOT_EXPERIMENTS = './Demo_no_download'
    ROOT_DATA = './Demo_no_download/Data'
else:
    ROOT_EXPERIMENTS = '.'
    ROOT_DATA = f'./Source_data'

PATH_TRAINED_MODELS = f'{ROOT_EXPERIMENTS}/Experiments/First_Training/{FIRST_TRAINING_DIRECTORY}/Models'
PATH_TRANSFER_LEARNING_ON_OBSERVATIONS = f'{ROOT_EXPERIMENTS}/Experiments/Transfer_Learning_on_Observations/Transfer_learning_obs_{ts_human}'
PATH_HISTORIES = f'{PATH_TRANSFER_LEARNING_ON_OBSERVATIONS}/Histories'
PATH_MODELS = f'{PATH_TRANSFER_LEARNING_ON_OBSERVATIONS}/Models'
PATH_PLOTS = f'{PATH_TRANSFER_LEARNING_ON_OBSERVATIONS}/Plots'
PATH_HYPERPARAMETERS = f'{PATH_TRANSFER_LEARNING_ON_OBSERVATIONS}/Hyperparameters'
if not os.path.exists(PATH_TRANSFER_LEARNING_ON_OBSERVATIONS): os.makedirs(PATH_TRANSFER_LEARNING_ON_OBSERVATIONS)
if not os.path.exists(PATH_MODELS): os.mkdir(PATH_MODELS)
if not os.path.exists(PATH_HISTORIES): os.mkdir(PATH_HISTORIES)
if not os.path.exists(PATH_PLOTS): os.mkdir(PATH_PLOTS)
if not os.path.exists(PATH_HYPERPARAMETERS): os.mkdir(PATH_HYPERPARAMETERS)

trained_models_list = os.listdir(PATH_TRAINED_MODELS)
trained_models_list.sort()

_, X_ssp245, _ = read_CO2_equivalent('./', 'ssp245', CO2eq_climate_model, withAerosolForcing)
_, X_ssp370, _ = read_CO2_equivalent('./', 'ssp370', CO2eq_climate_model, withAerosolForcing)
_, X_ssp585, _ = read_CO2_equivalent('./', 'ssp585', CO2eq_climate_model, withAerosolForcing)

# Since we are considering observations, we use CO2eq values from 1979 up to 2098
X_ssp245_1979_2022 = X_ssp245[start_year_training-1850:]
X_ssp370_1979_2022 = X_ssp370[start_year_training-1850:]
X_ssp585_1979_2022 = X_ssp585[start_year_training-1850:]

X_ssp_list = []
X_ssp_list.append(X_ssp245_1979_2022)
X_ssp_list.append(X_ssp370_1979_2022)
X_ssp_list.append(X_ssp585_1979_2022)

return_list = compute_values_for_scaling(X_ssp_list)

X_min_list = return_list[0]
X_max_list = return_list[1]


for idx_model, model  in enumerate(models_list):
    for idx_short_scenario, scenario_short in enumerate(short_scenarios_list):
            for i in range(1,6):
                start_loop_time = time.time()

                scenario = f'SSP{scenario_short[-3]}-{scenario_short[-2]}.{scenario_short[-1]}'
                trained_model_filename = [m for m in trained_models_list if (model in m and scenario_short in m)][0]
                
                print('\n************************************************************************')
                print(f'\nTrained Model: {trained_model_filename}')
                print(f'\nModel to transfer learn: {model} - Scenario: {scenario} - index: {i}\n\n')

                start_time = time.time()

                PATH_BEST_DATA = f'{ROOT_DATA}/BEST_data/gaussian_noise_{n_BEST_datasets_per_model_scenario}/BEST_regridded_annual_1979-2022_Gaussian_noise_{model}_{scenario_short}_{i}.nc'
                
                nc_BEST_data = Dataset(f'{PATH_BEST_DATA}', mode='r+', format='NETCDF3_CLASSIC')
                n_BEST_years = nc_BEST_data['st'].shape[0]
                n_lats = nc_BEST_data['lat'].shape[0]
                n_lons = nc_BEST_data['lon'].shape[0]

                BEST_data_array = np.zeros((n_BEST_years, n_lats, n_lons))
                BEST_data_array[:,:,:] = nc_BEST_data['st'][:,:,:]
                nc_BEST_data.close()

                PATH_TEST_SET_PREDICTIONS = f'{PATH_PLOTS}/Test_set_predictions/{variable_short}_{model}_{scenario_short}_{i}'
                PATH_TRAINING_SET_PREDICTIONS = f'{PATH_PLOTS}/Training_set_predictions/{variable_short}_{model}_{scenario_short}_{i}'
                PATH_PREDICTIONS_YEAR_2021 = f'{PATH_TRANSFER_LEARNING_ON_OBSERVATIONS}/Predictions_on_year_2022/{variable_short}_{model}_{scenario_short}_{i}'
                if not os.path.exists(PATH_TEST_SET_PREDICTIONS): os.makedirs(PATH_TEST_SET_PREDICTIONS)
                if not os.path.exists(PATH_TRAINING_SET_PREDICTIONS): os.makedirs(PATH_TRAINING_SET_PREDICTIONS)
                if not os.path.exists(PATH_PREDICTIONS_YEAR_2021): os.makedirs(PATH_PREDICTIONS_YEAR_2021)

                train_X = np.array(X_ssp_list[idx_short_scenario][:n_training_years])
                train_X = train_X.reshape(n_training_years,1,1)

                test_X = np.array(X_ssp_list[idx_short_scenario][n_training_years:n_training_years+n_test_years])
                test_X = test_X.reshape(n_test_years,1,1)

                train_y = np.zeros((n_training_years, n_lats, n_lons))
                train_y[:,:,:] = BEST_data_array[:n_training_years,:,:]

                trained_model = load_model(f'{PATH_TRAINED_MODELS}/{trained_model_filename}') # The model trained during the First Training must be loaded every time
                K.set_value(trained_model.optimizer.lr, lr)

                if compute_validation:
                    n_val_years = len(val_years_list)
                    val_X = np.zeros((n_val_years,1,1))
                    val_y = np.zeros((n_val_years, n_lats, n_lons))
                    
                    idx_to_remove = []
                    for idx_val_year, val_year in enumerate(val_years_list):
                        val_X[idx_val_year] = train_X[val_year-start_year_training]
                        val_y[idx_val_year] = train_y[val_year-start_year_training,:,:]
                        idx_to_remove.append(val_year-start_year_training)

                    train_X = np.delete(train_X, idx_to_remove, axis=0)
                    train_y = np.delete(train_y, idx_to_remove, axis=0)

                # The shuffle is done only on the train set. It is not needed on the test set
                if shuffling_dataset[0]:
                    idx_array = np.arange(0, n_training_years, 1, dtype=int)
                    np.random.seed(shuffling_dataset[1])
                    np.random.shuffle(idx_array)
                    train_X_shuffle, train_y_shuffle = train_X[idx_array[:]], train_y[idx_array[:],:,:]
                else:
                    train_X_shuffle = train_X
                    train_y_shuffle = train_y

                if scale_input:
                    X_min = X_min_list[idx_short_scenario]
                    X_max = X_max_list[idx_short_scenario]
                    train_X = normalize_img(train_X, feature_range[0], feature_range[1], X_min, X_max).reshape(-1,1)
                    train_X_shuffle = normalize_img(train_X_shuffle, feature_range[0], feature_range[1], X_min, X_max).reshape(-1,1)
                    if compute_validation:
                        val_X = normalize_img(val_X, feature_range[0], feature_range[1], X_min, X_max).reshape(-1,1)
                    test_X = normalize_img(test_X, feature_range[0], feature_range[1], X_min, X_max).reshape(-1,1)
                else:
                    train_X = train_X.reshape(-1, 1)
                    train_X_shuffle = train_X_shuffle.reshape(-1, 1)
                    if compute_validation:
                        val_X = val_X.reshape(-1, 1)
                    test_X = test_X.reshape(-1, 1)

                if scale_output:
                    train_y = normalize_img(train_y, feature_range[0], feature_range[1], y_min, y_max).reshape(-1, BEST_data_array.shape[1], BEST_data_array.shape[2], n_channels)
                    train_y_shuffle = normalize_img(train_y_shuffle, feature_range[0], feature_range[1], y_min, y_max).reshape(-1, BEST_data_array.shape[1], BEST_data_array.shape[2], n_channels)   
                    if compute_validation:
                        val_y = normalize_img(val_y, feature_range[0], feature_range[1], y_min, y_max).reshape(-1, BEST_data_array.shape[1], BEST_data_array.shape[2], n_channels)
                else:
                    train_y = train_y.reshape(-1, BEST_data_array.shape[1], BEST_data_array.shape[2], n_channels)
                    train_y_shuffle = train_y_shuffle.reshape(-1, BEST_data_array.shape[1], BEST_data_array.shape[2], n_channels)
                    if compute_validation:
                        val_y = val_y.reshape(-1, BEST_data_array.shape[1], BEST_data_array.shape[2], n_channels)

                print("\nTrain X shape: ", train_X.shape)
                print("Train y shape: ", train_y.shape)
                print("\nTrain X shuffle shape: ", train_X_shuffle.shape)
                print("Train y shuffle shape: ", train_y_shuffle.shape)
                if compute_validation:
                    print("Val X shuffle shape: ", val_X.shape)
                    print("Val y shuffle shape: ", val_y.shape)
                print("Test X shape: ", test_X.shape)
                print('\n******************************************************')

                if (save_predictions_on_validation_set and compute_validation):
                    if (not scale_output):
                        y_min = 0
                        y_max = 0
                    # We get 2095, which is the last one, and we use for the callback 
                    val_y_2022 = val_y[-1,:,:,:] 
                    val_y_2022 = val_y[-1,:,:,:]
                    # From (64,128,1) to (1,64,128,1)
                    val_y_2022 = val_y_2022[np.newaxis,:,:,:] 
                    val_X_2022 = val_X[-1,:]
                    val_X_2022 = val_X_2022[np.newaxis,:]
                    save_validation_predictions_callback = PerformancePlotCallback(val_X, val_y, val_years_list[-1], model, scenario_short, scenario, y_min, y_max, PATH_PREDICTIONS_YEAR_2021)
                else:
                    save_validation_predictions_callback = []

                callbacks = [save_validation_predictions_callback]
                
                start_train_time = time.time()
                if compute_validation:
                    # Continue fitting
                    history = trained_model.fit(train_X_shuffle,
                                                train_y_shuffle,
                                                epochs=epochs,
                                                batch_size=batch_size,
                                                validation_data=(val_X,val_y),
                                                use_multiprocessing=True,
                                                callbacks=callbacks)
                else:
                    # Continue fitting
                    history = trained_model.fit(train_X_shuffle,
                                                train_y_shuffle,
                                                epochs=epochs,
                                                batch_size=batch_size,
                                                use_multiprocessing=True,
                                                callbacks=callbacks)
                
                elapsed_train = (time.time() - start_train_time)
                elapsed_train_time = str(timedelta(seconds=elapsed_train))

                if compute_validation:
                    pd.DataFrame(np.array([history.history["loss"],
                                        history.history["val_loss"]]).T, columns=columns_history_df).to_csv(f'{PATH_HISTORIES}/{variable_short}_{model}_{scenario_short}_{ts_human}_history_{i}.csv')
                else:
                    pd.DataFrame(np.array([history.history["loss"]]).T, columns=columns_history_df).to_csv(f'{PATH_HISTORIES}/{variable_short}_{model}_{scenario_short}_{ts_human}_history_{i}.csv')

                if scale_input:
                    train_X = normalize_img(np.array(X_ssp_list[idx_short_scenario][:n_training_years]), feature_range[0], feature_range[1], X_min, X_max).reshape(-1,1)
                else:
                    train_X = np.array(X_ssp_list[idx_short_scenario][:n_training_years]).reshape(-1, 1)
                train_y_pred = trained_model.predict(train_X)

                test_y_pred = trained_model.predict(test_X)

                if scale_output:
                    train_y_pred_denorm = denormalize_img(train_y_pred,feature_range[0], feature_range[1], y_min, y_max)
                    train_y_denorm = denormalize_img(train_y,feature_range[0], feature_range[1], y_min, y_max)
                    test_y_pred_denorm = denormalize_img(test_y_pred,feature_range[0], feature_range[1], y_min, y_max)
                else:
                    train_y_pred_denorm = train_y_pred
                    train_y_denorm = train_y
                    test_y_pred_denorm = test_y_pred

                training_years = np.arange(start_year_training, end_year_training+1)
                for idx, year in enumerate(training_years):
                    # Save predictions
                    with open(f'{PATH_TRAINING_SET_PREDICTIONS}/{variable_short}_{model}_{scenario_short}_year-{int(year)}_epoch-last_{ts_human}_train_set_prediction_{i}.csv',"w+") as my_csv:
                        csvWriter = csv.writer(my_csv,delimiter=',')
                        csvWriter.writerows(train_y_pred_denorm[idx,:,:,0])
                print('\nSAVED PREDICTIONS ON TRAINING SET')
                    
                test_years = np.arange(start_year_test, end_year_test+1)
                for idx, year in enumerate(test_years):
                    # Save predictions
                    with open(f'{PATH_TEST_SET_PREDICTIONS}/{variable_short}_{model}_{scenario_short}_year-{int(year)}_epoch-last_{ts_human}_test_set_prediction_{i}.csv',"w+") as my_csv:
                        csvWriter = csv.writer(my_csv,delimiter=',')
                        csvWriter.writerows(test_y_pred_denorm[idx,:,:,0])
                print('\nSAVED PREDICTIONS ON TEST SET')

                if not save_predictions_on_validation_set and compute_validation:
                    PATH_TO_SAVE_PREDICTION = f'{PATH_PREDICTIONS_YEAR_2021}/{variable_short}_{model}_{scenario}_epoch-{epochs-1}_year-{val_years_list[-1]}_{ts_human}_val_set_prediction_{i}.png'
                    plot_prediction_mae_map(train_y_denorm[-1,:,:,0], train_y_pred_denorm[-1,:,:,0], model, scenario, epochs-1, val_years_list[-1], PATH_TO_SAVE_PREDICTION)

                PATH_HYPERPARAMETERS_CSV = f'{PATH_TRANSFER_LEARNING_ON_OBSERVATIONS}/Hyperparameters/{variable_short}_{model}_{scenario_short}_{ts_human}_hyperparameters_{i}.csv'

                transfer_learned_model_path_to_save = f'{PATH_MODELS}/{variable_short}_{model}_{scenario_short}_{ts_human}_model_{i}.tf'
                trained_model.save(transfer_learned_model_path_to_save)

                print('\nSAVED TRANSFER LEARNED MODEL')

                path_to_save_loss_curve = f'{PATH_PLOTS}/{variable_short}_{model}_{scenario_short}_{ts_human}_trainvalcurve_{i}'

                if compute_validation:
                    plot_train_val_loss_curve(history.history["loss"], history.history["val_loss"], loss, path_to_save_loss_curve)
                else:
                    plot_train_val_loss_curve(history.history["loss"], None, loss, path_to_save_loss_curve)
                print('\nSAVED TRAIN VAL LOSS CURVE')
                
                K.clear_session()

                elapsed_loop = (time.time() - start_loop_time)
                elapsed_loop_time = str(timedelta(seconds=elapsed_loop))

                if not os.path.exists(PATH_HYPERPARAMETERS_CSV): pd.DataFrame(columns=columns_model_hyperparameters_df).to_csv(PATH_HYPERPARAMETERS_CSV)
                df_hypp = pd.read_csv(PATH_HYPERPARAMETERS_CSV, dtype='str', usecols=columns_model_hyperparameters_df)
                df_hypp.loc[len(df_hypp.index)] = [f'Transfer_learning_{ts_human}', FIRST_TRAINING_DIRECTORY, end_year_training, model, scenario, ts_human, elapsed_loop_time, elapsed_train_time, epochs, batch_size, lr, shuffling_dataset[0], scale_input, scale_output, feature_range[0], feature_range[1], y_min, y_max, CO2eq_climate_model, withAerosolForcing, 'True']
                df_hypp.to_csv(PATH_HYPERPARAMETERS_CSV)