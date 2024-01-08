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
from tensorflow.keras import backend as K 
from tensorflow.keras.models import load_model 
from tensorflow.keras.callbacks import Callback 
from tensorflow.keras.optimizers.legacy import Adam 
from lib import *
from variables import *


"""
Script for implementing the Leave-one-out cross validation approach
"""

loss = 'mae'

##################### CALLBACKS #####################
save_predictions_on_validation_set = False

shuffling_dataset = (False, 42)
scale_input = True
scale_output = True

feature_range = (0,1)
n_channels = 1

# First training directory (only the directory name)
FIRST_TRAINING_DIRECTORY = ''

epochs = 500
batch_size = 16

lr = 1e-5
optim = Adam(learning_rate=lr, beta_1=0.9, beta_2=0.999, epsilon=1e-8)

end_year_training = 2022
start_year_test = end_year_training+1
end_year_test = 2098
n_ssp_training_years = end_year_training-2015+1
n_training_years = end_year_training-1850+1
n_test_years = end_year_test-start_year_test+1

# Years reserved for validation
val_years_list = [2025, 2035, 2045, 2055, 2065, 2075, 2085, 2095]

CO2eq_climate_model = 'MCE-v1-2'
withAerosolForcing = True

# Specify the idx of the model to take out (idx from 1 to 22)
if demo_download or demo_no_download:
    sensitivity_list = [1,2,3]
else:
    sensitivity_list = [9]

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

            if scale_output:
                val_y_pred_denorm = denormalize_img(val_y_pred[0,:,:,0], feature_range[0], feature_range[1], self.y_min, self.y_max)
                val_y_denorm = denormalize_img(self.val_y[0,:,:,0], feature_range[0], feature_range[1], self.y_min, self.y_max)
            else:
                val_y_pred_denorm = val_y_pred[0,:,:,0]
                val_y_denorm = val_y[0,:,:,0]

            PATH_TO_SAVE_PREDICTION = f'{self.path_to_save}/{variable_short}_{model}_{self.short_scenario}_shuffle-{shuffle_taken_out_model_number}_epoch-{epoch}_year-{self.val_year}_{ts_human}_val_set_prediction'

            with open(f'{PATH_TO_SAVE_PREDICTION}.csv',"w+") as my_csv:
                csvWriter = csv.writer(my_csv,delimiter=',')
                csvWriter.writerows(val_y_pred_denorm[:,:])

            plot_prediction_mae_map(val_y_denorm, val_y_pred_denorm, self.model_name, self.scenario, epoch, self.val_year, f'{PATH_TO_SAVE_PREDICTION}.png')
            
columns_history_df = ['train_loss', 'vals_loss']
columns_model_hyperparameters_df = ['transf_learn_directory', 'first_train_directory', 'end_year_training', 'model', 'scenario', 'date_time', 'elapsed_loop_time', 'elapsed_train_time', 'epochs', 'batch_size', 'learning_rate', 'shuffle', 'scale_input', 'scale_output', 'norm_min', 'norm_max', 'y_min', 'y_max', 'CO2eq_climate_model', 'withAerosolForcing']

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
PATH_ANNUAL_SIMULATIONS_DIRECTORY = f'{ROOT_DATA}/CMIP6_data/{variable}/Annual_uniform_remapped'


trained_models_list = os.listdir(PATH_TRAINED_MODELS)
trained_models_list.sort()

annual_simulations_list = os.listdir(PATH_ANNUAL_SIMULATIONS_DIRECTORY)
annual_simulations_list.sort()

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

ts = datetime.now()
ts_human = ts.strftime('%Y-%m-%d_%H-%M-%S')
print(f'\n******************************************* Transfer_learning_{ts_human} *******************************************')

for model_taken_out_idx, model_taken_out  in enumerate(models_list):

    if (model_taken_out_idx+1 not in sensitivity_list):
        continue

    if (model_taken_out_idx+1 < 10):
        shuffle_taken_out_model_number = f'0{model_taken_out_idx+1}'
    else:
        shuffle_taken_out_model_number = f'{model_taken_out_idx+1}'

    PATH_SHUFFLE = f'{ROOT_EXPERIMENTS}/Experiments/Transfer_Learning_on_Simulations/Transfer_learning_{ts_human}/Shuffle_{shuffle_taken_out_model_number}'
    PATH_HISTORIES = f'{PATH_SHUFFLE}/Histories'
    PATH_HYPERPARAMETERS = f'{PATH_SHUFFLE}/Hyperparameters'
    PATH_MODELS = f'{PATH_SHUFFLE}/Models'
    PATH_PLOTS = f'{PATH_SHUFFLE}/Plots'
    if not os.path.exists(PATH_SHUFFLE): os.makedirs(PATH_SHUFFLE)
    if not os.path.exists(PATH_HYPERPARAMETERS): os.mkdir(PATH_HYPERPARAMETERS)
    if not os.path.exists(PATH_MODELS): os.mkdir(PATH_MODELS)
    if not os.path.exists(PATH_HISTORIES): os.mkdir(PATH_HISTORIES)
    if not os.path.exists(PATH_PLOTS): os.mkdir(PATH_PLOTS)

    HISTORICAL_SIMULATION_TAKEOUT_FILENAME = [s for s in annual_simulations_list if (model_taken_out in s and 'historical' in s)][0]
    nc_historical_takeout_model_data = Dataset(f'{PATH_ANNUAL_SIMULATIONS_DIRECTORY}/{HISTORICAL_SIMULATION_TAKEOUT_FILENAME}', mode='r+', format='NETCDF3_CLASSIC')

    n_historical_takeout_model_years = nc_historical_takeout_model_data[variable_short].shape[0]

    n_lats = nc_historical_takeout_model_data['lat'].shape[0]
    n_lons = nc_historical_takeout_model_data['lon'].shape[0]

    historical_simulation_takeout_model_array = np.zeros((n_historical_takeout_model_years, n_lats, n_lons))
    historical_simulation_takeout_model_array[:,:,:] = nc_historical_takeout_model_data[variable_short][:,:,:]

    nc_historical_takeout_model_data.close()

    historical_takeout_model_years = np.arange(1850, 2015, 1, dtype=int)
    historical_takeout_model_years = historical_takeout_model_years.reshape(n_historical_takeout_model_years,1,1)

    for idx_model, model in enumerate(models_list):
        if(model == model_taken_out):
            continue
        elif exclude_family_members:
            if (shuffle_taken_out_model_number == '01' and (model=='KACE-1-0-G' or model=='UKESM1-0-LL')): continue
            if (shuffle_taken_out_model_number == '02' and (model=='CAMS-CSM1-0' or model=='MPI-ESM1-2-LR')): continue
            if (shuffle_taken_out_model_number == '04' and (model=='AWI-CM-1-1-MR' or model=='MPI-ESM1-2-LR')): continue
            if (shuffle_taken_out_model_number == '06' and (model=='NorESM2-MM' or model=='TaiESM1')): continue
            if (shuffle_taken_out_model_number == '07' and  (model=='CNRM-ESM2-1' or model=='IPSL-CM6A-LR')): continue
            if (shuffle_taken_out_model_number == '08' and  (model=='CNRM-CM6-1' or model=='IPSL-CM6A-LR')): continue
            if (shuffle_taken_out_model_number == '13' and  model=='INM-CM5-0'): continue
            if (shuffle_taken_out_model_number == '14' and  model=='INM-CM4-8'): continue
            if (shuffle_taken_out_model_number == '15' and  (model=='CNRM-CM6-1' or model=='CNRM-ESM2-1')): continue
            if (shuffle_taken_out_model_number == '16' and  (model=='ACCESS-CM2' or model=='UKESM1-0-LL')): continue
            if (shuffle_taken_out_model_number == '18' and  (model=='AWI-CM-1-1-MR' or model=='CAMS-CSM1-0')): continue
            if (shuffle_taken_out_model_number == '20' and  (model=='CMCC-CM2-SR5' or model=='TaiESM1')): continue
            if (shuffle_taken_out_model_number == '21' and  (model=='CMCC-CM2-SR5' or model=='NorESM2-MM')): continue
            if (shuffle_taken_out_model_number == '22' and  (model=='ACCESS-CM2' or model=='KACE-1-0-G')): continue

        for idx_short_scenario, short_scenario in enumerate(short_scenarios_list):
            start_loop_time = time.time()

            scenario = f'SSP{short_scenario[-3]}-{short_scenario[-2]}.{short_scenario[-1]}'

            trained_model_name = [m for m in trained_models_list if (model in m and short_scenario in m)][0]
            print('\n************************************************************************')
            print(f'\nTrained Model: {trained_model_name}')
            print(f'\nModel to transfer learn: {model} - Model takeout: {model_taken_out} - Shuffle: {shuffle_taken_out_model_number} - Scenario: {scenario}\n')

            PATH_TEST_SET_PREDICTIONS = f'{PATH_PLOTS}/Test_set_predictions/{variable_short}_{model}_{short_scenario}_shuffle-{shuffle_taken_out_model_number}'
            PATH_TRAINING_SET_PREDICTIONS = f'{PATH_PLOTS}/Training_set_predictions/{variable_short}_{model}_{short_scenario}_shuffle-{shuffle_taken_out_model_number}'
            PATH_PREDICTIONS_YEAR_2095 = f'{PATH_SHUFFLE}/Predictions_on_year_2095/{variable_short}_{model}_{short_scenario}_shuffle-{shuffle_taken_out_model_number}'
            if not os.path.exists(PATH_TEST_SET_PREDICTIONS): os.makedirs(PATH_TEST_SET_PREDICTIONS)
            if not os.path.exists(PATH_TRAINING_SET_PREDICTIONS): os.makedirs(PATH_TRAINING_SET_PREDICTIONS)
            if not os.path.exists(PATH_PREDICTIONS_YEAR_2095): os.makedirs(PATH_PREDICTIONS_YEAR_2095)
            
            SSP_SIMULATION_TAKEOUT_FILENAME = [s for s in annual_simulations_list if (model_taken_out in s and short_scenario in s)][0]

            print(f'Historical simulation takeout filename: {HISTORICAL_SIMULATION_TAKEOUT_FILENAME}')
            print(f'SSP simulation takeout filename: {SSP_SIMULATION_TAKEOUT_FILENAME}')

            nc_ssp_takeout_model_data = Dataset(f'{PATH_ANNUAL_SIMULATIONS_DIRECTORY}/{SSP_SIMULATION_TAKEOUT_FILENAME}', mode='r+', format='NETCDF3_CLASSIC')
            
            n_ssp_takeout_model_years = nc_ssp_takeout_model_data[variable_short].shape[0]
            
            ssp_simulation_takeout_model_array = np.zeros((end_year_test-2015+1, n_lats, n_lons))
            if n_ssp_takeout_model_years == 86: 
                ssp_simulation_takeout_model_array[:,:,:] = nc_ssp_takeout_model_data[variable_short][:-2,:,:]
            elif n_ssp_takeout_model_years == 85:
                ssp_simulation_takeout_model_array[:,:,:] = nc_ssp_takeout_model_data[variable_short][:-1,:,:]
            elif n_ssp_takeout_model_years == 84:
                ssp_simulation_takeout_model_array[:,:,:] = nc_ssp_takeout_model_data[variable_short][:,:,:]
                
            nc_ssp_takeout_model_data.close()

            ssp_takeout_model_years = np.arange(2015, 2015+n_ssp_takeout_model_years, 1, dtype=int)
            ssp_takeout_model_years = ssp_takeout_model_years.reshape(n_ssp_takeout_model_years,1,1)

            train_X = np.array(X_ssp_list[idx_short_scenario][:n_training_years])
            train_X = train_X.reshape(n_training_years,1,1)

            test_X = np.array(X_ssp_list[idx_short_scenario][n_training_years:n_training_years+n_test_years])
            test_X = test_X.reshape(n_test_years,1,1)

            train_y = np.zeros((n_training_years, n_lats, n_lons))
            # If the training set goes up to a year < 2015
            if (n_training_years < n_historical_takeout_model_years): 
                train_y[:,:,:] = historical_simulation_takeout_model_array[:n_training_years,:,:]
            # If the training set goes up to a year >= 2015
            else: 
                train_y[:n_historical_takeout_model_years,:,:] = historical_simulation_takeout_model_array[:,:,:]
                train_y[n_historical_takeout_model_years:,:,:] = ssp_simulation_takeout_model_array[:n_ssp_training_years,:,:]

            test_y = np.zeros((n_test_years, n_lats, n_lons))
                
            # If the training set goes up to a year < 2015
            if (n_training_years < n_historical_takeout_model_years): 
                test_y[:n_historical_takeout_model_years-n_training_years,:,:] = historical_simulation_takeout_model_array[n_training_years:,:,:]
                test_y[n_historical_takeout_model_years-n_training_years:,:,:] = ssp_simulation_takeout_model_array[:,:,:]
            # If the training set goes up to a year >= 2015
            else: 
                test_y[:,:,:] = ssp_simulation_takeout_model_array[n_ssp_training_years:,:,:]

            if loss == 'mae' or loss == 'mse':
                trained_model = load_model(f'{PATH_TRAINED_MODELS}/{trained_model_name}')

            K.set_value(trained_model.optimizer.lr, lr)
            new_model = trained_model

            # shuffle onyl the training set. It is not needed on test set
            if shuffling_dataset[0]: 
                idx_array = np.arange(0, n_training_years, 1, dtype=int)
                np.random.seed(shuffling_dataset[1])
                np.random.shuffle(idx_array)
                train_X_shuffle, train_y_shuffle = train_X[idx_array[:]], train_y[idx_array[:],:,:]
            else:
                train_X_shuffle = train_X
                train_y_shuffle = train_y
            
            n_val_years = len(val_years_list)
            val_X = np.zeros((n_val_years,1,1))
            val_y = np.zeros((n_val_years, n_lats, n_lons))

            for idx_val_year, val_year in enumerate(val_years_list):
                # Get from the test set those CO2 values to put into the val set
                val_X[idx_val_year] = test_X[val_year-start_year_test]
                val_y[idx_val_year] = test_y[val_year-start_year_test,:,:]
            
            if scale_input:
                X_min = X_min_list[idx_short_scenario]
                X_max = X_max_list[idx_short_scenario]
                train_X = normalize_img(train_X, feature_range[0], feature_range[1], X_min, X_max).reshape(-1,1)
                train_X_shuffle = normalize_img(train_X_shuffle, feature_range[0], feature_range[1], X_min, X_max).reshape(-1,1)
                val_X = normalize_img(val_X, feature_range[0], feature_range[1], X_min, X_max).reshape(-1,1)
                test_X = normalize_img(test_X, feature_range[0], feature_range[1], X_min, X_max).reshape(-1,1)
            else:
                train_X = train_X.reshape(-1, 1)
                train_X_shuffle = train_X_shuffle.reshape(-1, 1)
                val_X = val_X.reshape(-1, 1)
                test_X = test_X.reshape(-1, 1)

            if scale_output:
                train_y = normalize_img(train_y, feature_range[0], feature_range[1], y_min, y_max).reshape(-1, historical_simulation_takeout_model_array.shape[1], historical_simulation_takeout_model_array.shape[2], n_channels)
                train_y_shuffle = normalize_img(train_y_shuffle, feature_range[0], feature_range[1], y_min, y_max).reshape(-1, historical_simulation_takeout_model_array.shape[1], historical_simulation_takeout_model_array.shape[2], n_channels)   
                val_y = normalize_img(val_y, feature_range[0], feature_range[1], y_min, y_max).reshape(-1, ssp_simulation_takeout_model_array.shape[1], ssp_simulation_takeout_model_array.shape[2], n_channels)
                test_y = normalize_img(test_y, feature_range[0], feature_range[1], y_min, y_max).reshape(-1, ssp_simulation_takeout_model_array.shape[1], ssp_simulation_takeout_model_array.shape[2], n_channels)
            else:
                train_y = train_y.reshape(-1, historical_simulation_takeout_model_array.shape[1], historical_simulation_takeout_model_array.shape[2], n_channels)
                train_y_shuffle = train_y_shuffle.reshape(-1, historical_simulation_takeout_model_array.shape[1], historical_simulation_takeout_model_array.shape[2], n_channels)
                val_y = val_y.reshape(-1, ssp_simulation_takeout_model_array.shape[1], ssp_simulation_takeout_model_array.shape[2], n_channels)
                test_y = test_y.reshape(-1, ssp_simulation_takeout_model_array.shape[1], ssp_simulation_takeout_model_array.shape[2], n_channels)

            print("\nTrain X shape: ", train_X.shape)
            print("Train y shape: ", train_y.shape)
            print("\nTrain X shuffle shape: ", train_X_shuffle.shape)
            print("Train y shuffle shape: ", train_y_shuffle.shape)
            print("Val X shuffle shape: ", val_X.shape)
            print("Val y shuffle shape: ", val_y.shape)
            print("Test X shape: ", test_X.shape)
            print("Test y shape: ", test_y.shape)
            print('\n**********************************************************')

            if (save_predictions_on_validation_set):
                if (not scale_output):
                    y_min = 0
                    y_max = 0
                val_y_2095 = val_y[-1,:,:,:]
                val_y_2095 = val_y_2095[np.newaxis,:,:,:]
                val_X_2095 = val_X[-1,:]
                val_X_2095 = val_X_2095[np.newaxis,:]
                save_validation_predictions_callback = PerformancePlotCallback(val_X_2095, val_y_2095, val_years_list[-1], model, short_scenario, scenario, y_min, y_max, PATH_PREDICTIONS_YEAR_2095)
            else:
                save_validation_predictions_callback = []

            callbacks = [save_validation_predictions_callback]
            
            start_train_time = time.time()

            # Continue fitting
            history = new_model.fit(train_X_shuffle,
                                    train_y_shuffle,
                                    epochs=epochs,
                                    batch_size=batch_size,
                                    validation_data=(val_X,val_y),
                                    use_multiprocessing=True,
                                    callbacks=callbacks)
            
            elapsed_train = (time.time() - start_train_time)
            elapsed_train_time = str(timedelta(seconds=elapsed_train))

            pd.DataFrame(np.array([history.history["loss"],
                                history.history["val_loss"]]).T, columns=columns_history_df).to_csv(f'{PATH_HISTORIES}/{variable_short}_{model}_{short_scenario}_shuffle-{shuffle_taken_out_model_number}_{ts_human}_history.csv')

            

            test_y_pred = new_model.predict(test_X)
            train_y_pred = new_model.predict(train_X)

            if scale_output:
                train_y_pred_denorm = denormalize_img(train_y_pred,feature_range[0], feature_range[1], y_min, y_max)
                train_y_denorm = denormalize_img(train_y,feature_range[0], feature_range[1], y_min, y_max)
                test_y_pred_denorm = denormalize_img(test_y_pred,feature_range[0], feature_range[1], y_min, y_max)
                test_y_denorm = denormalize_img(test_y,feature_range[0], feature_range[1], y_min, y_max)
                val_y_denorm = denormalize_img(val_y, feature_range[0], feature_range[1], y_min, y_max)
            else:
                train_y_pred_denorm = train_y_pred
                train_y_denorm = train_y
                test_y_pred_denorm = test_y_pred
                test_y_denorm = test_y
                val_y_denorm = val_y

            training_years = np.arange(1850, end_year_training+1)
            for idx, year in enumerate(training_years):
                # Save predictions
                with open(f'{PATH_TRAINING_SET_PREDICTIONS}/{variable_short}_{model}_{short_scenario}_shuffle-{shuffle_taken_out_model_number}_year-{int(year)}_epoch-last_{ts_human}_train_set_prediction.csv',"w+") as my_csv:
                    csvWriter = csv.writer(my_csv,delimiter=',')
                    csvWriter.writerows(train_y_pred_denorm[idx,:,:,0])

            print('\nSAVED PREDICTIONS ON TRAINING SET')
            
            test_years = np.arange(start_year_test, end_year_test+1)
            for idx, year in enumerate(test_years):
                # Save predictions
                with open(f'{PATH_TEST_SET_PREDICTIONS}/{variable_short}_{model}_{short_scenario}_shuffle-{shuffle_taken_out_model_number}_year-{int(year)}_epoch-last_{ts_human}_test_set_prediction.csv',"w+") as my_csv:
                    csvWriter = csv.writer(my_csv,delimiter=',')
                    csvWriter.writerows(test_y_pred_denorm[idx,:,:,0])

            print('\nSAVED PREDICTIONS ON TEST SET')

            if not save_predictions_on_validation_set:
                PATH_TO_SAVE_PREDICTION = f'{PATH_PREDICTIONS_YEAR_2095}/{variable_short}_{model}_{short_scenario}_shuffle-{shuffle_taken_out_model_number}_epoch-{epochs-1}_year-{2095}_{ts_human}_val_set_prediction.png'
                plot_prediction_mae_map(test_y_denorm[2095-start_year_test,:,:,0], test_y_pred_denorm[2095-start_year_test,:,:,0], model, scenario, epochs-1, 2095, PATH_TO_SAVE_PREDICTION)

            PATH_HYPERPARAMETERS_CSV = f'{PATH_SHUFFLE}/Hyperparameters/{variable_short}_{model}_{short_scenario}_shuffle-{shuffle_taken_out_model_number}_{ts_human}_hyperparameters.csv'

            if not os.path.exists(PATH_HYPERPARAMETERS_CSV):
                pd.DataFrame(columns=columns_model_hyperparameters_df).to_csv(PATH_HYPERPARAMETERS_CSV)
            df_hypp = pd.read_csv(PATH_HYPERPARAMETERS_CSV, dtype='str', usecols=columns_model_hyperparameters_df)
            
            

            model_path_to_save = f'{PATH_MODELS}/{variable_short}_{model}_{short_scenario}_shuffle{shuffle_taken_out_model_number}_{ts_human}_model.tf'
            new_model.save(model_path_to_save)

            print('\nSAVED MODEL')

            path_to_save_loss_curve = f'{PATH_PLOTS}/{variable_short}_{model}_{short_scenario}_shuffle-{shuffle_taken_out_model_number}_{ts_human}_trainvalcurve'
            plot_train_val_loss_curve(history.history["loss"], history.history["val_loss"], loss, path_to_save_loss_curve)

            print('\nSAVED TRAIN VAL LOSS CURVE')

            K.clear_session()

            elapsed_loop = (time.time() - start_loop_time)
            elapsed_loop_time = str(timedelta(seconds=elapsed_loop))

            df_hypp.loc[len(df_hypp.index)] = [f'Transfer_learning_{ts_human}', FIRST_TRAINING_DIRECTORY, end_year_training, model, scenario, ts_human, elapsed_loop_time, elapsed_train_time, epochs, batch_size, lr, shuffling_dataset[0], scale_input, scale_output, feature_range[0], feature_range[1], y_min, y_max, CO2eq_climate_model, withAerosolForcing]
            df_hypp.to_csv(PATH_HYPERPARAMETERS_CSV)
