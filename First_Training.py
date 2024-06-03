"""
Author: Francesco Immorlano

Script for training one Deep Neural Network (DNN) on the simulation of one of the 22 ESMs under one of the 3 SSPs.
Overall, 66 DNNs are trained, each on a different simulation and with the same hyperparameters
and architecture.
"""

import time
from datetime import timedelta
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.optimizers.legacy import Adam
from tensorflow.keras.regularizers import l1_l2
import csv
from lib import *
from architectures import *

print(f'\n\n************************************ First_Training_{ts_human} ************************************')

PATH_ANNUAL_SIMULATIONS_DIRECTORY = f'{ROOT_SOURCE_DATA}/CMIP6_data/{variable}/Annual_uniform_remapped'
PATH_FIRST_TRAINING = f'{ROOT_EXPERIMENTS}/Experiments/First_Training/First_Training_{ts_human}'

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

shuffle = (True, 42)

# This is set to (1,0) to leave all the samples in the training set and then extract the ones reserved for the validation. Then shuffle is applied
train_val_ratio = (1, 0)

l1_regularization = 0
l2_regularization = 0
regularizer = l1_l2(l1=l1_regularization, l2=l2_regularization)

weight_initializer = 'glorot_uniform'

optim = Adam(learning_rate=lr_first_train, beta_1=0.9, beta_2=0.999, epsilon=1e-8)


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
X_ssp_list.append(X_ssp245[:-2])
X_ssp_list.append(X_ssp370[:-2])
X_ssp_list.append(X_ssp585[:-2])

if scale_input:
    return_list = compute_values_for_scaling(X_ssp_list)
    X_min_list = return_list[0]
    X_max_list = return_list[1]

# Read CMIP6 simulations
cmip6_simulations = read_all_cmip6_simulations()

for idx_model, model in enumerate(models_list):
    for idx_short_scenario, short_scenario in enumerate(short_scenarios_list):
        start_loop_time = time.time()

        scenario = f'SSP{short_scenario[-3]}-{short_scenario[-2]}.{short_scenario[-1]}'
        print(f'{idx_model+1}. Model: {model} - Scenario: {scenario}\n\n')

        PATH_HYPERPARAMETERS_CSV = f'{PATH_HYPERPARAMETERS}/{variable_short}_{model}_{short_scenario}_{ts_human}_models_hyperparameters.csv'
        if not os.path.exists(PATH_HYPERPARAMETERS_CSV): pd.DataFrame(columns=columns_model_hyperparameters_df_first_training).to_csv(PATH_HYPERPARAMETERS_CSV)
        
        df_hypp = pd.read_csv(PATH_HYPERPARAMETERS_CSV, dtype='str', usecols=columns_model_hyperparameters_df_first_training)

        
        X_co2 = X_ssp_list[idx_short_scenario].copy()

        years = np.arange(start_year_first_training, end_year_first_training+1, 1, dtype=int)
        years = years.reshape(n_training_years_first_training,1,1)

        print(f'\n val x val y shape before val reservation:')
        print(len(X_co2))
        
        val_year_idx = val_year_first_training - start_year_first_training
        val_X = X_co2[val_year_idx]
        val_X = np.array(val_X)
        X_co2.pop(val_year_idx)
        X_co2 = np.array(X_co2)

        simulation_array = cmip6_simulations[idx_model, idx_short_scenario, :, :, :]
        
        val_y = simulation_array[val_year_idx,:,:]
        simulation_array = np.delete(simulation_array, val_year_idx, axis=0)

        print(f'\n val x val y shape before val reservation:')
        print(X_co2.shape)
        print(val_y.shape)
        
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
            save_validation_predictions_callback = PerformancePlotCallback(val_X, val_y, val_year_first_training, model, short_scenario, scenario, y_min, y_max)
        else:
            save_validation_predictions_callback = []

        callbacks = [save_validation_predictions_callback]
    
        start_train_time = time.time()

        history = NN_model.fit(train_X,
                                train_y,
                                epochs=epochs,
                                batch_size=batch_size_first_train,
                                validation_data=(val_X,val_y),
                                use_multiprocessing=True,
                                callbacks=callbacks)
        
        elapsed_train = (time.time() - start_train_time)
        elapsed_train_time = str(timedelta(seconds=elapsed_train))

        NN_model_name = f'{PATH_MODELS}/{variable_short}_{model}_{short_scenario}_{ts_human}_model.tf'
        NN_model.save(NN_model_name)

        pd.DataFrame(np.array([history.history["loss"],
                            history.history["val_loss"]]).T, columns=columns_history_df).to_csv(f'{PATH_HISTORIES}/{variable_short}_{model}_{short_scenario}_{ts_human}_history.csv')

        path_to_save_plot = f'{PATH_PLOTS}/{variable_short}_{model}_{short_scenario}_{ts_human}_trainvalcurve.png'
        plot_train_val_loss_curve(history.history["loss"], history.history["val_loss"], loss, path_to_save_plot)

        val_y_pred = NN_model.predict(val_X)

        if scale_output:
            val_y_pred_denorm = denormalize_img(val_y_pred[0,:,:,0], feature_range[0], feature_range[1], y_min, y_max)
            val_y_denorm = denormalize_img(val_y[0,:,:,0], feature_range[0], feature_range[1], y_min, y_max)
        else:
            val_y_pred_denorm = val_y_pred[0,:,:,0]
            val_y_denorm = val_y[0,:,:,0]

        plot_prediction_mae_map(val_y_denorm, val_y_pred_denorm, model, scenario, epochs, val_year_first_training, f'{PATH_PLOTS}/{variable_short}_{model}_{short_scenario}_{ts_human}_pred-on-val_end-epoch.png')

        print('\n\n************************************ PREDICTION ON VALIDATION SET DONE ************************************\n')

        elapsed_loop = (time.time() - start_loop_time)
        elapsed_loop_time = str(timedelta(seconds=elapsed_loop))

        df_hypp.loc[len(df_hypp.index)] = [f'First_Training_{ts_human}', model, scenario, ts_human, elapsed_loop_time, elapsed_train_time, epochs, batch_size_first_train, start_year_first_training, '2098', val_year_first_training, l1_regularization, l2_regularization, shuffle[0], scale_input, scale_output, feature_range[0], feature_range[1], train_val_ratio[0], train_val_ratio[1], optim.get_config(), loss, weight_initializer, activation_functions, CO2eq_climate_model, withAerosolForcing, 'False']
        df_hypp.to_csv(PATH_HYPERPARAMETERS_CSV)