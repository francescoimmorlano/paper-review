"""
Author: Francesco Immorlano

Script for Transfer Learning on Observational Data
"""

from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers.legacy import Adam 
from keras.callbacks import Callback
import time
from datetime import timedelta
import csv
from lib import *

print(f'\n******************************************* Transfer_learning_obs_{ts_human} *******************************************')


shuffle = (False, 42)

compute_validation = True

# First training directory (only the directory name)
FIRST_TRAINING_DIRECTORY = ''

optim = Adam(learning_rate=lr_tl_obs, beta_1=0.9, beta_2=0.999, epsilon=1e-8)


class PerformancePlotCallback(Callback):
    def __init__(self, val_X, val_y, val_years, model_name, short_scenario, scenario, y_min, y_max, path_to_save):
        self.val_X = val_X
        self.val_y = val_y
        self.val_years = val_years
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
                val_y_pred_denorm = denormalize_img(val_y_pred[:,:,:,0], feature_range[0], feature_range[1], self.y_min, self.y_max)
                val_y_denorm = denormalize_img(self.val_y[:,:,:,0], feature_range[0], feature_range[1], self.y_min, self.y_max)
            else:
                val_y_pred_denorm = val_y_pred[:,:,:,0]
                val_y_denorm = val_y[:,:,:,0]

            for year_idx, year in enumerate(self.val_years):
                with open(f'{self.path_to_save}/{variable_short}_{model}_{self.short_scenario}_epoch-{epoch}_val_year-{year}_{ts_human}_val_set_prediction.csv',"w+") as my_csv:
                    csvWriter = csv.writer(my_csv,delimiter=',')
                    csvWriter.writerows(val_y_pred_denorm[year_idx,:,:])

            PATH_TO_SAVE_PLOT = f'{self.path_to_save}/{variable_short}_{model}_{self.short_scenario}_epoch-{epoch}_{ts_human}_val_set_prediction'
            plot_prediction_mae_map(val_y_denorm, val_y_pred_denorm, self.model_name, self.scenario, epoch, f'{PATH_TO_SAVE_PLOT}.png')

if compute_validation:
    columns_history_df = ['train_loss', 'val_loss']
else:
    columns_history_df = ['train_loss']

columns_model_hyperparameters_df = ['transf_learn_directory', 'first_train_directory', 'end_year_training', 'model', 'scenario', 'date_time', 'elapsed_loop_time', 'elapsed_train_time', 'epochs', 'val_years', 'batch_size', 'learning_rate', 'shuffle', 'scale_input', 'scale_output', 'norm_min', 'norm_max', 'y_min', 'y_max', 'CO2eq_climate_model', 'withAerosolForcing']

PATH_TRAINED_MODELS = f'{ROOT_EXPERIMENTS}/First_Training/{FIRST_TRAINING_DIRECTORY}/Models'
PATH_TRANSFER_LEARNING_ON_OBSERVATIONS = f'{ROOT_EXPERIMENTS}/Transfer_Learning_on_Observations/Transfer_learning_obs_{ts_human}'
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

_, X_ssp245, _ = read_CO2_equivalent('./', 'ssp245', CO2eq_climate_model, withAerosolForcing, start_year_training_tl_obs)
_, X_ssp370, _ = read_CO2_equivalent('./', 'ssp370', CO2eq_climate_model, withAerosolForcing, start_year_training_tl_obs)
_, X_ssp585, _ = read_CO2_equivalent('./', 'ssp585', CO2eq_climate_model, withAerosolForcing, start_year_training_tl_obs)

X_ssp_list = []
X_ssp_list.append(X_ssp245)
X_ssp_list.append(X_ssp370)
X_ssp_list.append(X_ssp585)

if scale_input:
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

                BEST_data_array = read_BEST_data(f'{ROOT_SOURCE_DATA}/BEST_data/gaussian_noise_{n_BEST_datasets_per_model_scenario}/BEST_regridded_annual_1979-2022_Gaussian_noise_{model}_{scenario_short}_{i}.nc')

                PATH_TEST_SET_PREDICTIONS = f'{PATH_PLOTS}/Test_set_predictions/{variable_short}_{model}_{scenario_short}_{i}'
                PATH_TRAINING_SET_PREDICTIONS = f'{PATH_PLOTS}/Training_set_predictions/{variable_short}_{model}_{scenario_short}_{i}'
                PATH_PREDICTIONS_VAL_YEARS = f'{PATH_TRANSFER_LEARNING_ON_OBSERVATIONS}/Predictions_on_val_years/{variable_short}_{model}_{scenario_short}_{i}'
                if not os.path.exists(PATH_TEST_SET_PREDICTIONS): os.makedirs(PATH_TEST_SET_PREDICTIONS)
                if not os.path.exists(PATH_TRAINING_SET_PREDICTIONS): os.makedirs(PATH_TRAINING_SET_PREDICTIONS)
                if not os.path.exists(PATH_PREDICTIONS_VAL_YEARS): os.makedirs(PATH_PREDICTIONS_VAL_YEARS)

                train_X = np.array(X_ssp_list[idx_short_scenario][:n_training_years_tl_obs])
                train_X = train_X.reshape(n_training_years_tl_obs,1,1)

                test_X = np.array(X_ssp_list[idx_short_scenario][n_training_years_tl_obs:n_training_years_tl_obs+n_test_years_tl_obs])
                test_X = test_X.reshape(n_test_years_tl_obs,1,1)

                train_y = np.zeros((n_training_years_tl_obs, 64, 128))
                train_y[:,:,:] = BEST_data_array[:n_training_years_tl_obs,:,:]

                trained_model = load_model(f'{PATH_TRAINED_MODELS}/{trained_model_filename}') # The model trained during the First Training must be loaded every time
                K.set_value(trained_model.optimizer.lr, lr_tl_obs)

                if compute_validation:
                    n_val_years = len(val_years_list_tl_obs)
                    val_X = np.zeros((n_val_years,1,1))
                    val_y = np.zeros((n_val_years, 64, 128))
                    
                    idx_to_remove = []
                    for idx_val_year, val_year in enumerate(val_years_list_tl_obs):
                        val_X[idx_val_year] = train_X[val_year-start_year_training_tl_obs]
                        val_y[idx_val_year] = train_y[val_year-start_year_training_tl_obs,:,:]
                        idx_to_remove.append(val_year-start_year_training_tl_obs)

                    train_X = np.delete(train_X, idx_to_remove, axis=0)
                    train_y = np.delete(train_y, idx_to_remove, axis=0)

                # The shuffle is done only on the train set. It is not needed on the test set
                if shuffle[0]:
                    idx_array = np.arange(0, n_training_years_tl_obs, 1, dtype=int)
                    np.random.seed(shuffle[1])
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
                    save_validation_predictions_callback = PerformancePlotCallback(val_X, val_y, val_years_list_tl_obs, model, scenario_short, scenario, y_min, y_max, PATH_PREDICTIONS_VAL_YEARS)
                else:
                    save_validation_predictions_callback = []

                callbacks = [save_validation_predictions_callback]
                
                start_train_time = time.time()
                if compute_validation:
                    # Fine-tuning
                    history = trained_model.fit(train_X_shuffle,
                                                train_y_shuffle,
                                                epochs=epochs,
                                                batch_size=batch_size_tl,
                                                validation_data=(val_X,val_y),
                                                use_multiprocessing=True,
                                                callbacks=callbacks)
                else:
                    # Fine-tuning
                    history = trained_model.fit(train_X_shuffle,
                                                train_y_shuffle,
                                                epochs=epochs,
                                                batch_size=batch_size_tl,
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
                    train_X = normalize_img(np.array(X_ssp_list[idx_short_scenario][:n_training_years_tl_obs]), feature_range[0], feature_range[1], X_min, X_max).reshape(-1,1)
                else:
                    train_X = np.array(X_ssp_list[idx_short_scenario][:n_training_years_tl_obs]).reshape(-1, 1)
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

                training_years = np.arange(start_year_training_tl_obs, end_year_training_tl_obs+1)
                for idx, year in enumerate(training_years):
                    # Save predictions
                    with open(f'{PATH_TRAINING_SET_PREDICTIONS}/{variable_short}_{model}_{scenario_short}_year-{int(year)}_epoch-last_{ts_human}_train_set_prediction_{i}.csv',"w+") as my_csv:
                        csvWriter = csv.writer(my_csv,delimiter=',')
                        csvWriter.writerows(train_y_pred_denorm[idx,:,:,0])
                print('\nSAVED PREDICTIONS ON TRAINING SET')
                    
                test_years = np.arange(start_year_test_tl_obs, end_year_test_tl_obs+1)
                for idx, year in enumerate(test_years):
                    # Save predictions
                    with open(f'{PATH_TEST_SET_PREDICTIONS}/{variable_short}_{model}_{scenario_short}_year-{int(year)}_epoch-last_{ts_human}_test_set_prediction_{i}.csv',"w+") as my_csv:
                        csvWriter = csv.writer(my_csv,delimiter=',')
                        csvWriter.writerows(test_y_pred_denorm[idx,:,:,0])
                print('\nSAVED PREDICTIONS ON TEST SET')

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
                df_hypp = pd.read_csv(PATH_HYPERPARAMETERS_CSV, dtype='str', usecols=columns_model_hyperparameters_df_tl)
                df_hypp.loc[len(df_hypp.index)] = [f'Transfer_learning_{ts_human}', FIRST_TRAINING_DIRECTORY, end_year_training_tl_obs, model, scenario, ts_human, elapsed_loop_time, elapsed_train_time, epochs, f'{start_year_first_training_val}-{end_year_first_training_val}', batch_size_tl, lr_tl_obs, shuffle[0], scale_input, scale_output, feature_range[0], feature_range[1], y_min, y_max, CO2eq_climate_model, withAerosolForcing]

                df_hypp.to_csv(PATH_HYPERPARAMETERS_CSV)