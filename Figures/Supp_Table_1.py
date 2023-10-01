import os
from netCDF4 import Dataset
import numpy as np

"""
Script for computing Supplementary Table 1 results
"""

models_list = [
        'ACCESS-CM2',
        'AWI-CM-1-1-MR',
        'BCC-CSM2-MR',
        'CAMS-CSM1-0',
        'CanESM5-CanOE',
        'CMCC-CM2-SR5',
        'CNRM-CM6-1',
        'CNRM-ESM2-1',
        'FGOALS-f3-L',
        'FGOALS-g3',
        'GFDL-ESM4',
        'IITM-ESM',
        'INM-CM4-8',
        'INM-CM5-0',
        'IPSL-CM6A-LR',
        'KACE-1-0-G',
        'MIROC6',
        'MPI-ESM1-2-LR',
        'MRI-ESM2-0',
        'NorESM2-MM',
        'TaiESM1',
        'UKESM1-0-LL'
        ]

short_scenarios_list = ['ssp245', 'ssp370', 'ssp585']
variable_short = 'tas'

ROOT_DATA = '../Source_data'
SIMULATIONS_DIRECTORY = f'{ROOT_DATA}/CMIP6_data/near_surface_air_temperature/Annual_uniform_remapped'

total_earth_area = 5.1009974e+14
# Avg global surface temperature in 1850-1900
global_mean_temp_1850_1900 = 13.798588235294114
with open('../area_cella.csv', newline='') as csvfile:
    area_cella = np.genfromtxt(csvfile, delimiter=',')

n_BEST_datasets_per_model_scenario = 5

for i in range(1,23):
    if i < 10:
        shuffle_idx = f'0{i}'
    else:
        shuffle_idx = f'{i}'

    models_list_take_out = models_list.copy()
    model_take_out = models_list[int(shuffle_idx)-1] # low-climate sensitivity

    print(f'\nLoading Model taken out: {model_take_out} - Shuffle: {shuffle_idx} ...')

    # Delete the current take out model (i.e., the model which other models are transfer learned on) from the list 
    models_list_take_out.remove(model_take_out)

    start_year_training = 1850
    end_year_training = 2022
    n_training_years = end_year_training-start_year_training+1

    start_year_test = end_year_training+1
    n_test_years = 2098-start_year_test+1

    """ Load predictions made by the DNNs after transfer learning on the take-out simulation """
    predictions = np.zeros((len(models_list_take_out), len(short_scenarios_list), n_training_years+n_test_years, 64, 128))
    for model_idx, model in enumerate(models_list_take_out):
        if model_take_out == model:
            continue
        for scenario_idx, scenario_short in enumerate(short_scenarios_list):
            TRAIN_SET_PREDICTIONS_DIRECTORY = f'{ROOT_DATA}/Transfer_Learning_on_Simulations/Shuffle_{shuffle_idx}/Training_set_predictions/tas_{model}_{scenario_short}_shuffle-{shuffle_idx}'
            TEST_SET_PREDICTIONS_DIRECTORY = f'{ROOT_DATA}/Transfer_Learning_on_Simulations/Shuffle_{shuffle_idx}/Test_set_predictions/tas_{model}_{scenario_short}_shuffle-{shuffle_idx}'
            # Training set predictions
            model_train_set_predictions_filenames_list = os.listdir(TRAIN_SET_PREDICTIONS_DIRECTORY)
            model_train_set_predictions_filenames_list = [fn for fn in model_train_set_predictions_filenames_list if (fn.endswith('.csv'))]
            model_train_set_predictions_filenames_list.sort()
            model_train_set_prediction_array = np.zeros((n_training_years, 64, 128))
            for mp_idx, mp_filename in enumerate(model_train_set_predictions_filenames_list):
                if (not mp_filename.endswith('.csv')):
                    continue
                file = open(f'{TRAIN_SET_PREDICTIONS_DIRECTORY}/{mp_filename}')
                model_train_set_prediction_array[mp_idx,:,:] = np.loadtxt(file, delimiter=',')
            predictions[model_idx,scenario_idx,:n_training_years,:,:] = model_train_set_prediction_array
            # Test set predictions
            model_test_set_predictions_filenames_list = os.listdir(TEST_SET_PREDICTIONS_DIRECTORY)
            model_test_set_predictions_filenames_list = [fn for fn in model_test_set_predictions_filenames_list if (fn.endswith('.csv'))]
            model_test_set_predictions_filenames_list.sort()
            model_test_set_prediction_array = np.zeros((n_test_years, 64, 128))
            for mp_idx, mp_filename in enumerate(model_test_set_predictions_filenames_list):
                if (not mp_filename.endswith('.csv')):
                    continue
                file = open(f'{TEST_SET_PREDICTIONS_DIRECTORY}/{mp_filename}')
                model_test_set_prediction_array[mp_idx,:,:] = np.loadtxt(file, delimiter=',')
            predictions[model_idx,scenario_idx,n_training_years:,:,:] = model_test_set_prediction_array[:,:,:]

    """ Load CMIP6 take-out simulation """
    simulations = np.zeros((len(short_scenarios_list), n_training_years+n_test_years, 64, 128))
    for scenario_idx, scenario_short in enumerate(short_scenarios_list):
        simulations_files_list = os.listdir(SIMULATIONS_DIRECTORY)
        simulations_files_list.sort()
        matching_simulations = [simulation_file for simulation_file in simulations_files_list if ((model_take_out in simulation_file and 'historical' in simulation_file)
                                                                                                or (model_take_out in simulation_file and scenario_short in simulation_file))]
        # maching_simulations[0] is the historical and matching_simulations[1] is the SSP simulation because of the sort operation
        # (for each model, the first simulation is the historical and then the SSP)  
        nc_historical_data = Dataset(f'{SIMULATIONS_DIRECTORY}/{matching_simulations[0]}', mode='r+', format='NETCDF3_CLASSIC')
        nc_ssp_data = Dataset(f'{SIMULATIONS_DIRECTORY}/{matching_simulations[1]}', mode='r+', format='NETCDF3_CLASSIC')
        n_historical_years = nc_historical_data[variable_short].shape[0]
        n_ssp_years = nc_ssp_data[variable_short].shape[0]
        n_lats = nc_ssp_data['lat'].shape[0]
        n_lons = nc_ssp_data['lon'].shape[0]
        simulations[scenario_idx,:n_historical_years,:,:] = nc_historical_data[variable_short][:]
        if (n_ssp_years == 84):
            simulations[scenario_idx,n_historical_years:,:,:] = nc_ssp_data[variable_short][:,:,:]
        elif (n_ssp_years == 85):
            simulations[scenario_idx,n_historical_years:,:,:] = nc_ssp_data[variable_short][:-1,:,:]
        elif (n_ssp_years == 86):
            simulations[scenario_idx,n_historical_years:,:,:] = nc_ssp_data[variable_short][:-2,:,:]
        nc_historical_data.close()
        nc_ssp_data.close()

    # Convert from K to Celsius degrees
    predictions_C = predictions - 273.15
    simulations_C = simulations - 273.15

    # Compute average global surface air temperature
    annual_predictions_means = ((predictions_C * area_cella).sum(axis=(-1,-2)))/total_earth_area
    annual_simulations_means = ((simulations_C * area_cella).sum(axis=(-1,-2)))/total_earth_area

    # Compute warming wrt pre-industrial period
    annual_predictions_means -= global_mean_temp_1850_1900
    annual_simulations_means -= global_mean_temp_1850_1900

    # Compute average across DNNs predictions
    ensemble_predictions_means = np.mean(annual_predictions_means, axis=0)

    """ Compute bias """
    ensemble_model_taken_out_difference = ensemble_predictions_means[:,2081-1850:2098-1850+1] - annual_simulations_means[:,2081-1850:2098-1850+1]
    avg_bias_2081_2098 = np.mean(ensemble_model_taken_out_difference, axis=1)
    print(f'\tGlobal average bias: {np.round(avg_bias_2081_2098,2)}')

    """ Compute 5-95% for temperatures predicted by the DNNs in 1850-2098 """
    q05_predictions = np.zeros((len(short_scenarios_list),249))
    q95_predictions = np.zeros((len(short_scenarios_list),249))
    for short_scenario_idx, short_scenario in enumerate(short_scenarios_list):
        for i in range(2098-1850+1):
            q05_predictions[short_scenario_idx,i] = np.percentile(annual_predictions_means[:,short_scenario_idx,i],5)
            q95_predictions[short_scenario_idx,i] = np.percentile(annual_predictions_means[:,short_scenario_idx,i],95)

    print(f'\t5% with respect to the average: [{np.round(q05_predictions[0,2081-1850:2098-1850+1].mean()-ensemble_predictions_means[0,2081-1850:2098-1850+1].mean(),2)},{np.round(q05_predictions[1,2081-1850:2098-1850+1].mean()-ensemble_predictions_means[1,2081-1850:2098-1850+1].mean(),2)},{np.round(q05_predictions[2,2081-1850:2098-1850+1].mean()-ensemble_predictions_means[2,2081-1850:2098-1850+1].mean(),2)}]')
    print(f'\t95% with respect to the average: [{np.round(q95_predictions[0,2081-1850:2098-1850+1].mean()-ensemble_predictions_means[0,2081-1850:2098-1850+1].mean(),2)},{np.round(q95_predictions[1,2081-1850:2098-1850+1].mean()-ensemble_predictions_means[1,2081-1850:2098-1850+1].mean(),2)},{np.round(q95_predictions[2,2081-1850:2098-1850+1].mean()-ensemble_predictions_means[2,2081-1850:2098-1850+1].mean(),2)}]')

    """ Compute RMSE """
    difference_means = annual_predictions_means - annual_simulations_means # (21,3,249)
    squared_diff = difference_means[:,:,2081-1850:2098-1850+1] ** 2 # (21,3,249)
    ms_diff = np.mean(squared_diff, axis=0) # (3,18)
    rms_years = np.sqrt(ms_diff)
    avg_rms_years = np.mean(rms_years, axis=1)
    print(f'\tGlobal RMSE: {np.round(avg_rms_years,2)}')