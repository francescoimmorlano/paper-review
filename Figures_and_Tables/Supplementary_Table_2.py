"""
Author: Francesco Immorlano

Script to compute Supplementary Table 2
"""

import os
from netCDF4 import Dataset
import numpy as np
import pickle

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
# Avg global surface temperature in 1995-2014
global_mean_temp_1995_2014 = 14.711500000000001
with open('../area_cella.csv', newline='') as csvfile:
    area_cella = np.genfromtxt(csvfile, delimiter=',')

""" Load CMIP6 ESMs simulations """
simulations = np.zeros((len(models_list), len(short_scenarios_list), 2098-1850+1, 64, 128))
for model_idx, model in enumerate(models_list):
    for scenario_idx, scenario_short in enumerate(short_scenarios_list):
        simulations_files_list = os.listdir(SIMULATIONS_DIRECTORY)
        simulations_files_list.sort()
        matching_simulations = [simulation_file for simulation_file in simulations_files_list if ((model in simulation_file and 'historical' in simulation_file)
                                                                                                or (model in simulation_file and scenario_short in simulation_file))]
        # maching_simuations[0] is the historical and matching_simulations[1] is the SSP simulation because of the sort operation
        # (for each model, the first simulation is the historical and then the SSP) 
        nc_historical_data = Dataset(f'{SIMULATIONS_DIRECTORY}/{matching_simulations[0]}', mode='r+', format='NETCDF3_CLASSIC')
        nc_ssp_data = Dataset(f'{SIMULATIONS_DIRECTORY}/{matching_simulations[1]}', mode='r+', format='NETCDF3_CLASSIC')
        n_historical_years = nc_historical_data[variable_short].shape[0]
        n_ssp_years = nc_ssp_data[variable_short].shape[0]
        n_lats = nc_ssp_data['lat'].shape[0]
        n_lons = nc_ssp_data['lon'].shape[0]
        simulations[model_idx,scenario_idx,:n_historical_years] = nc_historical_data[variable_short][:]
        if (n_ssp_years == 86):
            simulations[model_idx,scenario_idx,n_historical_years:] = nc_ssp_data[variable_short][:-2]
        elif (n_ssp_years == 85):
            simulations[model_idx,scenario_idx,n_historical_years:] = nc_ssp_data[variable_short][:-1]
        elif (n_ssp_years == 84):
            simulations[model_idx,scenario_idx,n_historical_years:] = nc_ssp_data[variable_short][:]
        nc_historical_data.close()
        nc_ssp_data.close()


ensemble_statistics = np.zeros((len(short_scenarios_list),3,len(models_list)))            # scenarios, (median, q05, q95), models
simulations_statistics = np.zeros((len(short_scenarios_list),3,len(models_list)))         # scenarios, (avg_taken_out, q05, q95), models
accuracy = np.zeros((len(short_scenarios_list),len(models_list)))
precision_simulations = np.zeros((len(short_scenarios_list),len(models_list)))

global_avg_bias = np.zeros((len(short_scenarios_list),len(models_list)))
rmse = np.zeros((len(short_scenarios_list),len(models_list)))

for shuffle_idx in range(len(models_list)):

    if shuffle_idx < 9: shuffle_number = f'0{shuffle_idx+1}'
    else: shuffle_number = f'{shuffle_idx+1}'

    """ Load DNNs predictions after TL on the take-out ESM simulation """
    pickle_in = open(f'{ROOT_DATA}/Transfer_Learning_on_Simulations/Predictions_shuffle-{shuffle_number}.pickle', 'rb')
    predictions_tl_on_simulations = pickle.load(pickle_in)

    remaining_models_idx = np.arange(22)
    remaining_models_idx = np.delete(remaining_models_idx, shuffle_idx)

    simulations_remaining = simulations[remaining_models_idx,:,:,:]
    simulation_takeout = simulations[shuffle_idx,:,:,:]

    # Convert from K to Celsius degrees
    predictions_C = predictions_tl_on_simulations - 273.15                  # (21,3,249,64,128)
    simulation_takeout_C = simulation_takeout - 273.15                      # (3,249,64,128)
    simulations_remaining_C = simulations_remaining - 273.15                # (21,3,249,64,128)

    # Compute average global surface air temperature
    annual_predictions_means = ((predictions_C * area_cella).sum(axis=(-1,-2)))/total_earth_area                                    # (21,3,249)
    annual_simulation_takeout_means = ((simulation_takeout_C * area_cella).sum(axis=(-1,-2)))/total_earth_area                      # (3,249)
    annual_simulations_remaining_means = ((simulations_remaining_C * area_cella).sum(axis=(-1,-2)))/total_earth_area                # (21,3,249)

    # Compute warming wrt pre-industrial period
    warming_predictions_means = annual_predictions_means - global_mean_temp_1995_2014                                       # (21,3,249)
    warming_simulation_takeout_means = annual_simulation_takeout_means - global_mean_temp_1995_2014                         # (3,249)
    warming_simulations_remaining_means = annual_simulations_remaining_means - global_mean_temp_1995_2014                   # (21,3,249)

    # Select warming values in 2081-2098
    warming_predictions_means_2081_2098 = warming_predictions_means[:,:,2081-1850:2098-1850+1]                                     # (21,3,18)
    warming_simulation_takeout_means_2081_2098 = warming_simulation_takeout_means[:,2081-1850:2098-1850+1]                         # (3,18)
    warming_simulations_remaining_means_2081_2098 = warming_simulations_remaining_means[:,:,2081-1850:2098-1850+1]                 # (21,3,18)

    # Compute ensemble of warming predicted by DNNs after TL on the take-out model
    ensemble_warming_predictions_2081_2098 = np.mean(warming_predictions_means_2081_2098, axis=0)         # (3, 18)

    # Compute global avg bias
    bias = ensemble_warming_predictions_2081_2098 - warming_simulation_takeout_means_2081_2098          # (3,18)
    global_avg_bias[:,shuffle_idx] = np.mean(bias, axis=1) 

    # COmpute RMSE
    mse = np.mean(bias**2, axis=1)          # (3)
    rmse[:,shuffle_idx] = np.sqrt(mse)                     # (3)

    # Compute median, 5% and 95% for the DNNs predictions and average temperature, 5% and 95% for the CMIP6 simulations
    median_predictions_means_2081_2098 = np.zeros((len(short_scenarios_list),18))
    q05_predictions_means_2081_2098 = np.zeros((len(short_scenarios_list),18))
    q95_predictions_means_2081_2098 = np.zeros((len(short_scenarios_list),18))

    q05_simulations_remaining_means_2081_2098 = np.zeros((len(short_scenarios_list),18))
    q95_simulations_remaining_means_2081_2098 = np.zeros((len(short_scenarios_list),18))

    for short_scenario_idx, short_scenario in enumerate(short_scenarios_list):
        for i in range(18):
            # 21 DNNs predictions after TL on simulations
            median_predictions_means_2081_2098[short_scenario_idx,i] = np.median(warming_predictions_means_2081_2098[:,short_scenario_idx,i])
            q05_predictions_means_2081_2098[short_scenario_idx,i] = np.percentile(warming_predictions_means_2081_2098[:,short_scenario_idx,i],5)
            q95_predictions_means_2081_2098[short_scenario_idx,i] = np.percentile(warming_predictions_means_2081_2098[:,short_scenario_idx,i],95)
            # 21 remaining CMIP6 simulations
            q05_simulations_remaining_means_2081_2098[short_scenario_idx,i] = np.percentile(warming_simulations_remaining_means_2081_2098[:,short_scenario_idx,i],5)
            q95_simulations_remaining_means_2081_2098[short_scenario_idx,i] = np.percentile(warming_simulations_remaining_means_2081_2098[:,short_scenario_idx,i],95)
            
        ensemble_statistics[short_scenario_idx,0,shuffle_idx] = median_predictions_means_2081_2098[short_scenario_idx,:].mean()
        ensemble_statistics[short_scenario_idx,1,shuffle_idx] = q05_predictions_means_2081_2098[short_scenario_idx,:].mean()
        ensemble_statistics[short_scenario_idx,2,shuffle_idx] = q95_predictions_means_2081_2098[short_scenario_idx,:].mean()

        simulations_statistics[short_scenario_idx,0,shuffle_idx] = warming_simulation_takeout_means_2081_2098[short_scenario_idx,:].mean()
        simulations_statistics[short_scenario_idx,1,shuffle_idx] = q05_simulations_remaining_means_2081_2098[short_scenario_idx,:].mean()
        simulations_statistics[short_scenario_idx,2,shuffle_idx] = q95_simulations_remaining_means_2081_2098[short_scenario_idx,:].mean()

        accuracy[short_scenario_idx, shuffle_idx] = ensemble_statistics[short_scenario_idx,0,shuffle_idx] - simulations_statistics[short_scenario_idx,0,shuffle_idx]
        precision_simulations[short_scenario_idx, shuffle_idx] = ((simulations_statistics[short_scenario_idx,2,shuffle_idx] - simulations_statistics[short_scenario_idx,1,shuffle_idx]) - (ensemble_statistics[short_scenario_idx,2,shuffle_idx] - ensemble_statistics[short_scenario_idx,1,shuffle_idx]))/(simulations_statistics[short_scenario_idx,2,shuffle_idx] - simulations_statistics[short_scenario_idx,1,shuffle_idx])*100


    

print('accuracy: DNN - takeout')
print('precision: reduction of 21 DNNs uncertainty (after TL on simulations) wrt to the remaining 21 CMIP6 ESMs uncertainty \n')

for idx_scenario, scenario in enumerate(short_scenarios_list):
    print(f'{scenario}')
    for idx_model, model in enumerate(models_list):
        print(f'--- {model}\
              \tglob avg bias: {np.round(global_avg_bias[idx_scenario,idx_model],2)}°C \
              \trmse: {np.round(rmse[idx_scenario,idx_model],2)}°C \
              \t% uncertainty reduction: {np.round(precision_simulations[idx_scenario,idx_model],2)}\
              \taccuracy: {np.round(accuracy[idx_scenario,idx_model],2)}°C\
              \t5th: {np.round(ensemble_statistics[idx_scenario,1,idx_model],2)}°C\
              \t95th: {np.round(ensemble_statistics[idx_scenario,2,idx_model],2)}°C')