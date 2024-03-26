"""
Author: Francesco Immorlano

Script for reproducing Supplementary Table 5
"""

import os
from netCDF4 import Dataset
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import sys

sys.path.insert(1, './..')
from lib import *

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

# window to compute time-to-threshold uncertainty
window_size = 21

""" Load predictions made by the DNNs after transfer learning on observational data """
pickle_in = open(f'{ROOT_DATA}/Transfer_Learning_on_Observations/Transfer_learning_obs.pickle','rb')
predictions = pickle.load(pickle_in)

""" Load CMIP6 simulations """
simulations_files_list = os.listdir(SIMULATIONS_DIRECTORY)
simulations_files_list.sort()
simulations = np.zeros((len(short_scenarios_list),len(models_list), 2098-1850+1, 64, 128))
for short_scenario_idx, short_scenario in enumerate(short_scenarios_list):
    for model_idx, model in enumerate(models_list):
        matching_simulations = [simulation_file for simulation_file in simulations_files_list if ((model in simulation_file and 'historical' in simulation_file)
                                                                                                or (model in simulation_file and short_scenario in simulation_file))]
        # maching_simuations[0] is the historical and matching_simulations[1] is the SSP simulation because of the sort operation
        # (for each model, the first simulation is the historical and then the SSP) 
        nc_historical_data = Dataset(f'{SIMULATIONS_DIRECTORY}/{matching_simulations[0]}', mode='r+', format='NETCDF3_CLASSIC')
        nc_ssp_data = Dataset(f'{SIMULATIONS_DIRECTORY}/{matching_simulations[1]}', mode='r+', format='NETCDF3_CLASSIC')
        n_historical_years = nc_historical_data[variable_short].shape[0]
        n_ssp_years = nc_ssp_data[variable_short].shape[0]
        n_lats = nc_ssp_data['lat'].shape[0]
        n_lons = nc_ssp_data['lon'].shape[0]
        simulations[short_scenario_idx,model_idx,:n_historical_years,:,:] = nc_historical_data[variable_short][:]
        if (n_ssp_years == 84):
            simulations[short_scenario_idx,model_idx,n_historical_years:,:,:] = nc_ssp_data[variable_short][:,:,:]
        elif (n_ssp_years == 85):
            simulations[short_scenario_idx,model_idx,n_historical_years:,:,:] = nc_ssp_data[variable_short][:-1,:,:]
        elif (n_ssp_years == 86):
            simulations[short_scenario_idx,model_idx,n_historical_years:,:,:] = nc_ssp_data[variable_short][:-2,:,:]
        nc_historical_data.close()
        nc_ssp_data.close()

# Convert from K to Celsius degrees
predictions_C = predictions - 273.15

# Compute average global surface air temperature
annual_predictions_means = ((predictions_C * area_cella).sum(axis=(-1,-2)))/total_earth_area

# Compute average across CMIP6 ESMs and DNNs predictions
ensemble_predictions_means = np.mean(annual_predictions_means, axis=(0,1))

# Compute warming wrt pre-industrial period
warming_ensemble_predictions_means = ensemble_predictions_means - global_mean_temp_1850_1900
warming_annual_predictions_means = annual_predictions_means - global_mean_temp_1850_1900

years_to_2_threshold_array = np.zeros((predictions.shape[0], len(models_list), len(short_scenarios_list))) # (5,22,3)
years_to_1_5_threshold_array = np.zeros((predictions.shape[0], len(models_list), len(short_scenarios_list))) # (5,22,3)

q05_years_to_2_threshold = np.zeros((len(short_scenarios_list))) # (3)
q95_years_to_2_threshold = np.zeros((len(short_scenarios_list))) # (3)

q05_years_to_1_5_threshold = np.zeros((len(short_scenarios_list))) # (3)
q95_years_to_1_5_threshold = np.zeros((len(short_scenarios_list))) # (3)

years_to_thresholds_ensemble = np.zeros((len(short_scenarios_list), 2))

for short_scenario_idx, short_scenario in enumerate(short_scenarios_list):
    years_to_thresholds_ensemble[short_scenario_idx] = compute_years_to_threshold(window_size, warming_ensemble_predictions_means[short_scenario_idx,:])

    for dataset_index in range(predictions.shape[0]):
        for model_idx, model in enumerate(models_list):
                year_to_thresholds = compute_years_to_threshold(window_size, warming_annual_predictions_means[dataset_index, model_idx, short_scenario_idx,:])

                years_to_1_5_threshold_array[dataset_index,model_idx,short_scenario_idx] = year_to_thresholds[0] # (5,22,3)
                years_to_2_threshold_array[dataset_index,model_idx,short_scenario_idx] = year_to_thresholds[1] # (5,22,3)

    q05_years_to_2_threshold[short_scenario_idx] = np.percentile(years_to_2_threshold_array[:,:,short_scenario_idx],5) #,method='nearest') # (3)
    q95_years_to_2_threshold[short_scenario_idx] = np.percentile(years_to_2_threshold_array[:,:,short_scenario_idx],95) #,method='nearest') # (3)
    
    q05_years_to_1_5_threshold[short_scenario_idx] = np.percentile(years_to_1_5_threshold_array[:,:,short_scenario_idx],5) #,method='nearest') # (3)
    q95_years_to_1_5_threshold[short_scenario_idx] = np.percentile(years_to_1_5_threshold_array[:,:,short_scenario_idx],95) #,method='nearest') # (3)

years_to_thresholds_ensemble = np.round(years_to_thresholds_ensemble).astype(int)
q05_years_to_2_threshold = np.round(q05_years_to_2_threshold).astype(int)
q95_years_to_2_threshold = np.round(q95_years_to_2_threshold).astype(int)
q05_years_to_1_5_threshold = np.round(q05_years_to_1_5_threshold).astype(int)
q95_years_to_1_5_threshold = np.round(q95_years_to_1_5_threshold).astype(int)



print(f'\nYear exceeding 1.5°C')
print(f'\t SSP2-4.5: {years_to_thresholds_ensemble[0,0]} ({q05_years_to_1_5_threshold[0]}—{q95_years_to_1_5_threshold[0]})')
print(f'\t SSP3-7.0: {years_to_thresholds_ensemble[1,0]} ({q05_years_to_1_5_threshold[1]}—{q95_years_to_1_5_threshold[1]})')
print(f'\t SSP5-8.5: {years_to_thresholds_ensemble[2,0]} ({q05_years_to_1_5_threshold[2]}—{q95_years_to_1_5_threshold[2]})\n')
print(f'Year exceeding 2°C')
print(f'\t SSP2-4.5: {years_to_thresholds_ensemble[0,1]} ({q05_years_to_2_threshold[0]}—{q95_years_to_2_threshold[0]})')
print(f'\t SSP3-7.0: {years_to_thresholds_ensemble[1,1]} ({q05_years_to_2_threshold[1]}—{q95_years_to_2_threshold[1]})')
print(f'\t SSP5-8.5: {years_to_thresholds_ensemble[2,1]} ({q05_years_to_2_threshold[2]}—{q95_years_to_2_threshold[2]})\n')

