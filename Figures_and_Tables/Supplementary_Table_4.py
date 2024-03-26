"""
Author: Francesco Immorlano

Script for reproducing Supplementary Table 4
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
# Avg global surface temperature in 1995-2014
global_mean_temp_1995_2014 = 14.711500000000001
with open('../area_cella.csv', newline='') as csvfile:
    area_cella = np.genfromtxt(csvfile, delimiter=',')

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
simulations_C = simulations - 273.15
predictions_C = predictions - 273.15

# Compute average global surface air temperature
annual_simulations_means = ((simulations_C * area_cella).sum(axis=(-1,-2)))/total_earth_area
annual_predictions_means = ((predictions_C * area_cella).sum(axis=(-1,-2)))/total_earth_area

# Compute average across CMIP6 ESMs and DNNs predictions
ensemble_simulations_means = np.mean(annual_simulations_means, axis=(1))
ensemble_predictions_means = np.mean(annual_predictions_means, axis=(0,1))

# Compute warming wrt pre-industrial period
warming_ensemble_simulations_means = ensemble_simulations_means - global_mean_temp_1995_2014
warming_ensemble_predictions_means = ensemble_predictions_means - global_mean_temp_1995_2014
warming_annual_simulations_means = annual_simulations_means - global_mean_temp_1995_2014
warming_annual_predictions_means = annual_predictions_means - global_mean_temp_1995_2014

""" Compute 5% and 95% """
# DNNs predictions
q05_predictions = np.zeros((len(short_scenarios_list),2098-1979+1))
q95_predictions = np.zeros((len(short_scenarios_list),2098-1979+1))
for short_scenario_idx, short_scenario in enumerate(short_scenarios_list):
    for i in range(2098-1979+1):
        q05_predictions[short_scenario_idx,i] = np.percentile(warming_annual_predictions_means[:,:,short_scenario_idx,i],5)
        q95_predictions[short_scenario_idx,i] = np.percentile(warming_annual_predictions_means[:,:,short_scenario_idx,i],95)
# CMIP6 ESMs simulations
q05_simulations = np.zeros((len(short_scenarios_list),2098-1850+1))
q95_simulations = np.zeros((len(short_scenarios_list),2098-1850+1))
for short_scenario_idx, short_scenario in enumerate(short_scenarios_list):
    for i in range(2098-1850+1):
        q05_simulations[short_scenario_idx,i] = np.percentile(warming_annual_simulations_means[short_scenario_idx,:,i],5)
        q95_simulations[short_scenario_idx,i] = np.percentile(warming_annual_simulations_means[short_scenario_idx,:,i],95)

print(f'\nDNNs average global temperature anomaly in 2021-2040 wrt 1850-1900: avg (5%—95%)')
print(f'\tSSP2-4.5: {np.round(np.mean(warming_ensemble_predictions_means[0,2021-1979:2040-1979+1]),2)} ({np.round(np.mean(q05_predictions[0,2021-1979:2040-1979+1]),2)}—{np.round(np.mean(q95_predictions[0,2021-1979:2040-1979+1]),2)})')
print(f'\tSSP3-7.0: {np.round(np.mean(warming_ensemble_predictions_means[1,2021-1979:2040-1979+1]),2)} ({np.round(np.mean(q05_predictions[1,2021-1979:2040-1979+1]),2)}—{np.round(np.mean(q95_predictions[1,2021-1979:2040-1979+1]),2)})')
print(f'\tSSP5-8.5: {np.round(np.mean(warming_ensemble_predictions_means[2,2021-1979:2040-1979+1]),2)} ({np.round(np.mean(q05_predictions[2,2021-1979:2040-1979+1]),2)}—{np.round(np.mean(q95_predictions[2,2021-1979:2040-1979+1]),2)})')

print(f'\nDNNs average global temperature anomaly in 2041-2060 wrt 1850-1900: avg (5%—95%)')
print(f'\tSSP2-4.5: {np.round(np.mean(warming_ensemble_predictions_means[0,2041-1979:2060-1979+1]),2)} ({np.round(np.mean(q05_predictions[0,2041-1979:2060-1979+1]),2)}—{np.round(np.mean(q95_predictions[0,2041-1979:2060-1979+1]),2)})')
print(f'\tSSP3-7.0: {np.round(np.mean(warming_ensemble_predictions_means[1,2041-1979:2060-1979+1]),2)} ({np.round(np.mean(q05_predictions[1,2041-1979:2060-1979+1]),2)}—{np.round(np.mean(q95_predictions[1,2041-1979:2060-1979+1]),2)})')
print(f'\tSSP5-8.5: {np.round(np.mean(warming_ensemble_predictions_means[2,2041-1979:2060-1979+1]),2)} ({np.round(np.mean(q05_predictions[2,2041-1979:2060-1979+1]),2)}—{np.round(np.mean(q95_predictions[2,2041-1979:2060-1979+1]),2)})')

# Values from IPCC WG1 AR6 
ipcc_wg1_2021_2040_q05 = [0.4, 0.4, 0.5]
ipcc_wg1_2021_2040_q95 = [0.9, 0.9, 1]
ipcc_wg1_2041_2060_q05 = [0.8, 0.9, 1.1]
ipcc_wg1_2041_2060_q95 = [1.6, 1.7, 2.1]

print('\nUncertainty reduction with respect to IPCC WG1 AR6')
for idx_short_scenario, scenario_short in enumerate(short_scenarios_list):
    print(f'SSP{scenario_short[-3]}-{scenario_short[-2]}.{scenario_short[-1]}')
    print(f'\tNear-term (2021-2040):\t\t{np.round(((ipcc_wg1_2021_2040_q95[idx_short_scenario]-ipcc_wg1_2021_2040_q05[idx_short_scenario])-(np.round(np.mean(q95_predictions[idx_short_scenario,2021-1979:2040-1979+1]),2)-np.round(np.mean(q05_predictions[idx_short_scenario,2021-1979:2040-1979+1]),2)))/(ipcc_wg1_2021_2040_q95[idx_short_scenario]-ipcc_wg1_2021_2040_q05[idx_short_scenario])*100).astype(int)}%')
    print(f'\tMid-term (2041-2060):\t\t{np.round(((ipcc_wg1_2041_2060_q95[idx_short_scenario]-ipcc_wg1_2041_2060_q05[idx_short_scenario])-(np.round(np.mean(q95_predictions[idx_short_scenario,2041-1979:2060-1979+1]),2)-np.round(np.mean(q05_predictions[idx_short_scenario,2041-1979:2060-1979+1]),2)))/(ipcc_wg1_2041_2060_q95[idx_short_scenario]-ipcc_wg1_2041_2060_q05[idx_short_scenario])*100).astype(int)}%')