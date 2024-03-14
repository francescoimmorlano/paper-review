"""
Author: Francesco Immorlano

Script for reproducing Table S3
"""

import os
from netCDF4 import Dataset
import numpy as np
import matplotlib.pyplot as plt
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
PATH_BEST_DATA = f'{ROOT_DATA}/BEST_data/BEST_regridded_annual_1979-2022.nc'
PATH_BEST_DATA_UNCERTAINTY = f'{ROOT_DATA}/BEST_data/Land_and_Ocean_global_average_annual.txt'

total_earth_area = 5.1009974e+14
# Avg global surface temperature in 1995-2014
global_mean_temp_1995_2014 = 14.711500000000001
with open('../area_cella.csv', newline='') as csvfile:
    area_cella = np.genfromtxt(csvfile, delimiter=',')

n_BEST_datasets_per_model_scenario = 5

start_year_training = 1979
end_year_training = 2022
n_training_years = end_year_training-start_year_training+1
start_year_test = end_year_training+1
end_year_test = 2098
n_test_years = end_year_test-start_year_test+1

# settings for modern reference time period and proxy for pre-industrial time period
refperiod_start = 1995
refperiod_end   = 2014
piperiod_start  = 1850
piperiod_end    = 1900

# historical warming estimate based on cross-chapter box 2.3 (https://www.ipcc.ch/report/ar6/wg1/downloads/report/IPCC_AR6_WGI_Chapter02.pdf)
refperiod_conversion = 0.85

""" Load DNNs predictions """
predictions = np.zeros((n_BEST_datasets_per_model_scenario, len(models_list), len(short_scenarios_list), n_training_years+n_test_years, 64, 128))
pickle_in = open(f'{ROOT_DATA}/Transfer_Learning_on_Observations/Transfer_learning_obs.pickle','rb')
predictions = pickle.load(pickle_in)

""" Load CMIP6 ESMs simulations """
simulation_array = np.zeros((len(models_list), len(short_scenarios_list), 249, 64, 128))
for model_idx, model in enumerate(models_list):
    for scenario_idx, scenario_short in enumerate(short_scenarios_list):
        # CMIP6 ESMs simulations
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
        simulation_array[model_idx, scenario_idx,:n_historical_years] = nc_historical_data[variable_short][:]
        if (n_ssp_years == 86):
            simulation_array[model_idx,scenario_idx,n_historical_years:] = nc_ssp_data[variable_short][:-2]
        elif (n_ssp_years == 85):
            simulation_array[model_idx,scenario_idx,n_historical_years:] = nc_ssp_data[variable_short][:-1]
        elif (n_ssp_years == 84):
            simulation_array[model_idx,scenario_idx,n_historical_years:] = nc_ssp_data[variable_short][:]
        nc_historical_data.close()
        nc_ssp_data.close()

# Convert from K to Celsius degrees
predictions_C = predictions - 273.15
simulation_array_C = simulation_array - 273.15

# Compute average global surface air temperature
annual_predictions_means = ((predictions_C * area_cella).sum(axis=(-1,-2)))/total_earth_area
annual_simulations_means = ((simulation_array_C * area_cella).sum(axis=(-1,-2)))/total_earth_area

# Compute warming wrt pre-industrial period
warming_predictions_means = annual_predictions_means - global_mean_temp_1995_2014
warming_simulations_means = annual_simulations_means - global_mean_temp_1995_2014

# Compute avg warming in 2081-2098
warming_predictions_means_2081_2098 = warming_predictions_means[:,:,:,2081-1979:]
warming_simulations_means_2081_2098 = warming_simulations_means[:,:,2081-1850:]

# Compute median, 5% and 95%
median_predictions_means_2081_2098 = np.zeros((len(short_scenarios_list),18))
median_simulations_means_2081_2098 = np.zeros((len(short_scenarios_list),18))
q05_predictions_means_2081_2098 = np.zeros((len(short_scenarios_list),18))
q05_simulations_means_2081_2098 = np.zeros((len(short_scenarios_list),18))
q95_predictions_means_2081_2098 = np.zeros((len(short_scenarios_list),18))
q95_simulations_means_2081_2098 = np.zeros((len(short_scenarios_list),18))
for short_scenario_idx, short_scenario in enumerate(short_scenarios_list):
    # DNNs predictions
    for i in range(18):
        median_predictions_means_2081_2098[short_scenario_idx,i] = np.median(np.ravel(warming_predictions_means_2081_2098[:,:,short_scenario_idx,i]))
        q05_predictions_means_2081_2098[short_scenario_idx,i] = np.percentile(warming_predictions_means_2081_2098[:,:,short_scenario_idx,i],5)
        q95_predictions_means_2081_2098[short_scenario_idx,i] = np.percentile(warming_predictions_means_2081_2098[:,:,short_scenario_idx,i],95)
    # CMIP6 ESMs simulations
    for i in range(18):
        median_simulations_means_2081_2098[short_scenario_idx,i] = np.median(np.ravel(warming_simulations_means_2081_2098[:,short_scenario_idx,i]))
        q05_simulations_means_2081_2098[short_scenario_idx,i] = np.percentile(warming_simulations_means_2081_2098[:,short_scenario_idx,i],5)
        q95_simulations_means_2081_2098[short_scenario_idx,i] = np.percentile(warming_simulations_means_2081_2098[:,short_scenario_idx,i],95)
 
# DNNs predictions
avg_median_ensemble = median_predictions_means_2081_2098.mean(axis=1)
q05_ensemble = q05_predictions_means_2081_2098.mean(axis=1)
q95_ensemble = q95_predictions_means_2081_2098.mean(axis=1)
# CMIP6 ESMs simulations
avg_median_simulations = median_simulations_means_2081_2098.mean(axis=1)
q05_simulations = q05_simulations_means_2081_2098.mean(axis=1)
q95_simulations = q95_simulations_means_2081_2098.mean(axis=1)

"""
5%, median, 95% average temperature values in 2081–2100 wrt 1995-2014
by Tokarska, Liang and IPCC WGI
5%, median, 95% average temperature values in 2081–2098 wrt 1995-2014
by Ribes
"""
ribes_q05 = [1.22, 2.07, 2.4]
ribes_mean = [1.83, 2.77, 3.46]
ribes_q95 = [2.44, 3.46, 4.53]

tokarska_q05 = [1.04, 1.75, 2.09]
tokarska_median = [1.81, 2.7, 3.43]
tokarska_q95 = [2.56, 3.63, 4.75]

yongxiao_q05 = [1.33, 2.28, 2.6]
yongxiao_median = [1.69, 2.65, 3.26]
yongxiao_q95 = [2.72, 3.85, 4.86]

# Values from IPCC AR6 Ch.4 Table 4.5
ipcc_wg1_q05 = [1.2, 2.0, 2.4]
ipcc_wg1_median = [1.8, 2.8, 3.5]
ipcc_wg1_q95 = [2.6, 3.7, 4.8]

q05_ensemble = np.round(q05_ensemble,2)
q95_ensemble = np.round(q95_ensemble,2)

print('\nDNNs ensemble warming in 2081–2098 wrt 1995–2014: (5–95% confidence range)')
print(f'\tSSP2-4.5: {np.round(avg_median_ensemble[0],2)} ({np.round(q05_ensemble[0],2)}—{np.round(q95_ensemble[0],2)})')
print(f'\tSSP3-7.0: {np.round(avg_median_ensemble[1],2)} ({np.round(q05_ensemble[1],2)}—{np.round(q95_ensemble[1],2)})')
print(f'\tSSP5-8.5: {np.round(avg_median_ensemble[2],2)} ({np.round(q05_ensemble[2],2)}—{np.round(q95_ensemble[2],2)})\n')

print('\nCMIP6 ensemble warming in 2081–2098 wrt 1995–2014: (5–95% confidence range)')
print(f'\tSSP2-4.5: {np.round(avg_median_simulations[0],2)} ({np.round(q05_simulations[0],2)}—{np.round(q95_simulations[0],2)})')
print(f'\tSSP3-7.0: {np.round(avg_median_simulations[1],2)} ({np.round(q05_simulations[1],2)}—{np.round(q95_simulations[1],2)})')
print(f'\tSSP5-8.5: {np.round(avg_median_simulations[2],2)} ({np.round(q05_simulations[2],2)}—{np.round(q95_simulations[2],2)})\n')

print('Uncertainty reduction')
for idx_short_scenario, scenario_short in enumerate(short_scenarios_list):
    print(f'SSP{scenario_short[-3]}-{scenario_short[-2]}.{scenario_short[-1]}')
    print(f'\tRibes:\t\t{np.round(((ribes_q95[idx_short_scenario]-ribes_q05[idx_short_scenario])-(q95_ensemble[idx_short_scenario]-q05_ensemble[idx_short_scenario]))/(ribes_q95[idx_short_scenario]-ribes_q05[idx_short_scenario])*100).astype(int)}%')
    print(f'\tLiang:\t\t{np.round(((yongxiao_q95[idx_short_scenario]-yongxiao_q05[idx_short_scenario])-(q95_ensemble[idx_short_scenario]-q05_ensemble[idx_short_scenario]))/(yongxiao_q95[idx_short_scenario]-yongxiao_q05[idx_short_scenario])*100).astype(int)}%')
    print(f'\tTokarska:\t{np.round(((tokarska_q95[idx_short_scenario]-tokarska_q05[idx_short_scenario])-(q95_ensemble[idx_short_scenario]-q05_ensemble[idx_short_scenario]))/(tokarska_q95[idx_short_scenario]-tokarska_q05[idx_short_scenario])*100).astype(int)}%')
    print(f'\tIPCC WG1 AR6:\t{np.round(((ipcc_wg1_q95[idx_short_scenario]-ipcc_wg1_q05[idx_short_scenario])-(q95_ensemble[idx_short_scenario]-q05_ensemble[idx_short_scenario]))/(ipcc_wg1_q95[idx_short_scenario]-ipcc_wg1_q05[idx_short_scenario])*100).astype(int)}%')
    print(f'\tCMIP6 ESMs:\t{np.round(((q95_simulations[idx_short_scenario]-q05_simulations[idx_short_scenario])-(q95_ensemble[idx_short_scenario]-q05_ensemble[idx_short_scenario]))/(q95_simulations[idx_short_scenario]-q05_simulations[idx_short_scenario])*100).astype(int)}%')
