"""
Author: Francesco Immorlano

Script for reproducing images used in Figure S7
"""

import os
from netCDF4 import Dataset
import numpy as np
import matplotlib.pyplot as plt
import pickle

models_list_complete = [
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

""" 
shuffle_idx = '01' to shuffle_idx = '22'
Set shuffle_idx = '01' to reproduce Figure S7 present in the paper
"""
shuffle_idx = '01'
models_list = models_list_complete.copy()
model_taken_out = models_list[int(shuffle_idx)-1]
models_list.remove(model_taken_out)

total_earth_area = 5.1009974e+14

with open('../area_cella.csv', newline='') as csvfile:
    area_cella = np.genfromtxt(csvfile, delimiter=',')

short_scenarios_list = ['ssp245', 'ssp370', 'ssp585']
variable_short = 'tas'

total_earth_area = 5.1009974e+14
with open('../area_cella.csv', newline='') as csvfile:
    area_cella = np.genfromtxt(csvfile, delimiter=',')

n_lat_points = 64
n_lon_points = 128

ROOT_DATA = '../Source_data'
SIMULATIONS_DIRECTORY = f'{ROOT_DATA}/CMIP6_data/near_surface_air_temperature/Annual_uniform_remapped'

print(f'\nModel taken out: {model_taken_out} - shuffle: {shuffle_idx}')
start_year_training = 2023
end_year_training = 2098
n_training_years = end_year_training-start_year_training+1

start_year_test = 1850
end_year_test = 2022
n_test_years = end_year_test-start_year_test+1

pickle_in = open(f'{ROOT_DATA}/Transfer_Learning_on_Simulations_reverse/Predictions_shuffle-{shuffle_idx}.pickle', 'rb')
predictions = pickle.load(pickle_in)

""" Load CMIP6 ESMs simulations """
simulations = np.zeros((len(models_list_complete), 3, 2098-1850+1, n_lat_points, n_lon_points))
for idx_model, model in enumerate(models_list_complete):
    for idx_scenario_short, scenario_short in enumerate(short_scenarios_list):
        scenario = f'SSP{scenario_short[-3]}-{scenario_short[-2]}.{scenario_short[-1]}'

        simulations_files_list = os.listdir(SIMULATIONS_DIRECTORY)
        simulations_files_list.sort()
        matching_simulations = [simulation_file for simulation_file in simulations_files_list if ((model in simulation_file and 'historical' in simulation_file)
                                                                                               or (model in simulation_file and scenario_short in simulation_file))]

        # maching_simulations[0] is the historical and matching_simulations[1] is the SSP simulation because of the sort operation
        # (for each model, the first simulation is the historical and then the SSP)  
        nc_historical_data = Dataset(f'{SIMULATIONS_DIRECTORY}/{matching_simulations[0]}', mode='r+', format='NETCDF3_CLASSIC')
        nc_ssp_data = Dataset(f'{SIMULATIONS_DIRECTORY}/{matching_simulations[1]}', mode='r+', format='NETCDF3_CLASSIC')

        n_historical_years = nc_historical_data[variable_short].shape[0]
        n_ssp_years = nc_ssp_data[variable_short].shape[0]
        simulations[idx_model,idx_scenario_short,:n_historical_years,:,:] = nc_historical_data[variable_short][:,:,:]
        if (n_ssp_years == 86):
            simulations[idx_model,idx_scenario_short,n_historical_years:,:,:] = nc_ssp_data[variable_short][:-2,:,:]
        elif (n_ssp_years == 85):
            simulations[idx_model,idx_scenario_short,n_historical_years:,:,:] = nc_ssp_data[variable_short][:-1,:,:]
        else:
            simulations[idx_model,idx_scenario_short,n_historical_years:,:,:] = nc_ssp_data[variable_short][:,:,:]
    nc_historical_data.close()
    nc_ssp_data.close()

print(int(shuffle_idx)-1)

taken_out_simulation = simulations[int(shuffle_idx)-1, :,:,:,:]


# Convert from K to Celsius degrees
predictions_C = predictions - 273.15
taken_out_simulation_C = taken_out_simulation - 273.15
simulations_C = simulations - 273.15

# Get average temperature maps in 2081-2098 simulated by the taken out model for SSP2-4.5
taken_out_simulation_2081_2098_ssp245 = taken_out_simulation_C[0,2081-1850:2098-1850+1,:,:]
avg_taken_out_simulation_2081_2098_ssp245 = taken_out_simulation_2081_2098_ssp245.mean(axis=1)

# Get average temperature maps in 2081-2098 predicted by the DNNs for SSP2-4.5
predictions_2081_2098_ssp245 = predictions_C[:,0,2081-1850:2098-1850+1,:,:]
avg_predictions_2081_2098_ssp245 = predictions_2081_2098_ssp245.mean(axis=(0,2))

# Get average temperature maps in 2081-2098 simulated by the taken out model for SSP2-4.5
simulations_2081_2098_ssp245 = simulations_C[0,2081-1850:2098-1850+1,:,:]
avg_simulations_2081_2098_ssp245 = simulations_2081_2098_ssp245.mean(axis=(0,2))

# Compute average global surface air temperature
annual_predictions_means = ((predictions_C * area_cella).sum(axis=(-1,-2)))/total_earth_area
annual_simulations_means = ((simulations_C * area_cella).sum(axis=(-1,-2)))/total_earth_area
annual_taken_out_simulation_means = ((taken_out_simulation_C * area_cella).sum(axis=(-1,-2)))/total_earth_area

global_mean_temp_taken_out = np.mean(annual_taken_out_simulation_means[:,:1900-1850], axis=1)

# Compute warming wrt pre-industrial period
for idx_short_scenario, short_scenario in enumerate(short_scenarios_list):
    annual_predictions_means[:,idx_short_scenario,:] -= global_mean_temp_taken_out[idx_short_scenario]
    annual_simulations_means[:,idx_short_scenario,:] -= global_mean_temp_taken_out[idx_short_scenario]
    annual_taken_out_simulation_means[idx_short_scenario,:] -= global_mean_temp_taken_out[idx_short_scenario]

# Compute average across DNNs predictions
ensemble_predictions_means = np.mean(annual_predictions_means, axis=0)

""" Compute 5-95% for temperatures predicted by the DNNs in 1850-2098 """
q05_predictions = np.zeros((len(short_scenarios_list),249))
q95_predictions = np.zeros((len(short_scenarios_list),249))
for short_scenario_idx, short_scenario in enumerate(short_scenarios_list):
    for i in range(2098-1850+1):
        q05_predictions[short_scenario_idx,i] = np.percentile(annual_predictions_means[:,short_scenario_idx,i],5)
        q95_predictions[short_scenario_idx,i] = np.percentile(annual_predictions_means[:,short_scenario_idx,i],95)

""" Compute 5-95% for temperatures simulated by CMIP6 ESMs in 1850-2098 """
q05_simulations = np.zeros((len(short_scenarios_list),2098-1850+1))
q95_simulations = np.zeros((len(short_scenarios_list),2098-1850+1))
for short_scenario_idx, short_scenario in enumerate(short_scenarios_list):
    for i in range(2098-1850+1):
        q05_simulations[short_scenario_idx,i] = np.percentile(annual_simulations_means[:,short_scenario_idx,i],5)
        q95_simulations[short_scenario_idx,i] = np.percentile(annual_simulations_means[:,short_scenario_idx,i],95)

""" Compute RMSE """
difference_means = annual_predictions_means - annual_taken_out_simulation_means
squared_diff = difference_means[:,:,:2023-1850+1] ** 2
ms_diff = np.mean(squared_diff, axis=0)
rms_years = np.sqrt(ms_diff)
rmse_scenario = np.mean(rms_years, axis=1)

fig, axs = plt.subplots(len(short_scenarios_list), figsize=(16,18))
plt.rcParams.update({'font.sans-serif': 'Helvetica'})
plt.subplots_adjust(hspace=0.5)

for scenario_short_idx, scenario_short in enumerate(short_scenarios_list):
    scenario = f'SSP{scenario_short[-3]}-{scenario_short[-2]}.{scenario_short[-1]}'
    for model_idx, model in enumerate(models_list):
        # DNNs predictions
        if model_idx==0:
            axs[scenario_short_idx].plot(np.arange(1850, 2099), annual_predictions_means[model_idx, scenario_short_idx,:], linewidth=0.4, linestyle='--', label=f'DNNs predictions', zorder=3)
        else:
            axs[scenario_short_idx].plot(np.arange(1850, 2099), annual_predictions_means[model_idx, scenario_short_idx,:], linewidth=0.4, linestyle='--', zorder=3)
    # taken out simulation
    axs[scenario_short_idx].plot(np.arange(1850, 2099), annual_taken_out_simulation_means[scenario_short_idx,:], linewidth=3, label=f'{model_taken_out} (model taken out)',  zorder=4)
    # CMIP6 spread shading
    axs[scenario_short_idx].fill_between(np.arange(1850, 2099), q05_simulations[scenario_short_idx,:], q95_simulations[scenario_short_idx,:], facecolor='#FFD67E', zorder=1)

    # train set shading
    axs[scenario_short_idx].fill_between(np.arange(start_year_training, end_year_training+1), np.floor(q05_simulations[scenario_short_idx,0]), ensemble_predictions_means[scenario_short_idx,2023-1850:2098-1850+1], color='red', alpha=0.1, zorder=0)
    # DNN ensemble
    axs[scenario_short_idx].plot(np.arange(1850, 2099), ensemble_predictions_means[scenario_short_idx,:], linewidth=3, label=f'DNNs ensemble', zorder=5)
    # DNN ensemble spread shading
    axs[scenario_short_idx].fill_between(np.arange(1850, 2099), q05_predictions[scenario_short_idx,:], q95_predictions[scenario_short_idx,:], facecolor='#7EFF99', zorder=2)
    axs[scenario_short_idx].set_xticks([1850, 1900, 1950, 2000, 2023, 2050, 2098])
    axs[scenario_short_idx].tick_params(axis='both', which='major', labelsize=15)
    axs[scenario_short_idx].set_title(f'Scenario {scenario}\nRMSE (1850–2022): {round(rmse_scenario[scenario_short_idx],2)} °C',
                                      size=19, linespacing=1.5, pad=20)

    scenario = f'SSP{scenario_short[-3]}-{scenario_short[-2]}.{scenario_short[-1]}'
    axs[scenario_short_idx].set_ylim([np.floor(q05_simulations[scenario_short_idx,0]), np.ceil(q95_simulations[scenario_short_idx,-1])+0.5])
    axs[scenario_short_idx].legend(loc='upper left', fontsize=14)

fig.add_subplot(1, 1, 1, frame_on=False)

plt.tick_params(labelcolor="none", bottom=False, left=False)

plt.xlabel('Years', fontsize=19, labelpad=30)
plt.ylabel('Near surface air temperature anomaly [°C]\nBase period: 1850–1900', fontsize=19, labelpad=30)

plt.savefig(f'Fig_S7_{model_taken_out}.png', dpi=300, bbox_inches='tight')

