"""
Author: Francesco Immorlano

Script for reproducing images used in Extended Data Figure 2
"""

import os
from netCDF4 import Dataset
import numpy as np
import pickle
import matplotlib.pyplot as plt

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

n_lat_points = 64
n_lon_points = 128
short_scenarios_list = ['ssp245', 'ssp370', 'ssp585']
variable_short = 'tas'

ROOT_DATA = '../Source_data'
SIMULATIONS_DIRECTORY = f'{ROOT_DATA}/CMIP6_data/near_surface_air_temperature/Annual_uniform_remapped'

first_year = 1850
last_year = 2098
n_projections_years = last_year-first_year+1

total_earth_area = 5.1009974e+14
with open('../area_cella.csv', newline='') as csvfile:
    area_cella = np.genfromtxt(csvfile, delimiter=',')

""" Load DNNs predictions after pre-training """
predictions = np.zeros((len(models_list), 3, n_projections_years, n_lat_points, n_lon_points))
pickle_in = open(f'{ROOT_DATA}/First_Training/Predictions_on_training_set.pickle', 'rb')
predictions = pickle.load(pickle_in)
pickle_in.close()

simulations = np.zeros((len(models_list), 3, n_projections_years, n_lat_points, n_lon_points))

""" Load CMIP6 ESMs simulations """
for idx_model, model in enumerate(models_list):
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

# Convert from K to Celsius degrees
predictions_C = predictions - 273.15
simulations_C = simulations - 273.15

# Compute average global surface air temperature
annual_predictions_means_C = ((predictions_C * area_cella).sum(axis=(-1,-2)))/total_earth_area
annual_simulations_means_C = ((simulations_C * area_cella).sum(axis=(-1,-2)))/total_earth_area

""" Plot """
plt.rcParams.update({'font.sans-serif': 'Arial'})
fig, axs = plt.subplots(len(short_scenarios_list),2, figsize=(40,30))
plt.subplots_adjust(wspace=0.1, hspace=0.4)
for scenario_short_idx, scenario_short in enumerate(short_scenarios_list):
    for model_idx, model in enumerate(models_list):
        axs[scenario_short_idx,0].plot(np.arange(1850, 2099), annual_simulations_means_C[model_idx,scenario_short_idx,:], linewidth=1, label=f'{model}')
        axs[scenario_short_idx,1].plot(np.arange(1850, 2099), annual_predictions_means_C[model_idx, scenario_short_idx,:], linewidth=2, linestyle='--', label=f'{model}')

    scenario = f'SSP{scenario_short[-3]}-{scenario_short[-2]}.{scenario_short[-1]}'
    
    axs[scenario_short_idx,0].set_title(f'CMIP6 — {scenario}', size=35, pad=15)
    axs[scenario_short_idx,1].set_title(f'DNNs — {scenario}', size=35, pad=15)

    axs[scenario_short_idx,0].tick_params(axis='both', which='major', labelsize=25)
    axs[scenario_short_idx,0].set_xticks([1850, 1900, 1950, 2000, 2050, 2098])

    axs[scenario_short_idx,1].tick_params(axis='both', which='major', labelsize=25)
    axs[scenario_short_idx,1].set_xticks([1850, 1900, 1950, 2000, 2050, 2098])
    
fig.add_subplot(1, 1, 1, frame_on=False)
plt.tick_params(labelcolor="none", bottom=False, left=False)
plt.xlabel('Years', fontsize=35, labelpad=50)
plt.ylabel('Near surface air temperature [°C]', fontsize=35, labelpad=50)
plt.text(x=0.11, y=0.9, s=f'a', fontweight='bold',
         fontsize=45, ha="center", transform=fig.transFigure)
plt.text(x=0.11, y=0.62, s=f'b', fontweight='bold',
         fontsize=45, ha="center", transform=fig.transFigure)
plt.text(x=0.11, y=0.34, s=f'c', fontweight='bold',
         fontsize=45, ha="center", transform=fig.transFigure)
plt.text(x=0.515, y=0.9, s=f'd', fontweight='bold',
         fontsize=45, ha="center", transform=fig.transFigure)
plt.text(x=0.515, y=0.62, s=f'e', fontweight='bold',
         fontsize=45, ha="center", transform=fig.transFigure)
plt.text(x=0.515, y=0.34, s=f'f', fontweight='bold',
         fontsize=45, ha="center", transform=fig.transFigure)
plt.savefig(f'Ext_Data_Fig_2.png', dpi=300, bbox_inches='tight')
plt.close()

