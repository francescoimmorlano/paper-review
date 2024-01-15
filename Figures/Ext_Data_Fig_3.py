"""
Author: Francesco Immorlano

Script for reproducing images used in Extended Data Figure 3
"""

import os
from netCDF4 import Dataset
import numpy as np
import matplotlib.pyplot as plt
import pickle

from cartopy.util import add_cyclic_point
import cartopy.crs as ccrs


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

pickle_in = open('lons.pickle', 'rb')
lons = pickle.load(pickle_in)
pickle_in.close()

pickle_in = open('lats.pickle', 'rb')
lats = pickle.load(pickle_in)
pickle_in.close()

# Compute DNNs ensemble and CMIP6 ensemble in 2095
ensemble_predictions_2095 = np.mean(predictions[:,:,2095-1850,:,:], axis=0)
ensemble_simulations_2095 = np.mean(simulations[:,:,2095-1850,:,:], axis=0)

""" Plot """
size_suptitlefig = 46
size_titlefig = 37
size_title_axes = 32
size_lat_lon_coords = 35
size_colorbar_labels = 40
size_colorbar_ticks = 35
colormap = 'seismic'

scenario_short = 'ssp370'
scenario_idx = 1
scenario = f'SSP{scenario_short[-3]}-{scenario_short[-2]}.{scenario_short[-1]}'

""" ax 0 """
difference_min = np.concatenate(ensemble_predictions_2095[:,:,:]-ensemble_simulations_2095[:,:,:]).min()
difference_max = np.concatenate(ensemble_predictions_2095[:,:,:]-ensemble_simulations_2095[:,:,:]).max()
max_abs_difference_value = np.max([abs(difference_min), abs(difference_max)])

n_levels = 40
levels = np.linspace(difference_min, difference_max, n_levels)
cbarticks_2 = [-1.2, -1.0, -0.8, -0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6]
vmin = -max_abs_difference_value-5
vmax = max_abs_difference_value+5

fig, axs = plt.subplots(nrows=1,ncols=3,
                        subplot_kw={'projection': ccrs.Robinson()},
                        figsize=(40,10))
plt.rcParams.update({'font.sans-serif': 'Arial'})
fig.subplots_adjust(top=0.97, wspace=0.2, hspace = 0.2)

axs=axs.flatten()

for idx_short_scenario, short_scenario in enumerate(short_scenarios_list):

    scenario = f'SSP{short_scenario[-3]}-{short_scenario[-2]}.{short_scenario[-1]}'

    difference_warming_cyclic_data,lons_cyclic=add_cyclic_point(ensemble_predictions_2095[idx_short_scenario,:,:]-ensemble_simulations_2095[idx_short_scenario,:,:],coord=lons)
    cs0=axs[idx_short_scenario].contourf(lons_cyclic,lats,difference_warming_cyclic_data, levels=levels,
                            transform = ccrs.PlateCarree(),
                            cmap=colormap, vmin=-max_abs_difference_value-5, vmax=max_abs_difference_value+5)
    axs[idx_short_scenario].coastlines()
    axs[idx_short_scenario].set_title(f'{scenario}', fontsize=40, pad=10)

    gl0 = axs[idx_short_scenario].gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linestyle='--', linewidth=0.05, color='black')
    gl0.xlabels_top = False
    gl0.ylabels_right = False
    gl0.xlabel_style = {'size': size_lat_lon_coords, 'color': 'black', 'weight': 'normal'}
    gl0.ylabel_style = {'size': size_lat_lon_coords, 'color': 'black', 'weight': 'normal'}

    plt.draw()
    for ea in gl0.ylabel_artists:
        right_label = ea.get_position()[0] > 0
        if right_label:
            ea.set_visible(False)

    plt.text(x=0.5, y=1, s=f'Bias (DNNs-CMIP6) — Validation Year: 2095',
            fontsize=size_suptitlefig, ha="center", transform=fig.transFigure)
    
    plt.text(x=0.1, y=0.89, s=f'a', fontweight='bold',
            fontsize=size_suptitlefig, ha="center", transform=fig.transFigure)
    plt.text(x=0.37, y=0.89, s=f'b', fontweight='bold',
            fontsize=size_suptitlefig, ha="center", transform=fig.transFigure)
    plt.text(x=0.64, y=0.89, s=f'c', fontweight='bold',
            fontsize=size_suptitlefig, ha="center", transform=fig.transFigure)
    
cbar0 = fig.colorbar(cs0, shrink=0.6, ax=axs, orientation='horizontal', pad=0.15, ticks=cbarticks_2, aspect=30)
cbar0.set_label(label='Surface Air Temperature difference [°C]', size=size_colorbar_labels, labelpad=30, family='Arial')
for l in cbar0.ax.xaxis.get_ticklabels():
    l.set_family('Arial')
    l.set_size(size_colorbar_ticks)

plt.savefig('Ext_Data_Fig_3.png', bbox_inches = 'tight', dpi=300)
plt.close()
