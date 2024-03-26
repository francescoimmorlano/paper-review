"""
Author: Francesco Immorlano

Script for reproducing Extended Data Figure 10
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

short_scenarios_list = ['ssp245', 'ssp370', 'ssp585']
variable_short = 'tas'

ROOT_DATA = '../Source_data'

SIMULATIONS_DIRECTORY = f'{ROOT_DATA}/CMIP6_data/near_surface_air_temperature/Annual_uniform_remapped'
PATH_BEST_DATA = f'{ROOT_DATA}/BEST_data/BEST_regridded_annual_1979-2022.nc'

n_BEST_datasets_per_model_scenario = 5

start_year_training = 1979
end_year_training = 2022
n_training_years = end_year_training-start_year_training+1
start_year_test = end_year_training+1
end_year_test = 2098
n_test_years = end_year_test-start_year_test+1

""" Load predictions made by the DNNs after transfer learning on observational data """
predictions = np.zeros((n_BEST_datasets_per_model_scenario, len(models_list), len(short_scenarios_list), n_training_years+n_test_years, 64, 128))
pickle_in = open(f'{ROOT_DATA}/Transfer_Learning_on_Observations/Transfer_learning_obs.pickle','rb')
predictions = pickle.load(pickle_in)

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

""" Load BEST observational data """
nc_BEST_data = Dataset(f'{PATH_BEST_DATA}', mode='r+', format='NETCDF3_CLASSIC')
n_BEST_years = nc_BEST_data['st'].shape[0]
n_lats = nc_BEST_data['lat'].shape[0]
n_lons = nc_BEST_data['lon'].shape[0]
lats = np.ma.getdata(nc_BEST_data['lat'][:])
lons = np.ma.getdata(nc_BEST_data['lon'][:])
BEST_data_array = np.zeros((n_BEST_years, n_lats, n_lons))
BEST_data_array[:,:,:] = nc_BEST_data['st'][:,:,:]
nc_BEST_data.close()

# Compute average surface air temperature maps across DNNs predictions and CMIP6 ESMs simulations
avg_predictions_maps = np.mean(predictions, axis=(0,1))
avg_simulations_maps = np.mean(simulations, axis=0)

# Compute average surface air temperature maps in 2081-2098
avg_predictions_maps_2081_2098 = np.mean(avg_predictions_maps[:,2081-1979:,:,:], axis=1)
avg_simulations_maps_2081_2098 = np.mean(avg_simulations_maps[:,2081-1850:,:,:], axis=1)

# Compute average BEST map in 1980-1990 that will be used as baseline
BEST_baseline_map_1980_1990 = np.mean(BEST_data_array[1:1990-1979+1,:,:], axis=0)

# Compute avg warming maps in 2081-2098 wrt 1980-1990
prediction_warming = avg_predictions_maps_2081_2098 - BEST_baseline_map_1980_1990
simulation_warming = avg_simulations_maps_2081_2098 - BEST_baseline_map_1980_1990

""" Plot """
size_suptitlefig = 42
size_titlefig = 37
size_title_axes = 32
size_lat_lon_coords = 22
size_colorbar_labels = 32
size_colorbar_ticks = 29
colormap = 'seismic'
n_levels = 40

scenario_short = 'ssp370'
scenario_idx = 1
scenario = f'SSP{scenario_short[-3]}-{scenario_short[-2]}.{scenario_short[-1]}'

min_value = np.concatenate((prediction_warming[scenario_idx,:,:], simulation_warming[scenario_idx,:,:])).min()
max_value = np.concatenate((prediction_warming[scenario_idx,:,:], simulation_warming[scenario_idx,:,:])).max()

fig, axs = plt.subplots(nrows=1,ncols=3,
                        subplot_kw={'projection': ccrs.Robinson()},
                        figsize=(40,10))
fig.subplots_adjust(top=0.97, wspace=0.10, hspace = 0.2)
plt.rcParams['font.sans-serif'] = 'Arial'

abs_max_value = np.max([abs(min_value), abs(max_value)])

axs=axs.flatten()
levels = np.linspace(min_value, max_value, n_levels)
cbarticks = np.arange(-2,14+2,2)

""" ax1 """
NN_ensemble_warming_cyclic_data,lons_cyclic=add_cyclic_point(prediction_warming[scenario_idx,:,:],coord=lons)
cs1=axs[0].contourf(lons_cyclic,lats,NN_ensemble_warming_cyclic_data, levels=levels,
                        transform = ccrs.PlateCarree(),
                        cmap=colormap, vmin=-30, vmax=30)
axs[0].set_title('DNNs ensemble', loc='center', size=size_title_axes, pad=17)
axs[0].coastlines()

gl1 = axs[0].gridlines(draw_labels=True, linestyle='--', linewidth=0.1, color='black')
gl1.top_labels = False
gl1.right_labels = False
gl1.xlabel_style = {'size': size_lat_lon_coords, 'color': 'black', 'weight': 'normal'}
gl1.ylabel_style = {'size': size_lat_lon_coords, 'color': 'black', 'weight': 'normal'}

cbar1 = fig.colorbar(cs1, shrink=0.8, ax=axs[0], ticks=cbarticks, orientation='horizontal', pad=0.12)
cbar1.set_label(label='Surface Air Temperature anomaly [°C]', size=size_colorbar_labels, labelpad=20)
for l in cbar1.ax.xaxis.get_ticklabels():
    l.set_size(size_colorbar_ticks)

""" ax2 """
simulation_warming_cyclic_data,lons_cyclic=add_cyclic_point(simulation_warming[scenario_idx,:,:],coord=lons)
cs2=axs[1].contourf(lons_cyclic,lats,simulation_warming_cyclic_data,levels=levels,
                        transform = ccrs.PlateCarree(),
                        cmap=colormap,vmin=-30, vmax=30)
axs[1].set_title('CMIP6 ensemble', loc='center', size=size_title_axes, pad=17)
axs[1].coastlines()

gl2 = axs[1].gridlines(draw_labels=True, linestyle='--', linewidth=0.1, color='black')
gl2.top_labels = False
gl2.right_labels = False
gl2.xlabel_style = {'size': size_lat_lon_coords, 'color': 'black', 'weight': 'normal'}
gl2.ylabel_style = {'size': size_lat_lon_coords, 'color': 'black', 'weight': 'normal'}

cbar2 = fig.colorbar(cs2, ax=axs[1], shrink=0.8, ticks=cbarticks, orientation='horizontal', pad=0.12)
cbar2.set_label(label='Surface Air Temperature anomaly [°C]', size=size_colorbar_labels, labelpad=20)
for l in cbar2.ax.xaxis.get_ticklabels():
    l.set_size(size_colorbar_ticks)

""" ax3 """
difference_min = np.min(prediction_warming[scenario_idx,:,:]-simulation_warming[scenario_idx,:,:])
difference_max = np.max(prediction_warming[scenario_idx,:,:]-simulation_warming[scenario_idx,:,:])

max_abs_difference_value = np.max([abs(difference_min), abs(difference_max)])
levels_3 = np.linspace(-max_abs_difference_value, max_abs_difference_value, 40)
cbarticks_3 = [-7,-5,-3,-1,0,1,3,5,7]

difference_warming_cyclic_data,lons_cyclic=add_cyclic_point(prediction_warming[scenario_idx,:,:]-simulation_warming[scenario_idx,:,:],coord=lons)
cs3=axs[2].contourf(lons_cyclic,lats,difference_warming_cyclic_data, levels=40,
                        transform = ccrs.PlateCarree(),
                        cmap=colormap, vmin=-max_abs_difference_value-6, vmax=max_abs_difference_value+6)
axs[2].set_title('Difference (DNNs—CMIP6)', loc='center', size=size_title_axes, pad=17)
axs[2].coastlines()

gl3 = axs[2].gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linestyle='--', linewidth=0.1, color='black')
gl3.top_labels = False
gl3.right_labels = False
gl3.xlabel_style = {'size': size_lat_lon_coords, 'color': 'black', 'weight': 'normal'}
gl3.ylabel_style = {'size': size_lat_lon_coords, 'color': 'black', 'weight': 'normal'}

cbar3 = fig.colorbar(cs3, shrink=0.8, ax=axs[2], orientation='horizontal', pad=0.12, ticks=cbarticks_3)
cbar3.set_label(label='Surface Air Temperature difference [°C]', size=size_colorbar_labels, labelpad=20)
for l in cbar3.ax.xaxis.get_ticklabels():
    l.set_size(size_colorbar_ticks)

plt.text(x=0.12, y=0.88, s='a', fontweight='bold',
         fontsize=size_suptitlefig, ha="center", transform=fig.transFigure)
plt.text(x=0.385, y=0.88, s='b', fontweight='bold',
         fontsize=size_suptitlefig, ha="center", transform=fig.transFigure)
plt.text(x=0.65, y=0.88, s='c', fontweight='bold',
         fontsize=size_suptitlefig, ha="center", transform=fig.transFigure)

plt.savefig(f'Ext_Data_Fig_10.png', bbox_inches = 'tight', dpi=300)
plt.close()