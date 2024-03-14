"""
Author: Francesco Immorlano

Script for reproducing images used in Figure S4
"""

import os
from netCDF4 import Dataset
import numpy as np
import matplotlib.pyplot as plt

from cartopy.util import add_cyclic_point
import cartopy.crs as ccrs

from tqdm import tqdm
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

total_earth_area = 5.1009974e+14
with open('../area_cella.csv', newline='') as csvfile:
    area_cella = np.genfromtxt(csvfile, delimiter=',')

global_mean_temp_1850_1900 = 13.798588235294114
global_mean_temp_1995_2014 = 14.711500000000001

n_lat_points = 64
n_lon_points = 128

start_year_training = 1850
end_year_training = 2022
n_training_years = end_year_training-start_year_training+1
start_year_test = end_year_training+1
end_year_test = 2098
n_test_years = end_year_test-start_year_test+1
n_projections_years = end_year_test-start_year_training+1

""" 
shuffle_idx = 01 to shuffle_idx = 22
Set shuffle_idx = 17 to reproduce Figure S4 present in the paper
"""
shuffle_idx = '17'
model_taken_out = models_list[int(shuffle_idx)-1]

models_list_take_out = models_list.copy()

model_take_out = models_list[int(shuffle_idx)-1]
# Delete the current take out model (i.e., the model which other models are transfer learned on) from the list 
models_list_take_out.remove(model_take_out)

ROOT_DATA = '../Source_data'
SIMULATIONS_DIRECTORY = f'{ROOT_DATA}/CMIP6_data/near_surface_air_temperature/Annual_uniform_remapped'

print(f'Model take out: {model_take_out} - shuffle: {shuffle_idx}')

""" Load DNNs predictions after pre-training """
pickle_in = open(f'{ROOT_DATA}/First_Training/Predictions_on_training_set.pickle', 'rb')
pre_train_predictions = pickle.load(pickle_in)
pickle_in.close()

""" Load predictions made by the DNNs after transfer learning on the take-out simulation """
pickle_in = open(f'{ROOT_DATA}/Transfer_learning_on_Simulations/Predictions_shuffle-{shuffle_idx}.pickle', 'rb')
predictions_after_tl = pickle.load(pickle_in)
pickle_in.close()

""" Load CMIP6 take-out simulation """
take_out_simulation = np.zeros((len(short_scenarios_list), n_training_years+n_test_years, 64, 128))
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
    lats = nc_ssp_data['lat'][:]
    lons = nc_ssp_data['lon'][:]
    take_out_simulation[scenario_idx,:n_historical_years,:,:] = nc_historical_data[variable_short][:]
    if (n_ssp_years == 84):
        take_out_simulation[scenario_idx,n_historical_years:,:,:] = nc_ssp_data[variable_short][:,:,:]
    elif (n_ssp_years == 85):
        take_out_simulation[scenario_idx,n_historical_years:,:,:] = nc_ssp_data[variable_short][:-1,:,:]
    elif (n_ssp_years == 86):
        take_out_simulation[scenario_idx,n_historical_years:,:,:] = nc_ssp_data[variable_short][:-2,:,:]
    nc_historical_data.close()
    nc_ssp_data.close()

""" Load latitude grid points """
pickle_in = open(f'lats.pickle', 'rb')
lats = pickle.load(pickle_in)
pickle_in.close()

""" Load longitude grid points """
pickle_in = open(f'lons.pickle', 'rb')
lons = pickle.load(pickle_in)
pickle_in.close()

ensemble_firs_train_predictions_2081_2098 = np.mean(pre_train_predictions[:,:,2081-1850:2098-1850+1,:,:], axis=(0,2))
taken_out_simulation_2081_2098 = np.mean(take_out_simulation[:,2081-1850:2098-1850+1,:,:], axis=1)
ensemble_predictions_after_tl_2081_2098 = np.mean(predictions_after_tl[:,:,2081-1850:2098-1850+1,:,:], axis=(0,2))

""" Plot """
size_suptitlefig = 42
size_titlefig = 37
size_title_axes = 31
size_lat_lon_coords = 24
size_colorbar_labels = 32
size_colorbar_ticks = 29
colormap = 'seismic'

plt.rcParams['font.sans-serif'] = 'Helvetica'

fig = plt.figure(figsize=(30,27))
gs = fig.add_gridspec(2, 1, height_ratios=[1,0.05], hspace=0.05)

gs0 = gs[0].subgridspec(3,1, hspace=0.55)

for idx_short_scenario, short_scenario in enumerate(short_scenarios_list):
    scenario = f'SSP{short_scenario[-3]}-{short_scenario[-2]}.{short_scenario[-1]}'
    
    gs0_sub = gs0[idx_short_scenario].subgridspec(1,2, wspace=-0.1)
    ax_ghost = fig.add_subplot(gs0_sub[:])
    ax_ghost.axis('off')
    ax_ghost.set_title(f'{scenario}', loc='center', size=size_titlefig, y=1.2)
    
    """ ax 0 """
    ax0 = fig.add_subplot(gs0_sub[0,0], projection=ccrs.Robinson())
    difference_min = np.concatenate((ensemble_firs_train_predictions_2081_2098[idx_short_scenario,:,:]-taken_out_simulation_2081_2098[idx_short_scenario,:,:],
                                    ensemble_predictions_after_tl_2081_2098[idx_short_scenario,:,:]-taken_out_simulation_2081_2098[idx_short_scenario,:,:])).min()
    difference_max = np.concatenate((ensemble_firs_train_predictions_2081_2098[idx_short_scenario,:,:]-taken_out_simulation_2081_2098[idx_short_scenario,:,:],
                                    ensemble_predictions_after_tl_2081_2098[idx_short_scenario,:,:]-taken_out_simulation_2081_2098[idx_short_scenario,:,:])).max()
    max_abs_difference_value = np.max([abs(difference_min), abs(difference_max)])
    difference_min_pretrained = np.concatenate((ensemble_firs_train_predictions_2081_2098[idx_short_scenario,:,:]-taken_out_simulation_2081_2098[idx_short_scenario,:,:])).min()
    difference_max_pretrained = np.concatenate((ensemble_firs_train_predictions_2081_2098[idx_short_scenario,:,:]-taken_out_simulation_2081_2098[idx_short_scenario,:,:])).max()
    difference_min_after_tl = np.concatenate((ensemble_predictions_after_tl_2081_2098[idx_short_scenario,:,:]-taken_out_simulation_2081_2098[idx_short_scenario,:,:])).min()
    difference_max_after_tl = np.concatenate((ensemble_predictions_after_tl_2081_2098[idx_short_scenario,:,:]-taken_out_simulation_2081_2098[idx_short_scenario,:,:])).max()

    n_levels = 40
    levels = np.linspace(difference_min, difference_max, n_levels)
    cbarticks_2 = [i for i in range(int(np.floor(difference_min)), int(np.floor(difference_max))+1)]
    vmin = -max_abs_difference_value-15
    vmax = max_abs_difference_value+15

    difference_warming_cyclic_data,lons_cyclic=add_cyclic_point(ensemble_firs_train_predictions_2081_2098[idx_short_scenario,:,:]-taken_out_simulation_2081_2098[idx_short_scenario,:,:],coord=lons)
    cs0=ax0.contourf(lons_cyclic,lats,difference_warming_cyclic_data, levels=levels,
                            transform = ccrs.PlateCarree(),
                            cmap=colormap, vmin=vmin, vmax=vmax)
    ax0.set_title(f'Difference (Pre-trained DNNs–{model_taken_out})', loc='center', size=size_title_axes, pad=17)
    ax0.coastlines()
    gl0 = ax0.gridlines(crs=ccrs.PlateCarree(), draw_labels={'bottom': 'x', 'left': 'y'}, linestyle='--', linewidth=0.1, color='black')
    gl0.xlabel_style = {'size': size_lat_lon_coords, 'color': 'black', 'weight': 'normal'}
    gl0.ylabel_style = {'size': size_lat_lon_coords, 'color': 'black', 'weight': 'normal'}

    """ ax 1 """
    ax1 = fig.add_subplot(gs0_sub[0,1], projection=ccrs.Robinson())
    difference_warming_cyclic_data,lons_cyclic=add_cyclic_point(ensemble_predictions_after_tl_2081_2098[idx_short_scenario,:,:]-taken_out_simulation_2081_2098[idx_short_scenario,:,:],coord=lons)
    cs1=ax1.contourf(lons_cyclic,lats,difference_warming_cyclic_data, levels=levels,
                            transform = ccrs.PlateCarree(),
                            cmap=colormap, vmin=vmin, vmax=vmax)
    ax1.coastlines()
    ax1.set_title(f'Difference (DNNs after TL–{model_taken_out})', loc='center', size=size_title_axes, pad=17)
    gl1 = ax1.gridlines(crs=ccrs.PlateCarree(), draw_labels={'bottom': 'x', 'left': 'y'}, linestyle='--', linewidth=0.1, color='black')
    gl1.xlabel_style = {'size': size_lat_lon_coords, 'color': 'black', 'weight': 'normal'}
    gl1.ylabel_style = {'size': size_lat_lon_coords, 'color': 'black', 'weight': 'normal'}

    plt.text(x=0.13, y=0.9, s='A', fontweight='bold',
            fontsize=size_suptitlefig, ha="center", transform=fig.transFigure)
    plt.text(x=0.87, y=0.9, s='D', fontweight='bold',
            fontsize=size_suptitlefig, ha="center", transform=fig.transFigure)
    plt.text(x=0.13, y=0.63, s='B', fontweight='bold',
            fontsize=size_suptitlefig, ha="center", transform=fig.transFigure)
    plt.text(x=0.87, y=0.63, s='E', fontweight='bold',
            fontsize=size_suptitlefig, ha="center", transform=fig.transFigure)
    plt.text(x=0.13, y=0.35, s='C', fontweight='bold',
            fontsize=size_suptitlefig, ha="center", transform=fig.transFigure)
    plt.text(x=0.87, y=0.35, s='F', fontweight='bold',
            fontsize=size_suptitlefig, ha="center", transform=fig.transFigure)

axcbar = fig.add_subplot(gs[1])
axcbar.axis('off')
cbar5 = fig.colorbar(cs1, ax=axcbar, orientation='horizontal', ticks=cbarticks_2, fraction=0.5)
cbar5.set_label(label='Surface Air Temperature difference [°C]', size=size_colorbar_labels, labelpad=20)
for l in cbar5.ax.xaxis.get_ticklabels():
    l.set_size(size_colorbar_ticks)

plt.savefig(f'Fig_S4_{model_taken_out}.png', bbox_inches='tight', dpi=300)
plt.close()