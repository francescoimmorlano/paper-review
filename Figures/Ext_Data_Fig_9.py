"""
Author: Francesco Immorlano

Script for reproducing images used in Extended Data Figure 9
"""

from matplotlib import pyplot as plt 
import cartopy.crs as ccrs
from cartopy.util import add_cyclic_point
import cartopy.crs as ccrs
from netCDF4 import Dataset
import numpy as np 
import os
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

n_lat_points = 64
n_lon_points = 128
short_scenarios_list = ['ssp245', 'ssp370', 'ssp585']
variable_short = 'tas'

ROOT_DATA = '../Source_data'
SIMULATIONS_DIRECTORY = f'{ROOT_DATA}/CMIP6_data/near_surface_air_temperature/Annual_uniform_remapped'
PATH_BEST_DATA = f'{ROOT_DATA}/BEST_data/BEST_regridded_annual_1979-2022.nc'

with open('../area_cella.csv', newline='') as csvfile:
    area_cella = np.genfromtxt(csvfile, delimiter=',')

n_BEST_datasets_per_model_scenario = 5

start_year_training = 1979
end_year_training = 2022
n_training_years = end_year_training-start_year_training+1

start_year_test = end_year_training+1
end_year_test = 2098
n_test_years = end_year_test-start_year_test+1

""" Load DNNs predictions """
predictions = np.zeros((n_BEST_datasets_per_model_scenario, len(models_list), len(short_scenarios_list), n_training_years+n_test_years, n_lat_points, n_lon_points))
pickle_in = open(f'{ROOT_DATA}/Transfer_Learning_on_Observations/Transfer_learning_obs.pickle','rb')
predictions = pickle.load(pickle_in)

""" Load CMIP6 ESMs simulations"""
simulations = np.zeros((len(models_list), len(short_scenarios_list), 2098-1850+1, n_lat_points, n_lon_points))
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

nc_BEST_data = Dataset(PATH_BEST_DATA, mode='r+', format='NETCDF3_CLASSIC')

""" Loada BEST observational data """
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

area_cella_sum_over_lon = np.sum(area_cella, axis=1)

# Compute avg warming values across latitudes
predictions_means_over_lons = ((prediction_warming * area_cella).sum(axis=(2)))/area_cella_sum_over_lon
simulations_means_over_lons = ((simulation_warming * area_cella).sum(axis=(2)))/area_cella_sum_over_lon

""" Plot """
size_suptitlefig = 35
size_title_fig = 30
size_scenario_text = 27
size_subplots_letters = 40
size_title_axes = 28
size_line_plot_legend = 16
size_lat_lon_coords = 23
size_colorbar_labels = 30
size_colorbar_ticks = 26
plt.rcParams['font.sans-serif'] = 'Arial'

for scenario_idx, scenario_short in enumerate(short_scenarios_list):
    scenario = f'SSP{scenario_short[-3]}-{scenario_short[-2]}.{scenario_short[-1]}'

    min_value = np.concatenate((prediction_warming[scenario_idx,:,:], simulation_warming[scenario_idx,:,:])).min()
    max_value = np.concatenate((prediction_warming[scenario_idx,:,:], simulation_warming[scenario_idx,:,:])).max()

    max_abs_value = np.max([abs(min_value), abs(max_value)])
    levels = np.linspace(min_value, max_value, 40)

    if scenario_idx == 0:
        cbarticks = np.arange(-2,12,2)
    elif scenario_idx == 1:
        cbarticks = np.arange(-2,16,2)
    elif scenario_idx == 2:
        cbarticks = np.arange(-2,18,2)

    fig = plt.figure(constrained_layout=True, figsize=(30,10))
    ax = fig.add_gridspec(3,11)

    """ ax1 """
    ax1 = fig.add_subplot(ax[1:, :5], projection=ccrs.Robinson())

    data_1 = prediction_warming[scenario_idx]
    data_cyclic_1, lons_cyclic_1 = add_cyclic_point(data_1, lons)
    cs1=ax1.contourf(lons_cyclic_1, lats, data_cyclic_1,
                    vmin=-max_abs_value, vmax=max_abs_value, levels=levels, 
                    transform = ccrs.PlateCarree(),cmap='bwr')#, extend='both')
    ax1.coastlines()
    gl1 = ax1.gridlines(draw_labels=True, linestyle='--', color='black', linewidth=0.1)
    gl1.top_labels = False
    gl1.right_labels = False
    gl1.xlabel_style = {'size': size_lat_lon_coords, 'color': 'black', 'weight': 'normal'}
    gl1.ylabel_style = {'size': size_lat_lon_coords, 'color': 'black', 'weight': 'normal'}
    ax1.set_title(f'Deep Neural Networks ensemble', size=size_title_axes, pad=17)

    cbar1 = plt.colorbar(cs1,shrink=0.7, ticks=cbarticks, orientation='horizontal', label='Surface Air Temperature anomaly [°C]')
    cbar1.set_label(label='Surface Air Temperature anomaly [°C]', size=size_colorbar_labels, labelpad=20)
    for l in cbar1.ax.xaxis.get_ticklabels():
        l.set_size(size_colorbar_ticks)

    """ ax2 """
    ax2 = fig.add_subplot(ax[1:, 5])

    ax2.plot(predictions_means_over_lons[scenario_idx], lats, label='DNN')
    ax2.plot(simulations_means_over_lons[scenario_idx], lats, label='CMIP6')
    ax2.set_ylim(bottom=0, top=0)
    plt.yticks([-90,-60,-30,0,30,60,90], ['90°S','60°S', '30°S', '0', '30°N', '60°N', '90°N'], fontsize=size_lat_lon_coords)
    plt.xticks(fontsize=size_lat_lon_coords)
    ax2.tick_params(axis='both', which='major', labelsize=size_lat_lon_coords)
    ax2.legend(loc='lower right', prop={'size': size_line_plot_legend})

    """ ax3 """
    ax3 = fig.add_subplot(ax[1:, 6:], projection=ccrs.Robinson())
    data_3 = simulation_warming[scenario_idx]
    data_cyclic_3, lons_cyclic_3 = add_cyclic_point(data_3, lons)
    cs3=ax3.contourf(lons_cyclic_3, lats, data_cyclic_3,
                    vmin=-max_abs_value, vmax=max_abs_value, levels=levels,
                    transform = ccrs.PlateCarree(),cmap='bwr')
    ax3.coastlines()
    gl3 = ax3.gridlines(draw_labels=True, linestyle='--', color='black', linewidth=0.1)
    gl3.top_labels = False
    gl3.right_labels = False
    gl3.xlabel_style = {'size': size_lat_lon_coords, 'color': 'black', 'weight': 'normal'}
    gl3.ylabel_style = {'size': size_lat_lon_coords, 'color': 'black', 'weight': 'normal'}
    ax3.set_title(f'CMIP6 ensemble', size=size_title_axes, pad=17)
    cbar3 = plt.colorbar(cs3,shrink=0.7,ticks=cbarticks,orientation='horizontal')
    cbar3.set_label(label='Surface Air Temperature anomaly [°C]', size=size_colorbar_labels, labelpad=20)
    for l in cbar3.ax.xaxis.get_ticklabels():
        l.set_size(size_colorbar_ticks)

    plt.draw()
    for ea in gl1.ylabel_artists:
        right_label = ea.get_position()[0] > 0
        if right_label:
            ea.set_visible(False)
    for ea in gl3.ylabel_artists:
        right_label = ea.get_position()[0] > 0
        if right_label:
            ea.set_visible(False)

    fig.suptitle(f'{scenario}', y=0.85, size=size_suptitlefig)
    plt.savefig(f'Ext_Data_Fig_9_{scenario_short}.png', bbox_inches = 'tight', dpi=300)
    plt.close()