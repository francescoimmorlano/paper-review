"""
Author: Francesco Immorlano

Script for reproducing Figure 4
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

# Years reserved for validation during Transfer Learning on observations
val_years = [1985, 1995, 2005, 2015]

''' Dictionary that associates a well-known temperature biase with the CMIP6 ESM that is the worse wrt that bias
    according to: https://doi.org/10.1029/2022GL100888 '''
models_bias_dict = {
    'NA-cold': ['FGOALS-f3-L', 'North Atlantic - Gulf Stream (cold)'],
    'NEP-warm': ['MRI-ESM2-0', 'North East Pacific (warm)'],
    'NWP-cold': ['MIROC6', 'North West Pacific (cold)'],
    'SEA-warm': ['BCC-CSM2-MR', 'South East Atlantic (warm)'],
    'SO-warm': ['MIROC6', 'Southern Ocean (warm)'],
    'CT-cold': ['BCC-CSM2-MR', 'Cold Tongue (cold)']

}

""" Load DNNs predictions """
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
BEST_maps = np.zeros((n_BEST_years, n_lats, n_lons))
BEST_maps[:,:,:] = nc_BEST_data['st'][:,:,:]
nc_BEST_data.close()

# Compute average surface air temperature maps across DNNs predictions and CMIP6 ESMs simulations
avg_predictions_maps = np.mean(predictions, axis=(0,1))
avg_simulations_maps = np.mean(simulations, axis=0)

# Compute average surface air temperature maps in 2081-2098
avg_predictions_maps_2081_2098 = np.mean(avg_predictions_maps[:,2081-1979:,:,:], axis=1)
avg_simulations_maps_2081_2098 = np.mean(avg_simulations_maps[:,2081-1850:,:,:], axis=1)

# Compute average BEST map in 1980-1990 that will be used as baseline
BEST_baseline_map_1980_1990 = np.mean(BEST_maps[1:1990-1979+1,:,:], axis=0)

# Compute avg warming maps in 2081-2098 wrt 1980-1990
prediction_warming = avg_predictions_maps_2081_2098 - BEST_baseline_map_1980_1990
simulation_warming = avg_simulations_maps_2081_2098 - BEST_baseline_map_1980_1990

# Compute validation years indices for CMIP6 simulations and DNNs predictions
selected_val_years_models_idx = []
selected_val_years_predictions_idx = []
for year in val_years:
    selected_val_years_models_idx.append(year-1850)
    selected_val_years_predictions_idx.append(year-1979)

# Compute avg DNNs predictions maps in validation years
selected_dnns_ensemble_maps = predictions[:,:,:,selected_val_years_predictions_idx,:,:]
avg_dnns_ensemble_maps = np.mean(selected_dnns_ensemble_maps, axis=(0,1))

# Compute CMIP6 maps in validation years
selected_models_ensemble_maps = simulations[:,:,selected_val_years_models_idx,:,:]
avg_models_ensemble_maps = np.mean(selected_models_ensemble_maps, axis=0)

# Compute avg obs maps in validation years
selected_BEST_maps = BEST_maps[selected_val_years_predictions_idx,:,:]
avg_obs_maps_1980_2020 = np.mean(selected_BEST_maps, axis=0)
avg_obs_maps_1980_2020_C = avg_obs_maps_1980_2020 - 273.15

# Compute DNNs-obs avg difference maps in validation years
dnns_ensemble_diff_obs_maps = avg_dnns_ensemble_maps[scenario_idx,:,:,:] - selected_BEST_maps[:,:,:]
dnns_ensemble_diff_obs_maps_avg = np.mean(dnns_ensemble_diff_obs_maps, axis=0)

# Compute CMIP6-obs avg difference maps in validation years
models_ensemble_diff_obs_maps = avg_models_ensemble_maps[scenario_idx,:,:,:] - selected_BEST_maps[:,:,:]
models_ensemble_diff_obs_maps_avg = np.mean(models_ensemble_diff_obs_maps, axis=0)     

min_diff = min(models_ensemble_diff_obs_maps.min(), dnns_ensemble_diff_obs_maps_avg.min())
max_diff = max(models_ensemble_diff_obs_maps.max(), dnns_ensemble_diff_obs_maps_avg.max())
max_abs_diff = max(abs(min_diff), abs(max_diff))

""" Plot """
plt.rcParams.update({'font.sans-serif': 'Arial'})
colormap = 'seismic'
scenario_idx = 0

size_suptitlefig = 44
size_titlefig = 39
size_title_axes = 34
size_lat_lon_coords = 24
size_colorbar_labels = 34
size_colorbar_ticks = 31

fig = plt.figure(figsize=(50,50))

gs = fig.add_gridspec(3, 1, height_ratios=[0.3,1,0.3])
gs0 = gs[0].subgridspec(1, 3)
gs1 = gs[1].subgridspec(2, 1, height_ratios=[1,0.1], hspace=-0.02)
gs12 = gs1[0].subgridspec(3, 2, wspace=0.08, hspace=0.1)
gs2 = gs[2].subgridspec(1, 4, wspace=0.1)

n_levels = 55
levels_1 = np.linspace(min_diff, max_diff, n_levels)
vmin_1 = -max_abs_diff-2
vmax_1 = max_abs_diff+2
cbarticks_1 = np.arange(-9,9,2)

""" Plots DNN esnsemble-obs difference averaged in validation years """
ax00 = fig.add_subplot(gs0[0,0], projection=ccrs.Robinson())
difference_warming_cyclic_data,lons_cyclic=add_cyclic_point(dnns_ensemble_diff_obs_maps_avg,coord=lons)
cs00=ax00.contourf(lons_cyclic,lats,difference_warming_cyclic_data, levels=levels_1,
                        transform = ccrs.PlateCarree(),
                        cmap=colormap, vmin=vmin_1, vmax=vmax_1)
ax00.set_title(f'Bias (DNNs ensemble–Obs)', loc='center', size=size_title_axes, pad=17)
ax00.coastlines()
gl2 = ax00.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linestyle='--', linewidth=0.1, color='black')
gl2.xlabels_top = False
gl2.ylabels_right = False
gl2.xlabel_style = {'size': size_lat_lon_coords, 'color': 'black', 'weight': 'normal'}
gl2.ylabel_style = {'size': size_lat_lon_coords, 'color': 'black', 'weight': 'normal'}
plt.draw()
for ea in gl2.ylabel_artists:
    right_label = ea.get_position()[0] > 0
    if right_label:
        ea.set_visible(False)

""" Plot CMIP6 ensemble-obs difference averaged in validation years """
ax01 = fig.add_subplot(gs0[0,1], projection=ccrs.Robinson())
difference_warming_cyclic_data,lons_cyclic=add_cyclic_point(models_ensemble_diff_obs_maps_avg,coord=lons)
cs01=ax01.contourf(lons_cyclic,lats,difference_warming_cyclic_data, levels=levels_1,
                        transform = ccrs.PlateCarree(),
                        cmap=colormap, vmin=vmin_1, vmax=vmax_1)
ax01.set_title(f'Bias (CMIP6 ensemble–Obs)', loc='center', size=size_title_axes, pad=17)
ax01.coastlines()
gl2 = ax01.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linestyle='--', linewidth=0.1, color='black')
gl2.xlabels_top = False
gl2.ylabels_right = False
gl2.xlabel_style = {'size': size_lat_lon_coords, 'color': 'black', 'weight': 'normal'}
gl2.ylabel_style = {'size': size_lat_lon_coords, 'color': 'black', 'weight': 'normal'}
plt.draw()
for ea in gl2.ylabel_artists:
    right_label = ea.get_position()[0] > 0
    if right_label:
        ea.set_visible(False)
cbar01 = fig.colorbar(cs00, shrink=0.7, aspect=40, ax=[ax00, ax01], orientation='horizontal', ticks=cbarticks_1, pad=0.1)
cbar01.set_label(label='Surface Air Temperature difference [°C]', size=size_colorbar_labels, labelpad=20)
for l in cbar01.ax.xaxis.get_ticklabels():
    l.set_size(size_colorbar_ticks)

""" Plot observative maps averaged in validation years """
cbar0_ticks = np.arange(-55, 35, 10)
ax0 = fig.add_subplot(gs0[0,2], projection=ccrs.Robinson())
observed_cyclic_data,lons_cyclic=add_cyclic_point(avg_obs_maps_1980_2020_C[:,:],coord=lons)
cs0=ax0.contourf(lons_cyclic,lats,observed_cyclic_data, levels=40,
                        transform = ccrs.PlateCarree(),
                        cmap=colormap, vmin=-70, vmax=70, legend=True)
ax0.set_title('Obs', loc='center', size=size_title_axes, pad=17)
ax0.coastlines()
gl2 = ax0.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linestyle='--', linewidth=0.1, color='black')
gl2.xlabels_top = False
gl2.ylabels_right = False
gl2.xlabel_style = {'size': size_lat_lon_coords, 'color': 'black', 'weight': 'normal'}
gl2.ylabel_style = {'size': size_lat_lon_coords, 'color': 'black', 'weight': 'normal'}
plt.draw()
for ea in gl2.ylabel_artists:
    right_label = ea.get_position()[0] > 0
    if right_label:
        ea.set_visible(False)
cbar0 = fig.colorbar(cs0, shrink=0.8, ax=ax0, orientation='horizontal', ticks=cbar0_ticks, pad=0.1)
cbar0.set_label(label='Surface Air Temperature [°C]', size=size_colorbar_labels, labelpad=20)
for l in cbar0.ax.xaxis.get_ticklabels():
    l.set_size(size_colorbar_ticks)


i = 0
j = 0

diff_min = 0
diff_max = 0

n_levels = 55
levels_2 = np.linspace(-13.02197265625, 15.717803955078125, n_levels)

vmin_2 = -15.717803955078125+5
vmax_2 = 15.717803955078125-5
cbarticks_2 = [-13, -10, -7, -4, -1, 1, 4, 7, 10, 13, 15]

""" Plot DNNs-obs and CMIP6-obs difference for some well-known temperature biases averaged over validation years """
for index, key in enumerate(models_bias_dict):
    selected_models_idx = []
    selected_models_idx.append(models_list.index(models_bias_dict[key][0]))
    
    selected_models_array = simulations[selected_models_idx,:,:,:,:]
    selected_predictions_array = predictions[:,selected_models_idx,:,:,:,:]

    selected_models_array = selected_models_array[:,:,selected_val_years_models_idx,:,:]
    selected_predictions_array = selected_predictions_array[:,:,:,selected_val_years_predictions_idx,:,:]

    # Compute average surface air temperature maps across DNNs predictions and CMIP6 ESMs simulations
    avg_predictions_maps = np.mean(selected_predictions_array, axis=(0,1))
    avg_simulations_maps = np.mean(selected_models_array, axis=0)
    
    diff_predictions = avg_predictions_maps[scenario_idx,:,:,:]-selected_BEST_maps[:,:,:]
    diff_simulations = avg_simulations_maps[scenario_idx,:,:,:]-selected_BEST_maps[:,:,:]

    avg_diff_predictions = np.mean(diff_predictions, axis=0)
    avg_diff_simulations = np.mean(diff_simulations, axis=0)

    difference_DNN_min = avg_diff_predictions.min()
    difference_DNN_max = avg_diff_predictions.max()
    if difference_DNN_min < diff_min: diff_min = difference_DNN_min
    if difference_DNN_max > diff_max: diff_max = difference_DNN_max

    gs_local = gs12[i,j].subgridspec(1, 2, wspace=0.1)
    ax_ghost = fig.add_subplot(gs_local[:])
    ax_ghost.axis('off')
    ax_ghost.set_title(f'{models_bias_dict[key][1]}', loc='center', size=size_title_axes, pad=17)

    """ Plot DNNs-obs difference averaged over validation years """
    ax100 = fig.add_subplot(gs_local[0,0], projection=ccrs.Robinson())
    difference_warming_cyclic_data,lons_cyclic=add_cyclic_point(avg_diff_predictions,coord=lons)
    subscript = models_bias_dict[key][0][:2]
    cs100=ax100.contourf(lons_cyclic,lats,difference_warming_cyclic_data, levels=levels_2,
                            transform = ccrs.PlateCarree(),
                            cmap=colormap, vmin=vmin_2, vmax=vmax_2)
    ax100.set_title(f'Bias (DNN' + r'$_{{{}}}$'.format(subscript) + '–Obs)', loc='center', size=size_title_axes, pad=17)
    ax100.coastlines()
    gl2 = ax100.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linestyle='--', linewidth=0.1, color='black')
    gl2.xlabels_top = False
    gl2.ylabels_right = False
    gl2.xlabel_style = {'size': size_lat_lon_coords, 'color': 'black', 'weight': 'normal'}
    gl2.ylabel_style = {'size': size_lat_lon_coords, 'color': 'black', 'weight': 'normal'}
    plt.draw()
    for ea in gl2.ylabel_artists:
        right_label = ea.get_position()[0] > 0
        if right_label:
            ea.set_visible(False)

    """ Plot CMIP6-obs difference over averaged over validation years """
    ax101 = fig.add_subplot(gs_local[0,1], projection=ccrs.Robinson())
    difference_simulations_min = avg_diff_simulations.min()
    difference_simulations_max = avg_diff_simulations.max()
    if difference_simulations_min < diff_min: diff_min = difference_simulations_min
    if difference_simulations_max > diff_max: diff_max = difference_simulations_max
    difference_warming_cyclic_data,lons_cyclic=add_cyclic_point(avg_diff_simulations,coord=lons)
    cs101=ax101.contourf(lons_cyclic,lats,difference_warming_cyclic_data, levels=levels_2,
                            transform = ccrs.PlateCarree(),
                            cmap=colormap, vmin=vmin_2, vmax=vmax_2)
    ax101.set_title(f'Bias ({models_bias_dict[key][0]}–Obs)', loc='center', size=size_title_axes, pad=17)
    ax101.coastlines()
    gl1 = ax101.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linestyle='--', linewidth=0.1, color='black')
    gl1.xlabels_top = False
    gl1.ylabels_right = False
    gl1.xlabel_style = {'size': size_lat_lon_coords, 'color': 'black', 'weight': 'normal'}
    gl1.ylabel_style = {'size': size_lat_lon_coords, 'color': 'black', 'weight': 'normal'}
    plt.draw()
    for ea in gl1.ylabel_artists:
        right_label = ea.get_position()[0] > 0
        if right_label:
            ea.set_visible(False)

    if (index == 1) or (index == 3):
        j = 0
        i += 1
    else:
        j += 1

axcbar = fig.add_subplot(gs1[1])
axcbar.axis('off')
cbar5 = fig.colorbar(cs100, ax=axcbar, orientation='horizontal', aspect=35, fraction=0.8, shrink=0.6, ticks=cbarticks_2)
cbar5.set_label(label='Surface Air Temperature difference [°C]', size=size_colorbar_labels, labelpad=20)
for l in cbar5.ax.xaxis.get_ticklabels():
    l.set_family('Arial')
    l.set_size(size_colorbar_ticks)

""" DNNs ensemble-obs difference for each validation year """
diff_min_4 = (avg_dnns_ensemble_maps[scenario_idx,:,:,:]-selected_BEST_maps[:,:,:]).min()
diff_max_4 = (avg_dnns_ensemble_maps[scenario_idx,:,:,:]-selected_BEST_maps[:,:,:]).max()
abs_diff_max_4 = np.max(np.abs([diff_min_4, diff_max_4]))

n_levels_4 = 50
levels_4 = np.linspace(diff_min_4, diff_max_4, n_levels_4)
cbarticks_4 = [-3.42, -2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 3.5]
vmin_4 = -abs_diff_max_4-3
vmax_4 = abs_diff_max_4+3

ax200 = fig.add_subplot(gs2[0,0], projection=ccrs.Robinson())
difference_warming_cyclic_data,lons_cyclic=add_cyclic_point(avg_dnns_ensemble_maps[scenario_idx,0,:,:]-selected_BEST_maps[0,:,:],coord=lons)
cs200=ax200.contourf(lons_cyclic,lats,difference_warming_cyclic_data, levels=levels_4,
                        transform = ccrs.PlateCarree(),
                        cmap=colormap, vmin=vmin_4, vmax=vmax_4)
ax200.set_title(f'Bias (DNNs ensemble–Obs) — 1985', loc='center', size=size_title_axes, pad=17)
ax200.coastlines()
gl200 = ax200.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linestyle='--', linewidth=0.1, color='black')
gl200.xlabels_top = False
gl200.ylabels_right = False
gl200.xlabel_style = {'size': size_lat_lon_coords, 'color': 'black', 'weight': 'normal'}
gl200.ylabel_style = {'size': size_lat_lon_coords, 'color': 'black', 'weight': 'normal'}
plt.draw()
for ea in gl200.ylabel_artists:
    right_label = ea.get_position()[0] > 0
    if right_label:
        ea.set_visible(False)

ax201 = fig.add_subplot(gs2[0,1], projection=ccrs.Robinson())
difference_warming_cyclic_data,lons_cyclic=add_cyclic_point(avg_dnns_ensemble_maps[scenario_idx,1,:,:]-selected_BEST_maps[1,:,:],coord=lons)
cs201=ax201.contourf(lons_cyclic,lats,difference_warming_cyclic_data, levels=levels_4,
                        transform = ccrs.PlateCarree(),
                        cmap=colormap, vmin=vmin_4, vmax=vmax_4)
ax201.set_title(f'Bias (DNNs ensemble–Obs) — 1995', loc='center', size=size_title_axes, pad=17)
ax201.coastlines()
gl201 = ax201.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linestyle='--', linewidth=0.1, color='black')
gl201.xlabels_top = False
gl201.ylabels_right = False
gl201.xlabel_style = {'size': size_lat_lon_coords, 'color': 'black', 'weight': 'normal'}
gl201.ylabel_style = {'size': size_lat_lon_coords, 'color': 'black', 'weight': 'normal'}
plt.draw()
for ea in gl201.ylabel_artists:
    right_label = ea.get_position()[0] > 0
    if right_label:
        ea.set_visible(False)


ax202 = fig.add_subplot(gs2[0,2], projection=ccrs.Robinson())
difference_warming_cyclic_data,lons_cyclic=add_cyclic_point(avg_dnns_ensemble_maps[scenario_idx,2,:,:]-selected_BEST_maps[2,:,:],coord=lons)
cs202=ax202.contourf(lons_cyclic,lats,difference_warming_cyclic_data, levels=levels_4,
                        transform = ccrs.PlateCarree(),
                        cmap=colormap, vmin=vmin_4, vmax=vmax_4)
ax202.set_title(f'Bias (DNNs ensemble–Obs) — 2005', loc='center', size=size_title_axes, pad=17)
ax202.coastlines()
gl202 = ax202.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linestyle='--', linewidth=0.1, color='black')
gl202.xlabels_top = False
gl202.ylabels_right = False
gl202.xlabel_style = {'size': size_lat_lon_coords, 'color': 'black', 'weight': 'normal'}
gl202.ylabel_style = {'size': size_lat_lon_coords, 'color': 'black', 'weight': 'normal'}
plt.draw()
for ea in gl202.ylabel_artists:
    right_label = ea.get_position()[0] > 0
    if right_label:
        ea.set_visible(False)


ax203 = fig.add_subplot(gs2[0,3], projection=ccrs.Robinson())
difference_warming_cyclic_data,lons_cyclic=add_cyclic_point(avg_dnns_ensemble_maps[scenario_idx,3,:,:]-selected_BEST_maps[3,:,:],coord=lons)
cs203=ax203.contourf(lons_cyclic,lats,difference_warming_cyclic_data, levels=levels_4,
                        transform = ccrs.PlateCarree(),
                        cmap=colormap, vmin=vmin_4, vmax=vmax_4)
ax203.set_title(f'Bias (DNNs ensemble–Obs) — 2015', loc='center', size=size_title_axes, pad=17)
ax203.coastlines()
gl203 = ax203.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linestyle='--', linewidth=0.1, color='black')
gl203.xlabels_top = False
gl203.ylabels_right = False
gl203.xlabel_style = {'size': size_lat_lon_coords, 'color': 'black', 'weight': 'normal'}
gl203.ylabel_style = {'size': size_lat_lon_coords, 'color': 'black', 'weight': 'normal'}
plt.draw()
for ea in gl203.ylabel_artists:
    right_label = ea.get_position()[0] > 0
    if right_label:
        ea.set_visible(False)

cbar4 = fig.colorbar(cs203, aspect=35, shrink=0.6, ax=[ax200, ax201, ax202, ax203], orientation='horizontal', ticks=cbarticks_4, pad=0.1)
cbar4.set_label(label='Surface Air Temperature difference [°C]', size=size_colorbar_labels, labelpad=20, family='Arial')
for l in cbar4.ax.xaxis.get_ticklabels():
    l.set_family('Arial')
    l.set_size(size_colorbar_ticks)

plt.savefig('Fig_4.png', dpi=300, bbox_inches='tight')
plt.close()