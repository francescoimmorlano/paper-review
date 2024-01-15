"""
Author: Francesco Immorlano

Script for reproducing Figure 1
"""

import os
from netCDF4 import Dataset
import numpy as np
import matplotlib.pyplot as plt
import pickle
from matplotlib.legend_handler import HandlerTuple
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

total_earth_area = 5.1009974e+14
with open('../area_cella.csv', newline='') as csvfile:
    area_cella = np.genfromtxt(csvfile, delimiter=',')

ROOT_DATA = '../Source_data'
SIMULATIONS_DIRECTORY = f'{ROOT_DATA}/CMIP6_data/near_surface_air_temperature/Annual_uniform_remapped'

start_year_training = 1850
end_year_training = 2022
n_training_years = end_year_training-start_year_training+1
start_year_test = end_year_training+1
end_year_test = 2098
n_test_years = end_year_test-start_year_test+1

""" 
shuffle_idx = '01' to shuffle_idx = '22'
Set shuffle_idx = '09' to reproduce Figure 1 present in the paper
"""
shuffle_idx = '09'

models_list_take_out = models_list.copy()

model_take_out = models_list[int(shuffle_idx)-1]

# Delete the current take out model (i.e., the model which other models are transfer learned on) from the list 
models_list_take_out.remove(model_take_out)

print(f'\nModel taken out: {model_take_out} - shuffle: {shuffle_idx}')

""" Load predictions made by the DNNs after transfer learning on the take-out simulation """
predictions = np.zeros((len(models_list_take_out), len(short_scenarios_list), n_training_years+n_test_years, 64, 128))
pickle_in = open(f'{ROOT_DATA}/Transfer_Learning_on_Simulations/Predictions_shuffle-{shuffle_idx}.pickle', 'rb')
predictions = pickle.load(pickle_in)
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

# Convert from K to Celsius degrees
predictions_C = predictions - 273.15
simulations_C = take_out_simulation - 273.15

# Get average temperature maps in 2081-2098 simulated by the take-out model for SSP2-4.5
simulations_2081_2098_ssp245 = simulations_C[0,2081-1850:2098-1850+1,:,:]
avg_simulations_2081_2098_ssp245 = simulations_2081_2098_ssp245.mean(axis=0)

# Get average temperature maps in 2081-2098 predicted by the DNNs for SSP2-4.5
predictions_2081_2098_ssp245 = predictions_C[:,0,2081-1850:2098-1850+1,:,:]
avg_predictions_2081_2098_ssp245 = predictions_2081_2098_ssp245.mean(axis=(0,1))

# Compute average global surface air temperature
annual_predictions_means = ((predictions_C * area_cella).sum(axis=(-1,-2)))/total_earth_area
annual_simulations_means = ((simulations_C * area_cella).sum(axis=(-1,-2)))/total_earth_area

# Compute average temperature simulated by the take-out model in 1850-1900
global_mean_temp_taken_out = np.mean(annual_simulations_means[:,:1900-1850], axis=1)

# Compute warming wrt pre-industrial period
for idx_short_scenario, short_scenario in enumerate(short_scenarios_list):
    annual_predictions_means[:,idx_short_scenario,:] -= global_mean_temp_taken_out[idx_short_scenario]
    annual_simulations_means[idx_short_scenario,:] -= global_mean_temp_taken_out[idx_short_scenario]

# Compute average across DNNs predictions
ensemble_predictions_means = np.mean(annual_predictions_means, axis=0)

""" Compute 5-95% for temperatures predicted by the DNNs in 1850-2098 """
q05_predictions = np.zeros((len(short_scenarios_list),249))
q95_predictions = np.zeros((len(short_scenarios_list),249))
for short_scenario_idx, short_scenario in enumerate(short_scenarios_list):
    for i in range(2098-1850+1):
        q05_predictions[short_scenario_idx,i] = np.percentile(annual_predictions_means[:,short_scenario_idx,i],5)
        q95_predictions[short_scenario_idx,i] = np.percentile(annual_predictions_means[:,short_scenario_idx,i],95)

""" Compute RMSE """
difference_means = annual_predictions_means - annual_simulations_means
squared_diff = difference_means[:,:,2023-1850:2098-1850+1] ** 2
ms_diff = np.mean(squared_diff, axis=0)
rms_years = np.sqrt(ms_diff)
rmse_scenario = np.mean(rms_years, axis=1)

""" Compute bias map for SSP2-4.5 """
bias_avg_2081_2098_ssp245_map = avg_predictions_2081_2098_ssp245 - avg_simulations_2081_2098_ssp245
bias_avg_map_min = bias_avg_2081_2098_ssp245_map.min()
bias_avg_map_max = bias_avg_2081_2098_ssp245_map.max()

""" Plot """
font = {'fontname':'Arial'}
font2 = {'family': 'Arial',
         'weight': 'bold'}
colormap = 'seismic'

size_suptitlefig = 45
size_titlefig = 46
size_annotatation_letters = 43
size_title_scenario_axes = 35
size_title_map_axes = 35
size_temp_annotation = 31
size_x_y_ticks = 32
size_plot_legend = 27
size_x_y_legend = 39
size_colorbar_labels = 32
size_colorbar_ticks = 29

fig = plt.figure(figsize=(42,35))
plt.rcParams.update({'font.sans-serif': 'Arial'})

gs0 = fig.add_gridspec(1, 2, wspace=0.1, width_ratios=[1.7, 1])
gs00 = gs0[0].subgridspec(3, 1, hspace=0.5)
gs01 = gs0[1].subgridspec(3, 1, hspace=0.5)

""" ax1 """
ax1 = fig.add_subplot(gs00[0, 0])
for model_idx, model in enumerate(models_list_take_out):
    p11, = ax1.plot(np.arange(1850, 2099), annual_predictions_means[model_idx, 0], linewidth=1.5, linestyle='--', label=f'{model}')
p12, = ax1.plot(np.arange(1850, 2099), annual_simulations_means[0,:], linewidth=5, label=f'{model_take_out} (model taken out)')
p13, = ax1.plot(np.arange(1850, 2099), ensemble_predictions_means[0,:], linewidth=5, label=f'Ensemble')
# Train set shading
ax1.fill_between(np.arange(start_year_training, end_year_training+1), -2, ensemble_predictions_means[0,:end_year_training-start_year_training+1], color='red', alpha=0.1)
# Predictions 5-95% uncertainty shading
ax1.fill_between(np.arange(1850, 2099), q05_predictions[0,:], q95_predictions[0,:], zorder=0, facecolor='#A8FFBA')
ax1.set_xticks([1850, 1900, 1950, 2000, end_year_training, 2050, 2098])
plt.xticks(fontsize=size_x_y_ticks)
plt.yticks(fontsize=size_x_y_ticks)
ax1.set_ylim([-1, np.ceil(np.max(annual_predictions_means[:,0]))])
ax1.set_title(f'SSP2-4.5\nRMSE (2023–2098): {round(rmse_scenario[0],2)}°C — Temperature in 2098: {np.round(ensemble_predictions_means[0,-1],2)} °C [{np.round(q05_predictions[0,-1],2)}–{np.round(q95_predictions[0,-1],2)} °C]', size=size_title_scenario_axes,
                pad=30, linespacing=1.5)
l1 = ax1.legend([p11, p12, p13], ['DNNs', 'FGOALS-f3-L', 'DNNs average'],
               handler_map={tuple: HandlerTuple(ndivide=None)}, prop={"family":"Arial",'size':size_plot_legend}, loc='upper left')

""" ax2 """
ax2 = fig.add_subplot(gs00[1, 0])
for model_idx, model in enumerate(models_list_take_out):
    p21, = ax2.plot(np.arange(1850, 2099), annual_predictions_means[model_idx, 1], linewidth=1.5, linestyle='--', label=f'{model}')
p22, = ax2.plot(np.arange(1850, 2099), annual_simulations_means[1,:], linewidth=5, label=f'{model_take_out} (model taken out)')
p23, = ax2.plot(np.arange(1850, 2099), ensemble_predictions_means[1,:], linewidth=5, label=f'Ensemble')
ax2.fill_between(np.arange(start_year_training, end_year_training+1), -2, ensemble_predictions_means[1,:end_year_training-start_year_training+1], color='red', alpha=0.1)
# Predictions 5-95% uncertainty shading
ax2.fill_between(np.arange(1850, 2099), q05_predictions[1,:], q95_predictions[1,:], zorder=0, facecolor='#A8FFBA')
ax2.set_xticks([1850, 1900, 1950, 2000, end_year_training, 2050, 2098])
plt.xticks(fontsize=size_x_y_ticks)
plt.yticks(fontsize=size_x_y_ticks)
ax2.set_ylim([-1, np.ceil(np.max(annual_predictions_means[:,1]))])
ax2.set_title(f'SSP3-7.0\nRMSE (2023–2098): {round(rmse_scenario[1],2)} °C — Temperature in 2098: {np.round(ensemble_predictions_means[1,-1],2)} °C [{np.round(q05_predictions[1,-1],2)}–{np.round(q95_predictions[1,-1],2)} °C]',
              size=size_title_scenario_axes, linespacing=1.5, pad=30)
plt.ylabel('Near surface air temperature anomaly [°C]\nBase period: 1850–1900', fontsize=size_x_y_legend, linespacing=1.5,labelpad=80)
l2 = ax2.legend([p11, p12, p13], ['DNNs', 'FGOALS-f3-L', 'DNNs average'],
               handler_map={tuple: HandlerTuple(ndivide=None)}, prop={"family":"Arial",'size':size_plot_legend}, loc='upper left')

""" ax3 """
ax3 = fig.add_subplot(gs00[2, 0])
for model_idx, model in enumerate(models_list_take_out):
    p31, = ax3.plot(np.arange(1850, 2099), annual_predictions_means[model_idx, 2], linewidth=1.5, linestyle='--', label=f'{model}')
p32, = ax3.plot(np.arange(1850, 2099), annual_simulations_means[2,:], linewidth=5, label=f'{model_take_out} (model taken out)')
p33, = ax3.plot(np.arange(1850, 2099), ensemble_predictions_means[2,:], linewidth=5, label=f'Ensemble')
ax3.fill_between(np.arange(start_year_training, end_year_training+1), -2, ensemble_predictions_means[2,:end_year_training-start_year_training+1], color='red', alpha=0.1)
ax3.fill_between(np.arange(1850, 2099), q05_predictions[2,:], q95_predictions[2,:], zorder=0, facecolor='#A8FFBA')
ax3.set_xticks([1850, 1900, 1950, 2000, end_year_training, 2050, 2098])
plt.xticks(fontsize=size_x_y_ticks)
plt.yticks(fontsize=size_x_y_ticks)
ax3.set_ylim([-1, np.ceil(np.max(annual_predictions_means[:,2]))])
ax3.set_title(f'SSP5-8.5\nRMSE (2023–2098): {round(rmse_scenario[2],2)} °C — Temperature in 2098: {np.round(ensemble_predictions_means[2,-1],2)} °C [{np.round(q05_predictions[2,-1],2)}–{np.round(q95_predictions[2,-1],2)} °C]',
              size=size_title_scenario_axes, linespacing=1.5, pad=30)
plt.xlabel('Years', fontsize=size_x_y_legend, labelpad=40)
l = ax3.legend([p11, p12, p13], ['DNNs', 'FGOALS-f3-L', 'DNNs average'],
               handler_map={tuple: HandlerTuple(ndivide=None)}, prop={"family":"Arial",'size':size_plot_legend}, loc='upper left')
min_value = np.concatenate((avg_predictions_2081_2098_ssp245, avg_simulations_2081_2098_ssp245)).min()
max_value = np.concatenate((avg_predictions_2081_2098_ssp245, avg_simulations_2081_2098_ssp245)).max()
levels = np.linspace(min_value, max_value, 30)

""" ax4 """
ax4 = fig.add_subplot(gs01[0, 0], projection=ccrs.Robinson())
data = avg_simulations_2081_2098_ssp245
data_cyclic, lons_cyclic = add_cyclic_point(data, lons)
cs=ax4.contourf(lons_cyclic, lats, data_cyclic,
                levels=40, vmin=-90, vmax=90,
                transform = ccrs.PlateCarree(),cmap=colormap)
ax4.coastlines()
gl4 = ax4.gridlines(draw_labels=True, linestyle='--')
gl4.top_labels = False
gl4.right_labels = False
gl4.xlabel_style = {'size': size_x_y_ticks, 'color': 'k', 'rotation':0}
gl4.ylabel_style = {'size': size_x_y_ticks, 'color': 'k', 'rotation':0}
cbar4 = plt.colorbar(cs,shrink=0.8, orientation='horizontal', pad=0.12)
cbar4.set_label('Surface Air Temperature [°C]',size=size_colorbar_labels,rotation='horizontal', labelpad=15)
for l in cbar4.ax.xaxis.get_ticklabels():
    l.set_family('Arial')
    l.set_size(size_colorbar_ticks)
ax4.set_title(f'{model_take_out} average temperature (2081–2098)', size=size_title_map_axes, pad=17)

""" ax5 """
ax5 = fig.add_subplot(gs01[1, 0], projection=ccrs.Robinson())
data = avg_predictions_2081_2098_ssp245
data_cyclic, lons_cyclic = add_cyclic_point(data, lons)
cs = ax5.contourf(lons_cyclic, lats, data_cyclic,
                levels=40, vmin=-90, vmax=90,
                transform = ccrs.PlateCarree(),cmap=colormap)
ax5.coastlines()
gl5 = ax5.gridlines(draw_labels=True, linestyle='--')
gl5.top_labels = False
gl5.right_labels = False
gl5.xlabel_style = {'size': size_x_y_ticks, 'color': 'k', 'rotation':0}
gl5.ylabel_style = {'size': size_x_y_ticks, 'color': 'k', 'rotation':0}
cbar5 = plt.colorbar(cs,shrink=0.8, orientation='horizontal', pad=0.12)
cbar5.set_label('Surface Air Temperature [°C]',size=size_colorbar_labels, rotation='horizontal', labelpad=15)
for l in cbar5.ax.xaxis.get_ticklabels():
    l.set_size(size_colorbar_ticks)
ax5.set_title(f'Ensemble DNNs average temperature (2081–2098)', size=size_title_map_axes, pad=17, **font)

""" ax6 """
ax6 = fig.add_subplot(gs01[2, 0], projection=ccrs.Robinson())
data=bias_avg_2081_2098_ssp245_map
data_cyclic, lons_cyclic = add_cyclic_point(data, lons)
levels_3 = np.linspace(-bias_avg_map_max, bias_avg_map_max, 40)
cs=ax6.contourf(lons_cyclic, lats, data_cyclic, levels=40, vmin=-bias_avg_map_max-6, vmax=bias_avg_map_max+6,
            transform = ccrs.PlateCarree(),cmap=colormap)
ax6.coastlines()
gl6 = ax6.gridlines(draw_labels=True, linestyle='--')
gl6.top_labels = False
gl6.right_labels = False
gl6.xlabel_style = {'size': size_x_y_ticks, 'color': 'k', 'rotation':0}
gl6.ylabel_style = {'size': size_x_y_ticks, 'color': 'k', 'rotation':0}
cbarticks_6 = [-4,-3,-2,-1,0,1,2,3,4]
cbar = plt.colorbar(cs,shrink=0.8, ticks=cbarticks_6, orientation='horizontal', pad=0.12)
cbar.set_label('Surface Air Temperature [°C]',size=size_colorbar_labels, family='Arial', rotation='horizontal', labelpad=15)
for l in cbar.ax.xaxis.get_ticklabels():
    l.set_family("Arial")
    l.set_size(size_colorbar_ticks)
ax6.set_title(f'Bias (DNNs - CMIP6)', size=size_title_map_axes, pad=17)

plt.draw()
for ea in gl4.ylabel_artists:
    right_label = ea.get_position()[0] > 0
    if right_label:
        ea.set_visible(False)
for ea in gl5.ylabel_artists:
    right_label = ea.get_position()[0] > 0
    if right_label:
        ea.set_visible(False)
for ea in gl6.ylabel_artists:
    right_label = ea.get_position()[0] > 0
    if right_label:
        ea.set_visible(False)

plt.text(0.1, 0.92, "a", transform=fig.transFigure, fontsize=size_annotatation_letters)
plt.text(0.62, 0.92, "b", transform=fig.transFigure, fontsize=size_annotatation_letters)
plt.text(0.7, 0.92, "Scenario SSP2-4.5", transform=fig.transFigure, fontsize=size_annotatation_letters-3)
plt.text(0.62, 0.63, "c", transform=fig.transFigure, fontsize=size_annotatation_letters)
plt.text(0.62, 0.33, "d", transform=fig.transFigure, fontsize=size_annotatation_letters)
plt.savefig(f'Fig_1_{model_take_out}.png', dpi=300, bbox_inches='tight')
plt.close()