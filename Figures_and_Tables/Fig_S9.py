"""
Author: Francesco Immorlano

Script for reproducing Figure S9
"""

import cartopy.crs as ccrs
from cartopy.util import add_cyclic_point
import sys
sys.path.insert(1, './..')
from lib import *

""" Load DNNs predictions after TL on observations """
predictions = read_tl_obs_predictions(n_BEST_datasets_per_model_scenario, compute_figures_tables)

""" Load CMIP6 ESMs simulations"""
simulations = read_all_cmip6_simulations()

""" Loada BEST observational data """
BEST_data_array = read_BEST_data(PATH_BEST_DATA)

pickle_in = open('lons.pickle', 'rb')
lons = pickle.load(pickle_in)
pickle_in.close()

pickle_in = open('lats.pickle', 'rb')
lats = pickle.load(pickle_in)
pickle_in.close()

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
    ax1.set_title(f'DNNs ensemble', size=size_title_axes, pad=17)

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

    fig.suptitle(f'{scenario}', y=0.85, size=size_suptitlefig)
    plt.savefig(f'Fig_S9_{scenario_short}.png', bbox_inches = 'tight', dpi=300)
    plt.close()