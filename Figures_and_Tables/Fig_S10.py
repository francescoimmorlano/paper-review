"""
Author: Francesco Immorlano

Script for reproducing Figure S10
"""

from cartopy.util import add_cyclic_point
import cartopy.crs as ccrs
import sys
sys.path.insert(1, './..')
from lib import *

""" Load predictions made by the DNNs after transfer learning on observational data """
predictions = read_tl_obs_predictions(n_BEST_datasets_per_model_scenario, compute_figures_tables)

""" Load CMIP6 ESMs simulations """
simulations = read_all_cmip6_simulations()

""" Load BEST observational data """
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
axs[2].set_title('Difference (DNNs-CMIP6)', loc='center', size=size_title_axes, pad=17)
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

plt.text(x=0.12, y=0.88, s='A', fontweight='bold',
         fontsize=size_suptitlefig, ha="center", transform=fig.transFigure)
plt.text(x=0.385, y=0.88, s='B', fontweight='bold',
         fontsize=size_suptitlefig, ha="center", transform=fig.transFigure)
plt.text(x=0.65, y=0.88, s='C', fontweight='bold',
         fontsize=size_suptitlefig, ha="center", transform=fig.transFigure)

plt.savefig(f'Fig_S10.png', bbox_inches = 'tight', dpi=300)
plt.close()