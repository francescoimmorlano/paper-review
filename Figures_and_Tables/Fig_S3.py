"""
Author: Francesco Immorlano

Script for reproducing images used in Figure S3
"""

from cartopy.util import add_cyclic_point
import cartopy.crs as ccrs
import sys
sys.path.insert(1, '..')
from lib import *

""" Load DNNs predictions after pre-training """
predictions = read_first_train_predictions(compute_figures_tables)

""" Load CMIP6 simulations """
simulations = read_all_cmip6_simulations()

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

    gl0 = axs[idx_short_scenario].gridlines(crs=ccrs.PlateCarree(), draw_labels={"bottom": "x", "left": "y"}, linestyle='--', linewidth=0.05, color='black')
    gl0.xlabel_style = {'size': size_lat_lon_coords, 'color': 'black', 'weight': 'normal'}
    gl0.ylabel_style = {'size': size_lat_lon_coords, 'color': 'black', 'weight': 'normal'}

    plt.text(x=0.1, y=0.89, s=f'A', fontweight='bold',
            fontsize=size_suptitlefig, ha="center", transform=fig.transFigure)
    plt.text(x=0.37, y=0.89, s=f'B', fontweight='bold',
            fontsize=size_suptitlefig, ha="center", transform=fig.transFigure)
    plt.text(x=0.64, y=0.89, s=f'C', fontweight='bold',
            fontsize=size_suptitlefig, ha="center", transform=fig.transFigure)
    
cbar0 = fig.colorbar(cs0, shrink=0.6, ax=axs, orientation='horizontal', pad=0.15, ticks=cbarticks_2, aspect=30)
cbar0.set_label(label='Surface Air Temperature difference [°C]', size=size_colorbar_labels, labelpad=30)
for l in cbar0.ax.xaxis.get_ticklabels():
    l.set_size(size_colorbar_ticks)

plt.savefig('Fig_S3.png', bbox_inches = 'tight', dpi=300)
plt.close()
