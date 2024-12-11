"""
Author: Francesco Immorlano

Script for reproducing images used in Figure S3
"""

from cartopy.util import add_cyclic_point
import cartopy.crs as ccrs
import sys
sys.path.insert(1, '..')
from lib import *

val_years = [i for i in range(start_year_first_training_val, end_year_first_training_val+1)]

# Load DNNs predictions after pre-training
predictions = read_first_train_predictions(compute_figures_tables_paper, 'First_Training_')

# Load CMIP6 simulations
simulations = read_all_cmip6_simulations()

pickle_in = open('lons.pickle', 'rb')
lons = pickle.load(pickle_in)
pickle_in.close()

pickle_in = open('lats.pickle', 'rb')
lats = pickle.load(pickle_in)
pickle_in.close()

# Compute DNNs ensemble and CMIP6 ensemble in val years
ensemble_predictions_val = np.mean(predictions[:,:,val_years[0]-1850:val_years[-1]-1850+1,:,:], axis=0)
ensemble_simulations_val = np.mean(simulations[:,:,val_years[0]-1850:val_years[-1]-1850+1,:,:], axis=0)

difference = predictions[:,:,val_years[0]-1850:val_years[-1]-1850+1,:,:]-simulations[:,:,val_years[0]-1850:val_years[-1]-1850+1,:,:]
ensemble_difference = np.mean(difference[:,:,:,:,:], axis=0)

size_suptitlefig = 46
size_titlefig = 37
size_title_axes = 32
size_lat_lon_coords = 35
size_colorbar_labels = 40
size_colorbar_ticks = 35
colormap = 'seismic'

diff_avg_years = np.mean(ensemble_difference[:,:,:,:], axis=1)

""" Plot """
difference_min = np.min(diff_avg_years)
difference_max = np.max(diff_avg_years)
max_abs_difference_value = np.max([abs(difference_min), abs(difference_max)])
vmin = -max_abs_difference_value-2
vmax = max_abs_difference_value+2

n_levels = 45
levels = np.linspace(difference_min, difference_max, n_levels)

fig, axs = plt.subplots(nrows=1,ncols=3,
                        subplot_kw={'projection': ccrs.Robinson(central_longitude=180)},
                        figsize=(40,10))
fig.subplots_adjust(top=0.97, wspace=0.08, hspace = 0.2)

axs=axs.flatten()

for idx_short_scenario, short_scenario in enumerate(short_scenarios_list):

        scenario = f'SSP{short_scenario[-3]}-{short_scenario[-2]}.{short_scenario[-1]}'

        difference_warming_cyclic_data,lons_cyclic=add_cyclic_point(diff_avg_years[idx_short_scenario,:,:],coord=lons)

        cs0=axs[idx_short_scenario].contourf(lons_cyclic,lats,difference_warming_cyclic_data, levels=levels,
                                                transform = ccrs.PlateCarree(),
                                                cmap=colormap, vmin=vmin, vmax=vmax)
        axs[idx_short_scenario].coastlines()
        axs[idx_short_scenario].set_title(f'{scenario}', fontsize=40, pad=10)

        gl0 = axs[idx_short_scenario].gridlines(crs=ccrs.PlateCarree(), draw_labels=False, linestyle='--', linewidth=0.05, color='black')
        gl0.xlabel_style = {'size': size_lat_lon_coords, 'color': 'black', 'weight': 'normal'}
        gl0.ylabel_style = {'size': size_lat_lon_coords, 'color': 'black', 'weight': 'normal'}

        plt.text(x=0.13, y=0.89, s=f'A', fontweight='bold',
                fontsize=size_suptitlefig, ha="center", transform=fig.transFigure)
        plt.text(x=0.395, y=0.89, s=f'B', fontweight='bold',
                fontsize=size_suptitlefig, ha="center", transform=fig.transFigure)
        plt.text(x=0.66, y=0.89, s=f'C', fontweight='bold',
                fontsize=size_suptitlefig, ha="center", transform=fig.transFigure)

cbarticks = [-0.44, -0.2, 0, 0.2, 0.4, 0.6, 0.73, 0.8, 0.9]
cbar0 = fig.colorbar(cs0, shrink=0.6, ax=axs, orientation='horizontal', pad=0.08, ticks=cbarticks, aspect=30)
cbar0.set_label(label='Surface Air Temperature difference [°C]', size=size_colorbar_labels, labelpad=30)
cbar0.ax.tick_params(labelsize=size_colorbar_ticks)

plt.savefig('Fig_S3.png', bbox_inches = 'tight', dpi=300, facecolor='white', transparent=False)
plt.close()