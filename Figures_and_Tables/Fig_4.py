"""
Author: Francesco Immorlano

Script for reproducing Figure 4
"""

import matplotlib.patches as mpatches
from cartopy.util import add_cyclic_point
import cartopy.crs as ccrs
import sys
sys.path.insert(1, '..')
from lib import *

# 0 for SSP2-4.5
idx_scenario = 0

# Load DNNs predictions
predictions = read_tl_obs_predictions(n_BEST_datasets_per_model_scenario, compute_figures_tables_paper, 'Transfer_learning_obs_')

# Load CMIP6 simulations
simulations = read_all_cmip6_simulations()

# Load BEST observational data
BEST_maps = read_BEST_data(PATH_BEST_DATA)

pickle_in = open('lons.pickle', 'rb')
lons = pickle.load(pickle_in)
pickle_in.close()

pickle_in = open('lats.pickle', 'rb')
lats = pickle.load(pickle_in)
pickle_in.close()

# Compute validation years indices for CMIP6 simulations and DNNs predictions
selected_val_years_models_idx = []
selected_val_years_predictions_idx = []
for year in val_years_list_tl_obs:
    selected_val_years_models_idx.append(year-1850)
    selected_val_years_predictions_idx.append(year-1979)

""" Plot """
colormap = 'seismic'

size_title_axes = 35
size_lat_lon_coords = 30
size_colorbar_labels = 31
size_colorbar_ticks = 30

fig = plt.figure(figsize=(45,25))

gs = fig.add_gridspec(2, 1, height_ratios=[1,1], hspace=-0.05)
gs1 = gs[0].subgridspec(1, 2, wspace=-0.05)
gs2 = gs[1].subgridspec(1, 4, wspace=0.08)

selected_ensemble_array = predictions[:,:,:,selected_val_years_predictions_idx,:,:]
avg_ensemble_maps = np.mean(selected_ensemble_array, axis=(0,1))

selected_models_ensemble_array = simulations[:,:,selected_val_years_models_idx,:,:]
avg_models_ensemble_maps = np.mean(selected_models_ensemble_array, axis=0)

selected_BEST_data_array = BEST_maps[selected_val_years_predictions_idx,:,:]

diff_ensemble_array = avg_ensemble_maps[idx_scenario,:,:,:] - selected_BEST_data_array[:,:,:]
diff_ensemble_array_avg = np.mean(diff_ensemble_array, axis=0)

diff_models_ensemble_array = avg_models_ensemble_maps[idx_scenario,:,:,:] - selected_BEST_data_array[:,:,:]
diff_models_ensemble_array_avg = np.mean(diff_models_ensemble_array, axis=0)

min_diff = min(diff_models_ensemble_array_avg.min(), diff_ensemble_array_avg.min())
max_diff = max(diff_models_ensemble_array_avg.max(), diff_ensemble_array_avg.max())
max_abs_diff = max(abs(min_diff), abs(max_diff))

n_levels = 55
levels_bias = np.linspace(min_diff, max_diff, n_levels)
vmin_bias = -max_abs_diff-7
vmax_bias = max_abs_diff+7

###############
#   DNNs-OBS  #
###############
""" Plots DNN ensemble-obs difference averaged in validation years """
ax100 = fig.add_subplot(gs1[0,0], projection=ccrs.Robinson())
difference_warming_cyclic_data,lons_cyclic=add_cyclic_point(diff_ensemble_array_avg,coord=lons)
cs100=ax100.contourf(lons_cyclic,lats,difference_warming_cyclic_data, levels=levels_bias,
                        transform = ccrs.PlateCarree(),
                        cmap=colormap, vmin=vmin_bias, vmax=vmax_bias)
ax100.set_title(f'Bias (DNNs ensemble–Obs)', loc='center', size=size_title_axes, pad=17)
ax100.coastlines()
gl_bias = ax100.gridlines(crs=ccrs.PlateCarree(), draw_labels=False, linestyle='--', linewidth=0.1, color='black', x_inline=False, y_inline=False, dms=True)
gl_bias.xlabel_style = {'size': size_lat_lon_coords, 'color': 'black', 'weight': 'normal'}
gl_bias.ylabel_style = {'size': size_lat_lon_coords, 'color': 'black', 'weight': 'normal'}


''' Well-known temperature biase exhibited by CMIP6 models from: https://doi.org/10.1029/2022GL100888 '''
# Artic
ax100.add_patch(mpatches.Rectangle(xy=[-177, -86], width=354, height=38,
                                facecolor='none', edgecolor='#299103', linewidth=6, zorder=2, # green
                                transform=ccrs.PlateCarree()))
# Cold tongue
ax100.add_patch(mpatches.Rectangle(xy=[-178, -7], width=90, height=14,
                                facecolor='none', edgecolor='#000000', linewidth=6, zorder=2, # black
                                transform=ccrs.PlateCarree()))
# South East Atlantic
ax100.add_patch(mpatches.Rectangle(xy=[-25, -38], width=45, height=48,
                                facecolor='none', edgecolor='#0ff7eb', linewidth=6, zorder=2, # light blue
                                transform=ccrs.PlateCarree()))
# North West Atlantic
ax100.add_patch(mpatches.Rectangle(xy=[155, 10], width=75, height=35,
                                facecolor='none', edgecolor='#f7ae0f', linewidth=6, zorder=2, # orange
                                transform=ccrs.PlateCarree()))
# Gulf Stream
ax100.add_patch(mpatches.Rectangle(xy=[-80, 30], width=60, height=35,
                                facecolor='none', edgecolor='#f70feb', linewidth=6, zorder=2, # purple
                                transform=ccrs.PlateCarree()))
# North East Pacific
ax100.add_patch(mpatches.Rectangle(xy=[-127, 15], width=23, height=25,
                                facecolor='none', edgecolor='#36e605', linewidth=6, zorder=2, # light green
                                transform=ccrs.PlateCarree()))

#################
#   CMIP6-OBS   #
#################
""" Plot CMIP6 ensemble-obs difference averaged in validation years """
ax101 = fig.add_subplot(gs1[0,1], projection=ccrs.Robinson())
difference_warming_cyclic_data,lons_cyclic=add_cyclic_point(diff_models_ensemble_array_avg,coord=lons)
cs101=ax101.contourf(lons_cyclic,lats,difference_warming_cyclic_data, levels=levels_bias,
                        transform = ccrs.PlateCarree(),
                        cmap=colormap, vmin=vmin_bias, vmax=vmax_bias)
ax101.set_title(f'Bias (CMIP6 ensemble–Obs)', loc='center', size=size_title_axes, pad=17)
ax101.coastlines()
gl_bias = ax101.gridlines(crs=ccrs.PlateCarree(), draw_labels=False, linestyle='--', linewidth=0.1, color='black')
gl_bias.xlabel_style = {'size': size_lat_lon_coords, 'color': 'black', 'weight': 'normal'}
gl_bias.ylabel_style = {'size': size_lat_lon_coords, 'color': 'black', 'weight': 'normal'}

# Artic
ax101.add_patch(mpatches.Rectangle(xy=[-177, -86], width=354, height=38,
                                facecolor='none', edgecolor='#299103', linewidth=6, zorder=2, # green
                                transform=ccrs.PlateCarree()))
# Cold tongue
ax101.add_patch(mpatches.Rectangle(xy=[-178, -7], width=90, height=14,
                                facecolor='none', edgecolor='#000000', linewidth=6, zorder=2, # black
                                transform=ccrs.PlateCarree()))
# South East Atlantic
ax101.add_patch(mpatches.Rectangle(xy=[-25, -38], width=45, height=48,
                                facecolor='none', edgecolor='#0ff7eb', linewidth=6, zorder=2, # light blue
                                transform=ccrs.PlateCarree()))
# North West Pacific
ax101.add_patch(mpatches.Rectangle(xy=[155, 10], width=75, height=35,
                                facecolor='none', edgecolor='#f7ae0f', linewidth=6, zorder=2, # orange
                                transform=ccrs.PlateCarree()))
# Gulf Stream
ax101.add_patch(mpatches.Rectangle(xy=[-80, 30], width=60, height=35,
                                facecolor='none', edgecolor='#f70feb', linewidth=6, zorder=2, # purple
                                transform=ccrs.PlateCarree()))
# North East Pacific
ax101.add_patch(mpatches.Rectangle(xy=[-127, 15], width=23, height=25,
                                facecolor='none', edgecolor='#36e605', linewidth=6, zorder=2, # light green
                                transform=ccrs.PlateCarree()))

'''
    Colorbar for DNNs-Obs e CMIP6-Obs 
'''
cbarticks_bias = [-5.8, -4, -2, 0, 2, 4, 6, 7.5]
cbar_bias = fig.colorbar(cs100, shrink=0.5, aspect=40, ax=[ax100, ax101], orientation='horizontal', ticks=cbarticks_bias, pad=0.07)
cbar_bias.set_label(label='Surface Air Temperature difference [°C]', size=size_colorbar_labels, labelpad=20)
for l in cbar_bias.ax.xaxis.get_ticklabels():
    l.set_size(size_colorbar_ticks)

ax2 = fig.add_subplot(gs2[:])
ax2.set_title('Difference (DNNs ensemble–Obs)', y=0.75, size=size_title_axes+5)
plt.axis('off')

""" DNNs ensemble-obs difference for each validation year """
diff_min_4 = (avg_ensemble_maps[idx_scenario,:,:,:]-selected_BEST_data_array[:,:,:]).min()
diff_max_4 = (avg_ensemble_maps[idx_scenario,:,:,:]-selected_BEST_data_array[:,:,:]).max()
abs_diff_max_4 = np.max(np.abs([diff_min_4, diff_max_4]))

ax200 = fig.add_subplot(gs2[0,0], projection=ccrs.Robinson())
difference_warming_cyclic_data,lons_cyclic=add_cyclic_point(avg_ensemble_maps[idx_scenario,0,:,:]-selected_BEST_data_array[0,:,:],coord=lons)
cs200=ax200.contourf(lons_cyclic,lats,difference_warming_cyclic_data, levels=levels_bias,
                        transform = ccrs.PlateCarree(),
                        cmap=colormap, vmin=vmin_bias, vmax=vmax_bias)
ax200.set_title(f'{val_years_list_tl_obs[0]}', loc='center', size=size_title_axes, pad=17)
ax200.coastlines()
gl200 = ax200.gridlines(crs=ccrs.PlateCarree(), draw_labels=False, linestyle='--', linewidth=0.1, color='black')
gl200.xlabel_style = {'size': size_lat_lon_coords, 'color': 'black', 'weight': 'normal'}
gl200.ylabel_style = {'size': size_lat_lon_coords, 'color': 'black', 'weight': 'normal'}


ax201 = fig.add_subplot(gs2[0,1], projection=ccrs.Robinson())
difference_warming_cyclic_data,lons_cyclic=add_cyclic_point(avg_ensemble_maps[idx_scenario,1,:,:]-selected_BEST_data_array[1,:,:],coord=lons)
cs201=ax201.contourf(lons_cyclic,lats,difference_warming_cyclic_data, levels=levels_bias,
                        transform = ccrs.PlateCarree(),
                        cmap=colormap, vmin=vmin_bias, vmax=vmax_bias)
ax201.set_title(f'{val_years_list_tl_obs[1]}', loc='center', size=size_title_axes, pad=17)
ax201.coastlines()
gl201 = ax201.gridlines(crs=ccrs.PlateCarree(), draw_labels=False, linestyle='--', linewidth=0.1, color='black')
gl201.xlabel_style = {'size': size_lat_lon_coords, 'color': 'black', 'weight': 'normal'}
gl201.ylabel_style = {'size': size_lat_lon_coords, 'color': 'black', 'weight': 'normal'}


ax202 = fig.add_subplot(gs2[0,2], projection=ccrs.Robinson())
difference_warming_cyclic_data,lons_cyclic=add_cyclic_point(avg_ensemble_maps[idx_scenario,2,:,:]-selected_BEST_data_array[2,:,:],coord=lons)
cs202=ax202.contourf(lons_cyclic,lats,difference_warming_cyclic_data, levels=levels_bias,
                        transform = ccrs.PlateCarree(),
                        cmap=colormap, vmin=vmin_bias, vmax=vmax_bias)
ax202.set_title(f'{val_years_list_tl_obs[2]}', loc='center', size=size_title_axes, pad=17)
ax202.coastlines()
gl202 = ax202.gridlines(crs=ccrs.PlateCarree(), draw_labels=False, linestyle='--', linewidth=0.1, color='black');
gl202.xlabel_style = {'size': size_lat_lon_coords, 'color': 'black', 'weight': 'normal'}
gl202.ylabel_style = {'size': size_lat_lon_coords, 'color': 'black', 'weight': 'normal'}


ax203 = fig.add_subplot(gs2[0,3], projection=ccrs.Robinson())
difference_warming_cyclic_data,lons_cyclic=add_cyclic_point(avg_ensemble_maps[idx_scenario,3,:,:]-selected_BEST_data_array[3,:,:],coord=lons)
cs203=ax203.contourf(lons_cyclic,lats,difference_warming_cyclic_data, levels=levels_bias,
                        transform = ccrs.PlateCarree(),
                        cmap=colormap, vmin=vmin_bias, vmax=vmax_bias)
ax203.set_title(f'{val_years_list_tl_obs[3]}', loc='center', size=size_title_axes, pad=17)
ax203.coastlines()
gl203 = ax203.gridlines(crs=ccrs.PlateCarree(), draw_labels=False, linestyle='--', linewidth=0.1, color='black')
gl203.xlabel_style = {'size': size_lat_lon_coords, 'color': 'black', 'weight': 'normal'}
gl203.ylabel_style = {'size': size_lat_lon_coords, 'color': 'black', 'weight': 'normal'}

cbar4 = fig.colorbar(cs203, aspect=40, shrink=0.5, ax=[ax200, ax201, ax202, ax203], orientation='horizontal', ticks=cbarticks_bias, pad=0.07)
cbar4.set_label(label='Surface Air Temperature difference [°C]', size=size_colorbar_labels, labelpad=20)
for l in cbar4.ax.xaxis.get_ticklabels():
    l.set_size(size_colorbar_ticks)

plt.text(x=0.12, y=0.9, s='A', fontsize=60, fontweight='bold', ha="center", transform=fig.transFigure)
plt.text(x=0.12, y=0.425, s='B', fontsize=60, fontweight='bold', ha="center", transform=fig.transFigure)
plt.savefig('./Fig_4.png', bbox_inches='tight', dpi=300)