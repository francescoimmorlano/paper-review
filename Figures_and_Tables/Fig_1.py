"""
Author: Francesco Immorlano

Script for reproducing Figure 1
"""

from matplotlib.legend_handler import HandlerTuple
from cartopy.util import add_cyclic_point
import cartopy.crs as ccrs
import sys
sys.path.insert(1, '..')
from lib import *

'''
shuffle_number = '01' to shuffle_number = '22'
Set shuffle_number = '09' to reproduce Figure 1 present in the paper
'''
shuffle_number = '09'
shuffle_idx = int(shuffle_number)-1

baseline_years = '1850-1900'

models_list_take_out = models_list.copy()

# Get and delete the current taken-out model from the list
model_take_out = models_list[shuffle_idx]
models_list_take_out.remove(model_take_out)

print(f'\nModel taken out: {model_take_out} - shuffle: {shuffle_number}')

# Load predictions made by the DNNs after transfer learning on the take-out simulation
predictions = read_tl_simulations_predictions_shuffle(shuffle_idx, compute_figures_tables_paper, 'Transfer_learning_', False, None)

# Load smoothed CMIP6 simulations
simulations = read_all_cmip6_simulations()

pickle_in = open('lons.pickle', 'rb')
lons = pickle.load(pickle_in)
pickle_in.close()

pickle_in = open('lats.pickle', 'rb')
lats = pickle.load(pickle_in)
pickle_in.close()

pickle_in = open(f'{PATH_SMOOTHED_CMIP6_SIMULATIONS_DIRECTORY}/smooth_splines_dof-{n_dof}_CMIP6_warming_{baseline_years}.pickle','rb')
smooth_warming_simulations = pickle.load(pickle_in)

remaining_models_idx = np.arange(22)
remaining_models_idx = np.delete(remaining_models_idx, shuffle_idx)

simulations_remaining = simulations[remaining_models_idx,:,:,:,:]
smooth_warming_simulations_remaining = smooth_warming_simulations[remaining_models_idx,:,:,:,:]

take_out_simulation = simulations[shuffle_idx,:,:,:]
smooth_warming_simulation_takeout = smooth_warming_simulations[shuffle_idx,:,:,:,:]

# Convert from K to Celsius degrees
predictions_C = predictions - 273.15
take_out_simulation_C = take_out_simulation - 273.15
simulations_remaining_C = simulations_remaining - 273.15

# Compute climatologies in 1850-1900 at the grid-point level
take_out_baseline = np.mean(take_out_simulation[:,:1900-1850+1,:,:], axis=1)
predictions_baseline = np.mean(predictions[:,:,:1900-1850+1,:,:], axis=2)
simulations_remaining_baseline = np.mean(simulations_remaining[:,:,:1900-1850+1,:,:], axis=2)

# Compute warming wrt pre-industrial period at the grid-point level
warming_predictions = predictions[:,:,:,:,:] - predictions_baseline[:,:,np.newaxis,:,:]
warming_simulation_takeout = take_out_simulation[:,:,:,:] - take_out_baseline[:,np.newaxis,:,:]
warming_simulations_remaining = simulations_remaining[:,:,:,:,:] - simulations_remaining_baseline[:,:,np.newaxis,:,:]

# Get average temperature maps in 2081-2098 simulated by the take-out model for SSP2-4.5
simulations_2081_2098_ssp245 = take_out_simulation_C[0,2081-1850:2098-1850+1,:,:]
avg_simulations_2081_2098_ssp245 = simulations_2081_2098_ssp245.mean(axis=0)

# Get average temperature maps in 2081-2098 predicted by the DNNs for SSP2-4.5
predictions_2081_2098_ssp245 = predictions_C[:,0,2081-1850:2098-1850+1,:,:]
avg_predictions_2081_2098_ssp245 = predictions_2081_2098_ssp245.mean(axis=(0,1))

# Compute latitude-weigthed global-mean warming
warming_predictions_means = ((warming_predictions * area_cella).sum(axis=(-1,-2)))/total_earth_area
warming_simulations_remaining_means = ((warming_simulations_remaining * area_cella).sum(axis=(-1,-2)))/total_earth_area
smooth_warming_simulation_takeout_means = ((smooth_warming_simulation_takeout * area_cella).sum(axis=(-1,-2)))/total_earth_area
warming_simulation_takeout_means = ((warming_simulation_takeout * area_cella).sum(axis=(-1,-2)))/total_earth_area
smooth_warming_simulations_remaining_means = ((smooth_warming_simulations_remaining * area_cella).sum(axis=(-1,-2)))/total_earth_area

# Compute average across DNNs predictions
ensemble_warming_predictions_means = np.mean(warming_predictions_means, axis=0)

# Compute 5-95% for temperatures predicted by the DNNs in 1850-2098
q05_warming_predictions = np.zeros((len(short_scenarios_list),249))
q95_warming_predictions = np.zeros((len(short_scenarios_list),249))
q05_smooth_warming_simulations_remaining = np.zeros((len(short_scenarios_list),249))
q95_smooth_warming_simulations_remaining = np.zeros((len(short_scenarios_list),249))
q05_warming_simulations_remaining = np.zeros((len(short_scenarios_list),249))
q95_warming_simulations_remaining = np.zeros((len(short_scenarios_list),249))

for short_scenario_idx, short_scenario in enumerate(short_scenarios_list):
    for i in range(2098-1850+1):
        q05_warming_predictions[short_scenario_idx,i] = np.percentile(warming_predictions_means[:,short_scenario_idx,i],5)
        q95_warming_predictions[short_scenario_idx,i] = np.percentile(warming_predictions_means[:,short_scenario_idx,i],95)
        q05_smooth_warming_simulations_remaining[short_scenario_idx,i] = np.percentile(smooth_warming_simulations_remaining_means[:,short_scenario_idx,i],5)
        q95_smooth_warming_simulations_remaining[short_scenario_idx,i] = np.percentile(smooth_warming_simulations_remaining_means[:,short_scenario_idx,i],95)
        q05_warming_simulations_remaining[short_scenario_idx,i] = np.percentile(warming_simulations_remaining_means[:,short_scenario_idx,i],5)
        q95_warming_simulations_remaining[short_scenario_idx,i] = np.percentile(warming_simulations_remaining_means[:,short_scenario_idx,i],95)

# Compute RMSE
difference_means = warming_predictions_means - smooth_warming_simulation_takeout_means
squared_diff = difference_means[:,:,2023-1850:2098-1850+1] ** 2
ms_diff = np.mean(squared_diff, axis=0)
rms_years = np.sqrt(ms_diff)
rmse_scenario = np.mean(rms_years, axis=1)

# Compute bias map for SSP2-4.5
bias_avg_2081_2098_ssp245_map = avg_predictions_2081_2098_ssp245 - avg_simulations_2081_2098_ssp245
bias_avg_map_max = bias_avg_2081_2098_ssp245_map.max()


""" Plot """
colormap = 'seismic'

size_suptitlefig = 45
size_titlefig = 46
size_annotatation_letters = 46
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
gs01 = gs0[1].subgridspec(3, 1, hspace=0.3)

""" ax1, ax2, ax3 """
for scenario_idx, short_scenario in enumerate(short_scenarios_list):
    ax = fig.add_subplot(gs00[scenario_idx, 0])
    p12, = ax.plot(np.arange(1850, 2099), smooth_warming_simulation_takeout_means[scenario_idx,:], linewidth=5, label=f'{model_take_out} (model taken out)', zorder=4, color='#D51500')
    p13, = ax.plot(np.arange(1850, 2099), ensemble_warming_predictions_means[scenario_idx,:], linewidth=5, label=f'Ensemble', zorder=2, color='#0064D5')
    p14, = ax.plot(np.arange(1850, 2099), q05_warming_simulations_remaining[scenario_idx,:], linewidth=4, zorder=3, color='grey', linestyle='dashed')
    p15, = ax.plot(np.arange(1850, 2099), q95_warming_simulations_remaining[scenario_idx,:], linewidth=4, zorder=3, color='grey', linestyle='dashed')
    # Train set shading
    ax.fill_between(np.arange(start_year_training_loo_cv, end_year_training_loo_cv+1), -2, ensemble_warming_predictions_means[scenario_idx,:end_year_training_loo_cv-start_year_training_loo_cv+1], color='red', alpha=0.07)
    # Predictions 5-95% uncertainty shading
    ax.fill_between(np.arange(1850, 2099), q05_warming_predictions[scenario_idx,:], q95_warming_predictions[scenario_idx,:], zorder=2, facecolor='#007FEA', alpha=0.5)
    ax.fill_between(np.arange(1850, 2099), q05_smooth_warming_simulations_remaining[scenario_idx,:], q95_smooth_warming_simulations_remaining[scenario_idx,:], zorder=1, facecolor='#00FFFF')
    ax.set_xticks([1850, 1900, 1950, 2000, end_year_training_loo_cv, 2050, 2098])
    plt.xticks(fontsize=size_x_y_ticks)
    plt.yticks(fontsize=size_x_y_ticks)

    if scenario_idx == 1:
        plt.ylabel('Surface Air Temperature relative to 1850–1900 (°C)', fontsize=size_x_y_legend, linespacing=1.5,labelpad=80)


    ax.set_ylim([-1, np.ceil(np.max(np.maximum(q95_warming_simulations_remaining[scenario_idx,:], q95_warming_predictions[scenario_idx,:])))])
    ax.set_title(f'SSP{short_scenario[-3]}-{short_scenario[-2]}.{short_scenario[-1]}\nRMSE (2023–2098): {round(rmse_scenario[scenario_idx],2)}°C — Temperature in 2098: {np.round(ensemble_warming_predictions_means[scenario_idx,-1],2)} °C [{np.round(q05_warming_predictions[scenario_idx,-1],2)}–{np.round(q95_warming_predictions[scenario_idx,-1],2)} °C]', size=size_title_scenario_axes,
                    pad=30, linespacing=1.5)
    if scenario_idx == 0:
        l1 = ax.legend([p13, p12, p14], ['DNNs ensemble', model_take_out, '5-95% CMIP6'],
                    handler_map={tuple: HandlerTuple(ndivide=None)}, prop={'size':size_plot_legend}, loc='upper left')

""" ax4 """
ax4 = fig.add_subplot(gs01[0, 0], projection=ccrs.Robinson())
data = avg_simulations_2081_2098_ssp245
data_cyclic, lons_cyclic = add_cyclic_point(data, lons)
cs=ax4.contourf(lons_cyclic, lats, data_cyclic,
                levels=40, vmin=-90, vmax=90,
                transform = ccrs.PlateCarree(),cmap=colormap)
ax4.coastlines()
gl4 = ax4.gridlines(draw_labels=False, linestyle='--')
gl4.xlabel_style = {'size': size_x_y_ticks, 'color': 'k', 'rotation':0}
gl4.ylabel_style = {'size': size_x_y_ticks, 'color': 'k', 'rotation':0}
cbar4 = plt.colorbar(cs,shrink=0.8, orientation='horizontal', pad=0.05)
cbar4.set_label('Surface Air Temperature [°C]',size=size_colorbar_labels,rotation='horizontal', labelpad=15)
for l in cbar4.ax.xaxis.get_ticklabels():
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
gl5 = ax5.gridlines(draw_labels=False, linestyle='--')
gl5.xlabel_style = {'size': size_x_y_ticks, 'color': 'k', 'rotation':0}
gl5.ylabel_style = {'size': size_x_y_ticks, 'color': 'k', 'rotation':0}
cbar5 = plt.colorbar(cs,shrink=0.8, orientation='horizontal', pad=0.05)
cbar5.set_label('Surface Air Temperature [°C]',size=size_colorbar_labels, rotation='horizontal', labelpad=15)
for l in cbar5.ax.xaxis.get_ticklabels():
    l.set_size(size_colorbar_ticks)
ax5.set_title(f'Ensemble DNNs average temperature (2081–2098)', size=size_title_map_axes, pad=17)

""" ax6 """
ax6 = fig.add_subplot(gs01[2, 0], projection=ccrs.Robinson())
data=bias_avg_2081_2098_ssp245_map
data_cyclic, lons_cyclic = add_cyclic_point(data, lons)
levels_3 = np.linspace(-bias_avg_map_max, bias_avg_map_max, 40)
cs=ax6.contourf(lons_cyclic, lats, data_cyclic, levels=40, vmin=-bias_avg_map_max-6, vmax=bias_avg_map_max+6,
            transform = ccrs.PlateCarree(),cmap=colormap)
ax6.coastlines()
gl6 = ax6.gridlines(draw_labels=False, linestyle='--')
gl6.xlabel_style = {'size': size_x_y_ticks, 'color': 'k', 'rotation':0}
gl6.ylabel_style = {'size': size_x_y_ticks, 'color': 'k', 'rotation':0}
cbarticks_6 = [-7,-5,-3,-2,-1,0,1,2,3,4]
cbar = plt.colorbar(cs,shrink=0.8, ticks=cbarticks_6, orientation='horizontal', pad=0.05)
cbar.set_label('Surface Air Temperature difference [°C]',size=size_colorbar_labels, rotation='horizontal', labelpad=15)
for l in cbar.ax.xaxis.get_ticklabels():
    l.set_size(size_colorbar_ticks)
ax6.set_title(f'Difference (DNNs–CMIP6)', size=size_title_map_axes, pad=17)

plt.text(0.1, 0.92, "A", transform=fig.transFigure, fontsize=size_annotatation_letters, fontweight='bold')
plt.text(0.62, 0.92, "B", transform=fig.transFigure, fontsize=size_annotatation_letters, fontweight='bold')
plt.text(0.7, 0.915, "Scenario SSP2-4.5", transform=fig.transFigure, fontsize=size_annotatation_letters-3)
plt.text(0.62, 0.63, "C", transform=fig.transFigure, fontsize=size_annotatation_letters, fontweight='bold')
plt.text(0.62, 0.33, "D", transform=fig.transFigure, fontsize=size_annotatation_letters, fontweight='bold')
plt.savefig(f'./Fig_1/Fig_1_{model_take_out}.png', dpi=300, bbox_inches='tight')
plt.close()