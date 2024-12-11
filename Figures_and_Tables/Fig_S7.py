"""
Author: Francesco Immorlano

Script for reproducing Figure S7
"""

import sys
sys.path.insert(1, './..')
from lib import *

baseline_years = '1850-1900'

'''
shuffle_number = '01' to shuffle_number = '22'
Set shuffle_number = '01' to reproduce Figure S7 present in the paper
'''
shuffle_number = '01'
shuffle_idx = int(shuffle_number) - 1
models_list = models_list.copy()
model_taken_out = models_list[shuffle_idx]
models_list.remove(model_taken_out)


print(f'\nModel taken out: {model_taken_out} - shuffle: {shuffle_number}')
pickle_in = open(f'{ROOT_SOURCE_DATA}/Transfer_Learning_on_Simulations_reverse/Predictions_shuffle-{shuffle_number}.pickle', 'rb')
predictions = pickle.load(pickle_in)

# Load smoothed CMIP6 simulations
pickle_in = open(f'{PATH_SMOOTHED_CMIP6_SIMULATIONS_DIRECTORY}/smooth_splines_dof-{n_dof}_CMIP6_warming_{baseline_years}.pickle','rb')
smooth_warming_simulations = pickle.load(pickle_in)

smooth_warming_taken_out_simulation = smooth_warming_simulations[shuffle_idx, :,:,:,:]

remaining_models_idx = np.arange(22)
remaining_models_idx = np.delete(remaining_models_idx, shuffle_idx)

smooth_warming_simulations_remaining = smooth_warming_simulations[remaining_models_idx,:,:,:]

# Convert from K to Celsius degrees
predictions_C = predictions - 273.15

# Compute climatologies is 1850-1900
predictions_baseline = np.mean(predictions[:,:,:1900-1850+1,:,:], axis=2) # (21,3,64,128)

# Compute warming wrt 1850-1900
predictions[:,:,:,:,:] -= predictions_baseline[:,:,np.newaxis,:,:]

# Compute spatial average warming
annual_predictions_means = ((predictions * area_cella).sum(axis=(-1,-2)))/total_earth_area # (21, 3, 249)
smooth_warming_simulations_remaining_means = ((smooth_warming_simulations_remaining * area_cella).sum(axis=(-1,-2)))/total_earth_area # (21, 3, 249)
smooth_warming_taken_out_simulation_means = ((smooth_warming_taken_out_simulation * area_cella).sum(axis=(-1,-2)))/total_earth_area # (3, 249)

# Compute average across DNNs predictions
ensemble_predictions_means = np.mean(annual_predictions_means, axis=0)

# Compute 5-95% for temperatures predicted by the DNNs in 1850-2098
q05_predictions = np.zeros((len(short_scenarios_list),249))
q95_predictions = np.zeros((len(short_scenarios_list),249))
for short_scenario_idx, short_scenario in enumerate(short_scenarios_list):
    for i in range(2098-1850+1):
        q05_predictions[short_scenario_idx,i] = np.percentile(annual_predictions_means[:,short_scenario_idx,i],5)
        q95_predictions[short_scenario_idx,i] = np.percentile(annual_predictions_means[:,short_scenario_idx,i],95)

# Compute 5-95% for temperatures simulated by CMIP6 ESMs in 1850-2098
q05_simulations = np.zeros((len(short_scenarios_list),2098-1850+1))
q95_simulations = np.zeros((len(short_scenarios_list),2098-1850+1))
for short_scenario_idx, short_scenario in enumerate(short_scenarios_list):
    for i in range(2098-1850+1):
        q05_simulations[short_scenario_idx,i] = np.percentile(smooth_warming_simulations_remaining_means[:,short_scenario_idx,i],5)
        q95_simulations[short_scenario_idx,i] = np.percentile(smooth_warming_simulations_remaining_means[:,short_scenario_idx,i],95)

# Compute RMSE
difference_means = annual_predictions_means - smooth_warming_simulations_remaining_means
squared_diff = difference_means[:,:,:2023-1850+1] ** 2
ms_diff = np.mean(squared_diff, axis=0)
rms_years = np.sqrt(ms_diff)
rmse_scenario = np.mean(rms_years, axis=1)

fig, axs = plt.subplots(len(short_scenarios_list), figsize=(16,18))
plt.subplots_adjust(hspace=0.5)

for scenario_short_idx, scenario_short in enumerate(short_scenarios_list):
    scenario = f'SSP{scenario_short[-3]}-{scenario_short[-2]}.{scenario_short[-1]}'
    # taken out simulation
    axs[scenario_short_idx].plot(np.arange(1850, 2099), smooth_warming_taken_out_simulation_means[scenario_short_idx,:], linewidth=3, label=f'{model_taken_out} (model taken out)',  zorder=3, color='#D51500')
    # CMIP6 spread shading
    axs[scenario_short_idx].fill_between(np.arange(1850, 2099), q05_simulations[scenario_short_idx,:], q95_simulations[scenario_short_idx,:], zorder=1, facecolor='#00FFFF')

    # train set shading
    axs[scenario_short_idx].fill_between(np.arange(start_year_training_loo_cv_reverse, end_year_training_loo_cv_reverse+1), np.floor(q05_simulations[scenario_short_idx,0]), ensemble_predictions_means[scenario_short_idx,2023-1850:2098-1850+1], color='red', alpha=0.07, zorder=0)
    # DNN ensemble
    axs[scenario_short_idx].plot(np.arange(1850, 2099), ensemble_predictions_means[scenario_short_idx,:], linewidth=3, label=f'DNNs ensemble', zorder=2, color='#0064D5')
    # DNN ensemble spread shading
    axs[scenario_short_idx].fill_between(np.arange(1850, 2099), q05_predictions[scenario_short_idx,:], q95_predictions[scenario_short_idx,:], facecolor='#007FEA', alpha=0.5, zorder=2)
    axs[scenario_short_idx].set_xticks([1850, 1900, 1950, 2000, 2023, 2050, 2098])
    axs[scenario_short_idx].tick_params(axis='both', which='major', labelsize=15)
    axs[scenario_short_idx].set_title(f'Scenario {scenario}\nRMSE (1850–2022): {round(rmse_scenario[scenario_short_idx],2)} °C',
                                      size=19, linespacing=1.5, pad=20)

    scenario = f'SSP{scenario_short[-3]}-{scenario_short[-2]}.{scenario_short[-1]}'
    axs[scenario_short_idx].set_ylim([np.floor(q05_simulations[scenario_short_idx,0]), np.ceil(q95_simulations[scenario_short_idx,-1])+0.5])
    if scenario_short_idx == 0:
        axs[scenario_short_idx].set_ylim([np.floor(q05_simulations[scenario_short_idx,0]), q95_simulations[scenario_short_idx,-1]+0.5])
        axs[scenario_short_idx].legend(loc='upper left', fontsize=14)

fig.add_subplot(1, 1, 1, frame_on=False)

plt.tick_params(labelcolor="none", bottom=False, left=False)

plt.xlabel('Years', fontsize=19, labelpad=30)
plt.ylabel('Surface Air Temperature relative to 1850–1900 (°C)', fontsize=19, labelpad=30)

plt.savefig(f'./Fig_S7/Fig_S7_{model_taken_out}.png', dpi=100, bbox_inches='tight')

