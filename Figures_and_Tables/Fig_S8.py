"""
Author: Francesco Immorlano

Script for reproducing Figure S8
"""

import sys
sys.path.insert(1, './..')
from lib import *

baseline_years = '1995-2014'

start_year_val_tl_obs = 2017
end_year_val_tl_obs = 2020

# Load predictions made by the DNNs after transfer learning on observational data
predictions_tl = read_tl_obs_predictions(n_BEST_datasets_per_model_scenario, compute_figures_tables_paper, 'Transfer_learning_obs_')

# Load predictions made by the DNNs after training solely on observational data
predictions_train = read_train_obs_predictions(5, 1979, 2022, 2023, 2098)

# Load smoothed CMIP6 simulations
pickle_in = open(f'{PATH_SMOOTHED_CMIP6_SIMULATIONS_DIRECTORY}/smooth_splines_dof-{n_dof}_CMIP6_warming_{baseline_years}.pickle','rb')
smooth_warming_simulations = pickle.load(pickle_in)

# Load BEST observational data
BEST_data_array = read_BEST_data(PATH_BEST_DATA)

# Load BEST observational data uncertainty
annual_uncertainties_list = read_BEST_data_uncertainty()

# Convert from K to Celsius degrees
predictions_tl_C = predictions_tl - 273.15
predictions_train_C = predictions_train - 273.15
BEST_data_array_C = BEST_data_array - 273.15

# Compute baseline in 1995-2014
predictions_train_baseline = np.mean(predictions_train[:,:,:,1995-1979:2014-1979+1,:,:], axis=3)
predictions_tl_baseline = np.mean(predictions_tl[:,:,:,1995-1979:2014-1979+1,:,:], axis=3)
BEST_data_baseline = np.mean(BEST_data_array[1995-1979:2014-1979+1,:,:], axis=0)

# Compute anomaly
warming_BEST_data = BEST_data_array[:,:,:] - BEST_data_baseline[np.newaxis,:,:]
warming_predictions_train = predictions_train[:,:,:,:,:,:] - predictions_train_baseline[:,:,:,np.newaxis,:,:]
warming_predictions_tl = predictions_tl[:,:,:,:,:,:] - predictions_tl_baseline[:,:,:,np.newaxis,:,:]

# Add 0.85 to get anomalies relative to 1850-1900
warming_BEST_data += refperiod_conversion
warming_predictions_tl += refperiod_conversion
warming_predictions_train += refperiod_conversion

smooth_warming_simulations += refperiod_conversion

# Delete years previous to 1850-1900
smooth_warming_simulations = smooth_warming_simulations[:,:,1979-1850:,:,:]

# Compute spatial averages warming
smooth_warming_simulations_means = ((smooth_warming_simulations * area_cella).sum(axis=(-1,-2)))/total_earth_area
warming_predictions_tl_means = ((warming_predictions_tl * area_cella).sum(axis=(-1,-2)))/total_earth_area
warming_predictions_train_means = ((warming_predictions_train * area_cella).sum(axis=(-1,-2)))/total_earth_area
warming_annual_BEST_data_means = ((warming_BEST_data * area_cella).sum(axis=(-1,-2)))/total_earth_area

# Compute average across CMIP6 ESMs and DNNs predictions
smooth_ensemble_warming_simulations_means = np.mean(smooth_warming_simulations_means, axis=0)
ensemble_predictions_tl_means = np.mean(warming_predictions_tl_means, axis=(0,1))
ensemble_predictions_train_means = np.mean(warming_predictions_train_means, axis=(0,1))

""" Compute 5% and 95% """
# DNNs predictions
q05_predictions_tl = np.zeros((len(short_scenarios_list),2098-1979+1))
q95_predictions_tl = np.zeros((len(short_scenarios_list),2098-1979+1))
q05_predictions_train = np.zeros((len(short_scenarios_list),2098-1979+1))
q95_predictions_train = np.zeros((len(short_scenarios_list),2098-1979+1))
for short_scenario_idx, short_scenario in enumerate(short_scenarios_list):
    for i in range(2098-1979+1):
        q05_predictions_tl[short_scenario_idx,i] = np.percentile(warming_predictions_tl_means[:,:,short_scenario_idx,i],5)
        q95_predictions_tl[short_scenario_idx,i] = np.percentile(warming_predictions_tl_means[:,:,short_scenario_idx,i],95)
        q05_predictions_train[short_scenario_idx,i] = np.percentile(warming_predictions_train_means[:,:,short_scenario_idx,i],5)
        q95_predictions_train[short_scenario_idx,i] = np.percentile(warming_predictions_train_means[:,:,short_scenario_idx,i],95)
# CMIP6 ESMs simulations
q05_simulations = np.zeros((len(short_scenarios_list),2098-1979+1))
q95_simulations = np.zeros((len(short_scenarios_list),2098-1979+1))
for short_scenario_idx, short_scenario in enumerate(short_scenarios_list):
    for i in range(2098-1979+1):
        q05_simulations[short_scenario_idx,i] = np.percentile(smooth_warming_simulations_means[:,short_scenario_idx,i],5)
        q95_simulations[short_scenario_idx,i] = np.percentile(smooth_warming_simulations_means[:,short_scenario_idx,i],95)

''' Plot '''
# zorder
# 0 train set shading
# 1 CMIP6 spread shading
# 2 CMIP6 simulations
# 3 spread shading of predictions after TL on obs ensemble
# 3 spread shading of predictions after pre-training
# 4 predictions
# 5 CMIP6 ensemble
# 6 predictions after TL on obs ensemble
# 6 predictions after pre-training on obs
# 7 BEST data
# 8 BEST data spread
# 9 Paris agreement thresholds
# 9 2098 temperature value
fig, axs = plt.subplots(len(short_scenarios_list), figsize=(16,18))
plt.rcParams['font.sans-serif'] = 'Arial'
plt.subplots_adjust(hspace=0.4)
for scenario_short_idx, scenario_short in enumerate(short_scenarios_list):
    # BEST
    axs[scenario_short_idx].scatter(np.arange(start_year_training_tl_obs, end_year_training_tl_obs+1), warming_annual_BEST_data_means, linewidth=1, label=f'BEST observational data', color='black', zorder=7)
    # BEST uncertainty shading
    axs[scenario_short_idx].fill_between(np.arange(start_year_training_tl_obs, end_year_training_tl_obs+1), warming_annual_BEST_data_means-annual_uncertainties_list, warming_annual_BEST_data_means+annual_uncertainties_list, facecolor='#FF5733', zorder=8)

    # training set shading
    axs[scenario_short_idx].fill_between(np.arange(start_year_training_tl_obs, start_year_val_tl_obs), -2, ensemble_predictions_tl_means[0,:(start_year_val_tl_obs-1)-start_year_training_tl_obs+1], color='red', alpha=0.07, zorder = 0)
    axs[scenario_short_idx].fill_between(np.arange(end_year_val_tl_obs+1, end_year_training_tl_obs+1), -2, ensemble_predictions_tl_means[0,(end_year_val_tl_obs+1)-start_year_training_tl_obs : end_year_training_tl_obs-start_year_training_tl_obs], color='red', alpha=0.07, zorder = 0)
    # validation set shading
    axs[scenario_short_idx].fill_between(np.arange(start_year_val_tl_obs, end_year_val_tl_obs+1), -2, ensemble_predictions_tl_means[0, start_year_val_tl_obs-start_year_training_tl_obs : end_year_val_tl_obs-start_year_training_tl_obs+1], color='grey', alpha=0.12, zorder = 0)
    
    # DNNs predictions TL ensemble
    axs[scenario_short_idx].plot(np.arange(start_year_training_tl_obs, end_year_test_tl_obs+1), ensemble_predictions_tl_means[scenario_short_idx,:], linewidth=4, label=f'DNNs multi-model mean', color='#1d73b3', zorder=6)
    # DNNs predictions after pre-train ensemble
    axs[scenario_short_idx].plot(np.arange(start_year_training_tl_obs, end_year_test_tl_obs+1), ensemble_predictions_train_means[scenario_short_idx,:], linewidth=4, label=f'DNNs multi-model mean (trained on observations only)', color='#1c7506', zorder=6)
    # predictions 5-95% range uncertainty shading
    axs[scenario_short_idx].fill_between(np.arange(start_year_training_tl_obs, end_year_test_tl_obs+1), q05_predictions_tl[scenario_short_idx,:], q95_predictions_tl[scenario_short_idx,:], facecolor='#7EFDFF', zorder=3)
    # DNNs predictions after pre-train 5-95% range uncertainty shading
    axs[scenario_short_idx].fill_between(np.arange(start_year_training_tl_obs, end_year_test_tl_obs+1), q05_predictions_train[scenario_short_idx,:], q95_predictions_train[scenario_short_idx,:], facecolor='#abff7e', zorder=3)
    # CMIP6 ensemble
    axs[scenario_short_idx].plot(np.arange(start_year_training_tl_obs, end_year_test_tl_obs+1), smooth_ensemble_warming_simulations_means[scenario_short_idx,:], linewidth=4, label=f'CMIP6 multi-model mean', color='#F56113', zorder=5)
    # CMIP6 5-95% range uncertainty shading
    axs[scenario_short_idx].fill_between(np.arange(start_year_training_tl_obs, end_year_test_tl_obs+1), q05_simulations[scenario_short_idx,:], q95_simulations[scenario_short_idx,:], facecolor='#FFD67E', zorder=1)
    axs[scenario_short_idx].set_xticks([1979, 2000, 2022, 2040, 2060, 2080, 2098])


for scenario_short_idx, scenario_short in enumerate(short_scenarios_list):
    scenario = f'SSP{scenario_short[-3]}-{scenario_short[-2]}.{scenario_short[-1]}'
    axs[scenario_short_idx].set_title(f'Scenario {scenario} — Temperature in 2098: {round(ensemble_predictions_tl_means[scenario_short_idx,-1],2)} °C [{np.round(q05_predictions_tl[scenario_short_idx,-1],2)}–{np.round(q95_predictions_tl[scenario_short_idx,-1],2)} °C]',
                                      size=22)
    # if scenario_short_idx > 0:
    axs[scenario_short_idx].set_ylim([-1, np.ceil(np.max(q95_predictions_train[scenario_short_idx,:]))+0.5])
    plt.xlim(left=1979)
    plt.sca(axs[scenario_short_idx])
    plt.yticks(fontsize=17)
    plt.xticks(fontsize=17)
    
    if scenario_short_idx == 0:
        axs[scenario_short_idx].legend(loc='upper left', prop={'size':14})

fig.add_subplot(1, 1, 1, frame_on=False)

plt.tick_params(labelcolor="none", bottom=False, left=False)
plt.xlabel('Years', fontsize=20, labelpad=30)
plt.ylabel('Surface Air Temperature relative to 1850–1900 (°C)', fontsize=22, labelpad=30)

plt.savefig(f'Fig_S8.png', dpi=300, bbox_inches='tight')
plt.close()