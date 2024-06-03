"""
Author: Francesco Immorlano

Script for reproducing Figure 2
"""

import sys
sys.path.insert(1, './..')
from lib import *

plot_figure_paper = True

start_year_val = 2017
end_year_val = 2020

# window to compute time-to-threshold uncertainty
window_size = 21

""" Load predictions made by the DNNs after transfer learning on observative data """
predictions = read_tl_obs_predictions(n_BEST_datasets_per_model_scenario, plot_figure_paper)

""" Load CMIP6 simulations """
simulations = read_all_cmip6_simulations()

""" Load BEST observational data """
BEST_data_array = read_BEST_data(PATH_BEST_DATA)

""" Load BEST observational data uncertainty """
annual_uncertainties_list = read_BEST_data_uncertainty()

# Convert from K to Celsius degrees
simulations_C = simulations - 273.15
predictions_C = predictions - 273.15
BEST_data_array_C = BEST_data_array - 273.15

# Compute average global surface air temperature
annual_simulations_means = ((simulations_C * area_cella).sum(axis=(-1,-2)))/total_earth_area
annual_predictions_means = ((predictions_C * area_cella).sum(axis=(-1,-2)))/total_earth_area
annual_BEST_data_means = ((BEST_data_array_C * area_cella).sum(axis=(-1,-2)))/total_earth_area

# Compute average across CMIP6 ESMs and DNNs predictions
ensemble_simulations_means = np.mean(annual_simulations_means, axis=(0))
ensemble_predictions_means = np.mean(annual_predictions_means, axis=(0,1))

# Compute warming wrt pre-industrial period
warming_ensemble_simulations_means = ensemble_simulations_means - global_mean_temp_1850_1900
warming_ensemble_predictions_means = ensemble_predictions_means - global_mean_temp_1850_1900
warming_annual_simulations_means = annual_simulations_means - global_mean_temp_1850_1900
warming_annual_predictions_means = annual_predictions_means - global_mean_temp_1850_1900
warming_annual_BEST_data_means = annual_BEST_data_means - global_mean_temp_1850_1900

# Compute years to 1.5°C and 2°C thresholds
years_to_thresholds_ensemble = np.zeros((len(short_scenarios_list_complete), 2))
for short_scenario_idx, short_scenario in enumerate(short_scenarios_list_complete):
    years_to_thresholds_ensemble[short_scenario_idx] = compute_years_to_threshold(window_size, warming_ensemble_predictions_means[short_scenario_idx,:])
years_to_thresholds_ensemble = np.round(years_to_thresholds_ensemble).astype(int)

# Compute 5-95% for DNNs predictions
q05_predictions = np.zeros((len(short_scenarios_list_complete),2098-1979+1))
q95_predictions = np.zeros((len(short_scenarios_list_complete),2098-1979+1))
for short_scenario_idx, short_scenario in enumerate(short_scenarios_list_complete):
    for i in range(2098-1979+1):
        q05_predictions[short_scenario_idx,i] = np.percentile(warming_annual_predictions_means[:,:,short_scenario_idx,i],5)
        q95_predictions[short_scenario_idx,i] = np.percentile(warming_annual_predictions_means[:,:,short_scenario_idx,i],95)

# Compute 5-95% for CMIP6 ESMs simulations
q05_simulations = np.zeros((len(short_scenarios_list_complete),2098-1850+1))
q95_simulations = np.zeros((len(short_scenarios_list_complete),2098-1850+1))
for short_scenario_idx, short_scenario in enumerate(short_scenarios_list_complete):
    for i in range(2098-1850+1):
        q05_simulations[short_scenario_idx,i] = np.percentile(warming_annual_simulations_means[:,short_scenario_idx,i],5)
        q95_simulations[short_scenario_idx,i] = np.percentile(warming_annual_simulations_means[:,short_scenario_idx,i],95)


""" Plot """
# zorder
# 0 train set shading
# 1 CMIP6 spread shading
# 2 CMIP6 simulations
# 3 predictions spread shading
# 4 predictions
# 5 CMIP6 ensemble
# 6 predictions ensemble
# 7 BEST data
# 8 BEST data spread
# 9 Paris agreement thresholds
# 9 2098 temperature value
fig, axs = plt.subplots(len(short_scenarios_list_complete), figsize=(16,18))
plt.subplots_adjust(hspace=0.4)
for scenario_short_idx, scenario_short in enumerate(short_scenarios_list_complete):
    # BEST
    axs[scenario_short_idx].scatter(np.arange(start_year_training_tl_obs, end_year_training_tl_obs+1), warming_annual_BEST_data_means, linewidth=1, label=f'BEST observational data', color='black', zorder=7)
    # BEST uncertainty shading
    axs[scenario_short_idx].fill_between(np.arange(start_year_training_tl_obs, end_year_training_tl_obs+1), warming_annual_BEST_data_means-annual_uncertainties_list, warming_annual_BEST_data_means+annual_uncertainties_list, facecolor='#FF5733', zorder=8)
    # training set shading
    axs[scenario_short_idx].fill_between(np.arange(start_year_training_tl_obs, start_year_val), -2, warming_ensemble_predictions_means[0,:(start_year_val-1)-start_year_training_tl_obs+1], color='red', alpha=0.15, zorder = 0)
    axs[scenario_short_idx].fill_between(np.arange(end_year_val+1, end_year_training_tl_obs+1), -2, warming_ensemble_predictions_means[0,:end_year_training_tl_obs-(end_year_val+1)+1], color='red', alpha=0.15, zorder = 0)
    # validation set shading
    axs[scenario_short_idx].fill_between(np.arange(start_year_val, end_year_val+1), -2, warming_ensemble_predictions_means[0,:end_year_val-start_year_val+1], color='grey', alpha=0.2, zorder = 0)
    # DNNs predictions ensemble
    axs[scenario_short_idx].plot(np.arange(start_year_training_tl_obs, end_year_test_tl_obs+1), warming_ensemble_predictions_means[scenario_short_idx,:], linewidth=4, label=f'DNNs multi-model mean', color='#1d73b3', zorder=6)
    # predictions 5-95% range uncertainty shading
    axs[scenario_short_idx].fill_between(np.arange(start_year_training_tl_obs, end_year_test_tl_obs+1), q05_predictions[scenario_short_idx,:], q95_predictions[scenario_short_idx,:], facecolor='#7EFDFF', zorder=3)
    # CMIP6 ensemble
    axs[scenario_short_idx].plot(np.arange(start_year_training_tl_obs, end_year_test_tl_obs+1), warming_ensemble_simulations_means[scenario_short_idx,1979-1850:], linewidth=4, label=f'CMIP6 multi-model mean', color='#F56113', zorder=5)
    # CMIP6 5-95% range uncertainty shading
    axs[scenario_short_idx].fill_between(np.arange(start_year_training_tl_obs, end_year_test_tl_obs+1), q05_simulations[scenario_short_idx,1979-1850:], q95_simulations[scenario_short_idx,1979-1850:], facecolor='#FFD67E', zorder=1)

for scenario_short_idx, scenario_short in enumerate(short_scenarios_list_complete):
    scenario = f'SSP{scenario_short[-3]}-{scenario_short[-2]}.{scenario_short[-1]}'
    axs[scenario_short_idx].set_title(f'Scenario {scenario} — Temperature in 2098: {round(warming_ensemble_predictions_means[scenario_short_idx,-1],2)} °C [{np.round(q05_predictions[scenario_short_idx,-1],2)}–{np.round(q95_predictions[scenario_short_idx,-1],2)} °C]',
                                      size=22)
    axs[scenario_short_idx].set_ylim([-1, np.ceil(np.max(q95_simulations[scenario_short_idx,:]))])

    year_to_2_threshold = int(years_to_thresholds_ensemble[scenario_short_idx,1])
    
    # Lines for surpassing 2°C threshold
    axs[scenario_short_idx].hlines(y=2, xmin=1979, xmax=year_to_2_threshold, color= 'r', zorder=9)
    axs[scenario_short_idx].vlines(x=year_to_2_threshold, ymin=-1, ymax=warming_ensemble_predictions_means[scenario_short_idx,year_to_2_threshold-1979], color= 'r', zorder=9)
    if scenario_short_idx == 0:
        axs[scenario_short_idx].set_xticks([1979, 2000, 2016, 2022, 2040, year_to_2_threshold, 2060, 2080, 2098])
    else:
        axs[scenario_short_idx].set_xticks([1979, 2000, 2016, 2022, year_to_2_threshold, 2060, 2080, 2098])

    plt.xlim(left=1979)
    plt.sca(axs[scenario_short_idx])
    plt.yticks(fontsize=17)
    plt.xticks(fontsize=17)
    axs[scenario_short_idx].legend(loc='upper left', prop={'size':14})

fig.add_subplot(1, 1, 1, frame_on=False)

plt.tick_params(labelcolor="none", bottom=False, left=False)
plt.xlabel('Years', fontsize=20, labelpad=30)
plt.ylabel('Global average near surface air temperature anomaly [°C]\nBase period: 1850–1900', fontsize=22, labelpad=30)

plt.savefig(f'Fig_2.png', dpi=300, bbox_inches='tight')
plt.close()
