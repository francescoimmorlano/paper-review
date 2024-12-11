"""
Author: Francesco Immorlano

Script for computing Table S6
"""

import sys
sys.path.insert(1, './..')
from lib import *

# window to compute time-to-threshold uncertainty
window_size = 21

# Load predictions made by the DNNs after transfer learning on observational data
predictions = read_tl_obs_predictions(n_BEST_datasets_per_model_scenario, compute_figures_tables_paper, 'Transfer_learning_obs_2024-11-20_01-04-11')

# Convert from K to Celsius degrees
predictions_C = predictions - 273.15

# Compute climatologies in 1995-2014
predictions_baseline = np.mean(predictions[:,:,:,1995-1979:2014-1979+1,:,:], axis=3) # (5, 22, 3, 64, 128)
# Compute warming wrt 1995-2014
warming_predictions = predictions[:,:,:,:,:,:] - predictions_baseline[:,:,:, np.newaxis,:,:] # (5, 22, 3, 120, 64, 128)
# Add 0.85 to get anomaly in 1850-1900
warming_predictions += refperiod_conversion

# Compute spatial average warming
warming_predictions_means = ((warming_predictions * area_cella).sum(axis=(-1,-2)))/total_earth_area

# Compute average across DNNs predictions
warming_ensemble_predictions_means = np.mean(warming_predictions_means, axis=(0,1))

years_to_2_threshold_array = np.zeros((predictions.shape[0], len(models_list), len(short_scenarios_list)))
years_to_1_5_threshold_array = np.zeros((predictions.shape[0], len(models_list), len(short_scenarios_list)))

q05_years_to_2_threshold = np.zeros((len(short_scenarios_list)))
q95_years_to_2_threshold = np.zeros((len(short_scenarios_list)))

q05_years_to_1_5_threshold = np.zeros((len(short_scenarios_list)))
q95_years_to_1_5_threshold = np.zeros((len(short_scenarios_list)))

years_to_thresholds_ensemble = np.zeros((len(short_scenarios_list), 2))

for short_scenario_idx, short_scenario in enumerate(short_scenarios_list):
    years_to_thresholds_ensemble[short_scenario_idx] = compute_years_to_threshold(window_size, warming_ensemble_predictions_means[short_scenario_idx,:])

    for dataset_index in range(predictions.shape[0]):
        for model_idx, model in enumerate(models_list):
                year_to_thresholds = compute_years_to_threshold(window_size, warming_predictions_means[dataset_index, model_idx, short_scenario_idx,:])

                years_to_1_5_threshold_array[dataset_index,model_idx,short_scenario_idx] = year_to_thresholds[0]
                years_to_2_threshold_array[dataset_index,model_idx,short_scenario_idx] = year_to_thresholds[1]

    q05_years_to_2_threshold[short_scenario_idx] = np.nanpercentile(years_to_2_threshold_array[:,:,short_scenario_idx],5)
    q95_years_to_2_threshold[short_scenario_idx] = np.nanpercentile(years_to_2_threshold_array[:,:,short_scenario_idx],95)
    
    q05_years_to_1_5_threshold[short_scenario_idx] = np.nanpercentile(years_to_1_5_threshold_array[:,:,short_scenario_idx],5)
    q95_years_to_1_5_threshold[short_scenario_idx] = np.nanpercentile(years_to_1_5_threshold_array[:,:,short_scenario_idx],95)

years_to_thresholds_ensemble = np.round(years_to_thresholds_ensemble).astype(int)
q05_years_to_2_threshold = np.round(q05_years_to_2_threshold).astype(int)
q95_years_to_2_threshold = np.round(q95_years_to_2_threshold).astype(int)
q05_years_to_1_5_threshold = np.round(q05_years_to_1_5_threshold).astype(int)
q95_years_to_1_5_threshold = np.round(q95_years_to_1_5_threshold).astype(int)


print(f'\nYear exceeding 1.5°C')
print(f'\t SSP2-4.5: {years_to_thresholds_ensemble[0,0]} ({q05_years_to_1_5_threshold[0]}—{q95_years_to_1_5_threshold[0]})')
print(f'\t SSP3-7.0: {years_to_thresholds_ensemble[1,0]} ({q05_years_to_1_5_threshold[1]}—{q95_years_to_1_5_threshold[1]})')
print(f'\t SSP5-8.5: {years_to_thresholds_ensemble[2,0]} ({q05_years_to_1_5_threshold[2]}—{q95_years_to_1_5_threshold[2]})\n')
print(f'Year exceeding 2°C')
print(f'\t SSP2-4.5: {years_to_thresholds_ensemble[0,1]} ({q05_years_to_2_threshold[0]}—{q95_years_to_2_threshold[0]})')
print(f'\t SSP3-7.0: {years_to_thresholds_ensemble[1,1]} ({q05_years_to_2_threshold[1]}—{q95_years_to_2_threshold[1]})')
print(f'\t SSP5-8.5: {years_to_thresholds_ensemble[2,1]} ({q05_years_to_2_threshold[2]}—{q95_years_to_2_threshold[2]})\n')

