"""
Author: Francesco Immorlano

Script for computing Table S4
"""

import sys
sys.path.insert(1, './..')
from lib import *

# Load predictions made by the DNNs after transfer learning on observational data
predictions = read_tl_obs_predictions(n_BEST_datasets_per_model_scenario, compute_figures_tables_paper, 'Transfer_learning_obs_')

# Convert from K to Celsius degrees
predictions_C = predictions - 273.15

# Compute climatologies in 1995-2014
predictions_baseline = np.mean(predictions[:,:,:,1995-1979:2014-1979+1,:,:], axis=3)

# Compute warming wrt 1995-2015
warming_predictions = predictions[:,:,:,:,:,:] - predictions_baseline[:,:,:, np.newaxis,:,:]

# Compute spatial average warming
warming_predictions_means = ((warming_predictions * area_cella).sum(axis=(-1,-2)))/total_earth_area

# Compute average across CMIP6 ESMs and DNNs predictions
warming_ensemble_predictions_means = np.mean(warming_predictions_means, axis=(0,1))

""" Compute 5% and 95% """
# DNNs predictions
q05_predictions = np.zeros((len(short_scenarios_list),2098-1979+1))
q95_predictions = np.zeros((len(short_scenarios_list),2098-1979+1))
for short_scenario_idx, short_scenario in enumerate(short_scenarios_list):
    for i in range(2098-1979+1):
        q05_predictions[short_scenario_idx,i] = np.percentile(warming_predictions_means[:,:,short_scenario_idx,i],5)
        q95_predictions[short_scenario_idx,i] = np.percentile(warming_predictions_means[:,:,short_scenario_idx,i],95)
print(f'\nDNNs average global temperature anomaly in 2021-2040 wrt 1850-1900: avg (5%—95%)')
print(f'\tSSP2-4.5: {np.round(np.mean(warming_ensemble_predictions_means[0,2021-1979:2040-1979+1]),2)} ({np.round(np.mean(q05_predictions[0,2021-1979:2040-1979+1]),2)}—{np.round(np.mean(q95_predictions[0,2021-1979:2040-1979+1]),2)})')
print(f'\tSSP3-7.0: {np.round(np.mean(warming_ensemble_predictions_means[1,2021-1979:2040-1979+1]),2)} ({np.round(np.mean(q05_predictions[1,2021-1979:2040-1979+1]),2)}—{np.round(np.mean(q95_predictions[1,2021-1979:2040-1979+1]),2)})')
print(f'\tSSP5-8.5: {np.round(np.mean(warming_ensemble_predictions_means[2,2021-1979:2040-1979+1]),2)} ({np.round(np.mean(q05_predictions[2,2021-1979:2040-1979+1]),2)}—{np.round(np.mean(q95_predictions[2,2021-1979:2040-1979+1]),2)})')

print(f'\nDNNs average global temperature anomaly in 2041-2060 wrt 1850-1900: avg (5%—95%)')
print(f'\tSSP2-4.5: {np.round(np.mean(warming_ensemble_predictions_means[0,2041-1979:2060-1979+1]),2)} ({np.round(np.mean(q05_predictions[0,2041-1979:2060-1979+1]),2)}—{np.round(np.mean(q95_predictions[0,2041-1979:2060-1979+1]),2)})')
print(f'\tSSP3-7.0: {np.round(np.mean(warming_ensemble_predictions_means[1,2041-1979:2060-1979+1]),2)} ({np.round(np.mean(q05_predictions[1,2041-1979:2060-1979+1]),2)}—{np.round(np.mean(q95_predictions[1,2041-1979:2060-1979+1]),2)})')
print(f'\tSSP5-8.5: {np.round(np.mean(warming_ensemble_predictions_means[2,2041-1979:2060-1979+1]),2)} ({np.round(np.mean(q05_predictions[2,2041-1979:2060-1979+1]),2)}—{np.round(np.mean(q95_predictions[2,2041-1979:2060-1979+1]),2)})')

# Values from IPCC WG1 AR6 
ipcc_wg1_2021_2040_q05 = [0.4, 0.4, 0.5]
ipcc_wg1_2021_2040_q95 = [0.9, 0.9, 1]
ipcc_wg1_2041_2060_q05 = [0.8, 0.9, 1.1]
ipcc_wg1_2041_2060_q95 = [1.6, 1.7, 2.1]

# fair-calibrate v1.4.1
smith_fair_q05_141_near = [0.41, 0.41, 0.45]
smith_fair_q95_141_near = [0.91, 0.89, 0.98]

smith_fair_q05_141_med = [0.72, 0.79, 0.94]
smith_fair_q95_141_med = [1.57, 1.54, 1.97]

# fair-calibrate v1.4.0
smith_fair_q05_140_near = [0.4, 0.41, 0.48]
smith_fair_q95_140_near = [0.86, 0.83, 1.02]

smith_fair_q05_140_med = [0.71, 0.86, 1.01]
smith_fair_q95_140_med = [1.54, 1.56, 2.06]

print('\nUncertainty reduction with respect to IPCC WG1 AR6')
for idx_short_scenario, scenario_short in enumerate(short_scenarios_list):
    print(f'SSP{scenario_short[-3]}-{scenario_short[-2]}.{scenario_short[-1]}')
    print(f'\tNear-term (2021-2040):\t\t{np.round(((ipcc_wg1_2021_2040_q95[idx_short_scenario]-ipcc_wg1_2021_2040_q05[idx_short_scenario])-(np.round(np.mean(q95_predictions[idx_short_scenario,2021-1979:2040-1979+1]),2)-np.round(np.mean(q05_predictions[idx_short_scenario,2021-1979:2040-1979+1]),2)))/(ipcc_wg1_2021_2040_q95[idx_short_scenario]-ipcc_wg1_2021_2040_q05[idx_short_scenario])*100).astype(int)}%')
    print(f'\tMid-term (2041-2060):\t\t{np.round(((ipcc_wg1_2041_2060_q95[idx_short_scenario]-ipcc_wg1_2041_2060_q05[idx_short_scenario])-(np.round(np.mean(q95_predictions[idx_short_scenario,2041-1979:2060-1979+1]),2)-np.round(np.mean(q05_predictions[idx_short_scenario,2041-1979:2060-1979+1]),2)))/(ipcc_wg1_2041_2060_q95[idx_short_scenario]-ipcc_wg1_2041_2060_q05[idx_short_scenario])*100).astype(int)}%')

print('\nUncertainty reduction with respect to fair-calibrate v1.4.1')
for idx_short_scenario, scenario_short in enumerate(short_scenarios_list):
    print(f'SSP{scenario_short[-3]}-{scenario_short[-2]}.{scenario_short[-1]}')
    print(f'\tNear-term (2021-2040):\t\t{np.round(((smith_fair_q95_141_near[idx_short_scenario]-smith_fair_q05_141_near[idx_short_scenario])-(np.round(np.mean(q95_predictions[idx_short_scenario,2021-1979:2040-1979+1]),2)-np.round(np.mean(q05_predictions[idx_short_scenario,2021-1979:2040-1979+1]),2)))/(smith_fair_q95_141_near[idx_short_scenario]-smith_fair_q05_141_near[idx_short_scenario])*100).astype(int)}%')
    print(f'\tMid-term (2041-2060):\t\t{np.round(((smith_fair_q95_141_med[idx_short_scenario]-smith_fair_q05_141_med[idx_short_scenario])-(np.round(np.mean(q95_predictions[idx_short_scenario,2041-1979:2060-1979+1]),2)-np.round(np.mean(q05_predictions[idx_short_scenario,2041-1979:2060-1979+1]),2)))/(smith_fair_q95_141_med[idx_short_scenario]-smith_fair_q05_141_med[idx_short_scenario])*100).astype(int)}%')

print('\nUncertainty reduction with respect to fair-calibrate v1.4.0')
for idx_short_scenario, scenario_short in enumerate(short_scenarios_list):
    print(f'SSP{scenario_short[-3]}-{scenario_short[-2]}.{scenario_short[-1]}')
    print(f'\tNear-term (2021-2040):\t\t{np.round(((smith_fair_q95_140_near[idx_short_scenario]-smith_fair_q05_140_near[idx_short_scenario])-(np.round(np.mean(q95_predictions[idx_short_scenario,2021-1979:2040-1979+1]),2)-np.round(np.mean(q05_predictions[idx_short_scenario,2021-1979:2040-1979+1]),2)))/(smith_fair_q95_140_near[idx_short_scenario]-smith_fair_q05_140_near[idx_short_scenario])*100).astype(int)}%')
    print(f'\tMid-term (2041-2060):\t\t{np.round(((smith_fair_q95_140_med[idx_short_scenario]-smith_fair_q05_140_med[idx_short_scenario])-(np.round(np.mean(q95_predictions[idx_short_scenario,2041-1979:2060-1979+1]),2)-np.round(np.mean(q05_predictions[idx_short_scenario,2041-1979:2060-1979+1]),2)))/(smith_fair_q95_140_med[idx_short_scenario]-smith_fair_q05_140_med[idx_short_scenario])*100).astype(int)}%')

