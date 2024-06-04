"""
Author: Francesco Immorlano

Script for computing Table S4
"""

import sys
sys.path.insert(1, './..')
from lib import *

""" Load predictions made by the DNNs after transfer learning on observational data """
predictions = read_tl_obs_predictions(n_BEST_datasets_per_model_scenario, compute_figures_tables)

""" Load CMIP6 simulations """
simulations = read_all_cmip6_simulations()

# Convert from K to Celsius degrees
simulations_C = simulations - 273.15
predictions_C = predictions - 273.15

# Compute average global surface air temperature
annual_simulations_means = ((simulations_C * area_cella).sum(axis=(-1,-2)))/total_earth_area
annual_predictions_means = ((predictions_C * area_cella).sum(axis=(-1,-2)))/total_earth_area

# Compute average across CMIP6 ESMs and DNNs predictions
ensemble_simulations_means = np.mean(annual_simulations_means, axis=(1))
ensemble_predictions_means = np.mean(annual_predictions_means, axis=(0,1))

# Compute warming wrt pre-industrial period
warming_ensemble_simulations_means = ensemble_simulations_means - global_mean_temp_1995_2014
warming_ensemble_predictions_means = ensemble_predictions_means - global_mean_temp_1995_2014
warming_annual_simulations_means = annual_simulations_means - global_mean_temp_1995_2014
warming_annual_predictions_means = annual_predictions_means - global_mean_temp_1995_2014

""" Compute 5% and 95% """
# DNNs predictions
q05_predictions = np.zeros((len(short_scenarios_list),2098-1979+1))
q95_predictions = np.zeros((len(short_scenarios_list),2098-1979+1))
for short_scenario_idx, short_scenario in enumerate(short_scenarios_list):
    for i in range(2098-1979+1):
        q05_predictions[short_scenario_idx,i] = np.percentile(warming_annual_predictions_means[:,:,short_scenario_idx,i],5)
        q95_predictions[short_scenario_idx,i] = np.percentile(warming_annual_predictions_means[:,:,short_scenario_idx,i],95)
# CMIP6 ESMs simulations
q05_simulations = np.zeros((len(short_scenarios_list),2098-1850+1))
q95_simulations = np.zeros((len(short_scenarios_list),2098-1850+1))
for short_scenario_idx, short_scenario in enumerate(short_scenarios_list):
    for i in range(2098-1850+1):
        q05_simulations[short_scenario_idx,i] = np.percentile(warming_annual_simulations_means[short_scenario_idx,:,i],5)
        q95_simulations[short_scenario_idx,i] = np.percentile(warming_annual_simulations_means[short_scenario_idx,:,i],95)

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

print('\nUncertainty reduction with respect to IPCC WG1 AR6')
for idx_short_scenario, scenario_short in enumerate(short_scenarios_list):
    print(f'SSP{scenario_short[-3]}-{scenario_short[-2]}.{scenario_short[-1]}')
    print(f'\tNear-term (2021-2040):\t\t{np.round(((ipcc_wg1_2021_2040_q95[idx_short_scenario]-ipcc_wg1_2021_2040_q05[idx_short_scenario])-(np.round(np.mean(q95_predictions[idx_short_scenario,2021-1979:2040-1979+1]),2)-np.round(np.mean(q05_predictions[idx_short_scenario,2021-1979:2040-1979+1]),2)))/(ipcc_wg1_2021_2040_q95[idx_short_scenario]-ipcc_wg1_2021_2040_q05[idx_short_scenario])*100).astype(int)}%')
    print(f'\tMid-term (2041-2060):\t\t{np.round(((ipcc_wg1_2041_2060_q95[idx_short_scenario]-ipcc_wg1_2041_2060_q05[idx_short_scenario])-(np.round(np.mean(q95_predictions[idx_short_scenario,2041-1979:2060-1979+1]),2)-np.round(np.mean(q05_predictions[idx_short_scenario,2041-1979:2060-1979+1]),2)))/(ipcc_wg1_2041_2060_q95[idx_short_scenario]-ipcc_wg1_2041_2060_q05[idx_short_scenario])*100).astype(int)}%')