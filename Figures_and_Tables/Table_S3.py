"""
Author: Francesco Immorlano

Script for computing Table S3
"""

import sys
sys.path.insert(1, './..')
from lib import *

baseline_years = '1995-2014'

# Load DNNs predictions
predictions = read_tl_obs_predictions(n_BEST_datasets_per_model_scenario, compute_figures_tables_paper, 'Transfer_learning_obs_')

# Load smoothed CMIP6 simulations
pickle_in = open(f'{PATH_SMOOTHED_CMIP6_SIMULATIONS_DIRECTORY}/smooth_splines_dof-{n_dof}_CMIP6_warming_{baseline_years}.pickle','rb')
smooth_warming_simulations = pickle.load(pickle_in)

# Convert from K to Celsius degrees
predictions_C = predictions - 273.15

# Compute climatologies in 1995-2014
predictions_baseline = np.mean(predictions[:,:,:,1995-1979:2014-1979+1,:,:], axis=3) # (5, 22, 3, 64, 128)

# Compute warming wrt pre-industrial period
warming_predictions = predictions[:,:,:,:,:,:] - predictions_baseline[:,:,:,np.newaxis,:,:] # (5, 22, 3, 120, 64, 128)

# Compute spatial average warming
warming_predictions_means = ((warming_predictions * area_cella).sum(axis=(-1,-2)))/total_earth_area # (5, 22, 3, 120)
smooth_warming_simulations_means = ((smooth_warming_simulations * area_cella).sum(axis=(-1,-2)))/total_earth_area # (22, 3, 120)

# Select predictions anomalies in 2081-2098
warming_predictions_means_2081_2098 = warming_predictions_means[:,:,:,2081-1979:]

# Compute median, 5% and 95%
median_predictions_means_2081_2098 = np.zeros((len(short_scenarios_list),2098-2081+1))
q05_predictions_means_2081_2098 = np.zeros((len(short_scenarios_list),2098-2081+1))
q95_predictions_means_2081_2098 = np.zeros((len(short_scenarios_list),2098-2081+1))
for short_scenario_idx, short_scenario in enumerate(short_scenarios_list):
    # DNNs predictions
    for i in range(2098-2081+1):
        median_predictions_means_2081_2098[short_scenario_idx,i] = np.median(np.ravel(warming_predictions_means_2081_2098[:,:,short_scenario_idx,i]))
        q05_predictions_means_2081_2098[short_scenario_idx,i] = np.percentile(warming_predictions_means_2081_2098[:,:,short_scenario_idx,i],5)
        q95_predictions_means_2081_2098[short_scenario_idx,i] = np.percentile(warming_predictions_means_2081_2098[:,:,short_scenario_idx,i],95)

# DNNs predictions
avg_median_ensemble = median_predictions_means_2081_2098.mean(axis=1)
q05_ensemble = q05_predictions_means_2081_2098.mean(axis=1)
q95_ensemble = q95_predictions_means_2081_2098.mean(axis=1)

# Select simulations anomalies in 2081-2098
smooth_warming_simulations_means_2081_2098 = smooth_warming_simulations_means[:,:,2081-1850:]

# Compute median, 5% and 95%
median_simulations_means_2081_2098 = np.zeros((len(short_scenarios_list),2098-2081+1))
q05_simulations_means_2081_2098 = np.zeros((len(short_scenarios_list),2098-2081+1))
q95_simulations_means_2081_2098 = np.zeros((len(short_scenarios_list),2098-2081+1))
for short_scenario_idx, short_scenario in enumerate(short_scenarios_list):
    # CMIP6 ESMs simulations
    for i in range(2098-2081+1):
        median_simulations_means_2081_2098[short_scenario_idx,i] = np.median(np.ravel(smooth_warming_simulations_means_2081_2098[:,short_scenario_idx,i]))
        q05_simulations_means_2081_2098[short_scenario_idx,i] = np.percentile(smooth_warming_simulations_means_2081_2098[:,short_scenario_idx,i],5)
        q95_simulations_means_2081_2098[short_scenario_idx,i] = np.percentile(smooth_warming_simulations_means_2081_2098[:,short_scenario_idx,i],95)
 
# Compute avg median, 5% and 95% in 2081-2098
# CMIP6 ESMs simulations
avg_median_simulations = median_simulations_means_2081_2098.mean(axis=1)
q05_simulations = q05_simulations_means_2081_2098.mean(axis=1)
q95_simulations = q95_simulations_means_2081_2098.mean(axis=1)

"""
5%, median, 95% average temperature values in 2081–2100 wrt 1995-2014
by Tokarska, Liang and IPCC WGI
5%, median, 95% average temperature values in 2081–2098 wrt 1995-2014
by Ribes
"""
ribes_q05 = [1.22, 2.07, 2.4]
ribes_mean = [1.83, 2.77, 3.46]
ribes_q95 = [2.44, 3.46, 4.53]

tokarska_q05 = [1.04, 1.75, 2.09]
tokarska_median = [1.81, 2.7, 3.43]
tokarska_q95 = [2.56, 3.63, 4.75]

yongxiao_q05 = [1.33, 2.28, 2.6]
yongxiao_median = [1.69, 2.65, 3.26]
yongxiao_q95 = [2.72, 3.85, 4.86]

# Values from IPCC AR6 Ch.4 Table 4.5
ipcc_wg1_q05 = [1.2, 2.0, 2.4]
ipcc_wg1_median = [1.8, 2.8, 3.5]
ipcc_wg1_q95 = [2.6, 3.7, 4.8]

# fair-calibrate v1.4.1
smith_fair_q05_141 = [1.06, 1.63, 2.12]
smith_fair_q95_141 = [2.66, 3.18, 4,37]

# fair-calibrate v1.4.0
smith_fair_q05_140 = [1.06, 1.85, 2.32]
smith_fair_q95_140 = [2.68, 3.52, 4.78]

q05_ensemble = np.round(q05_ensemble,2)
q95_ensemble = np.round(q95_ensemble,2)

print('\nDNNs ensemble warming in 2081–2098 wrt 1995–2014: (5–95% confidence range)')
print(f'\tSSP2-4.5: {np.round(avg_median_ensemble[0],2)} ({np.round(q05_ensemble[0],2)}—{np.round(q95_ensemble[0],2)})')
print(f'\tSSP3-7.0: {np.round(avg_median_ensemble[1],2)} ({np.round(q05_ensemble[1],2)}—{np.round(q95_ensemble[1],2)})')
print(f'\tSSP5-8.5: {np.round(avg_median_ensemble[2],2)} ({np.round(q05_ensemble[2],2)}—{np.round(q95_ensemble[2],2)})\n')

print('\nCMIP6 ensemble warming in 2081–2098 wrt 1995–2014: (5–95% confidence range)')
print(f'\tSSP2-4.5: {np.round(avg_median_simulations[0],2)} ({np.round(q05_simulations[0],2)}—{np.round(q95_simulations[0],2)})')
print(f'\tSSP3-7.0: {np.round(avg_median_simulations[1],2)} ({np.round(q05_simulations[1],2)}—{np.round(q95_simulations[1],2)})')
print(f'\tSSP5-8.5: {np.round(avg_median_simulations[2],2)} ({np.round(q05_simulations[2],2)}—{np.round(q95_simulations[2],2)})\n')

print('Uncertainty reduction (Long-term)')
for idx_short_scenario, scenario_short in enumerate(short_scenarios_list):
    print(f'SSP{scenario_short[-3]}-{scenario_short[-2]}.{scenario_short[-1]}')
    print(f'\tRibes:\t\t{np.round(((ribes_q95[idx_short_scenario]-ribes_q05[idx_short_scenario])-(q95_ensemble[idx_short_scenario]-q05_ensemble[idx_short_scenario]))/(ribes_q95[idx_short_scenario]-ribes_q05[idx_short_scenario])*100).astype(int)}%')
    print(f'\tLiang:\t\t{np.round(((yongxiao_q95[idx_short_scenario]-yongxiao_q05[idx_short_scenario])-(q95_ensemble[idx_short_scenario]-q05_ensemble[idx_short_scenario]))/(yongxiao_q95[idx_short_scenario]-yongxiao_q05[idx_short_scenario])*100).astype(int)}%')
    print(f'\tTokarska:\t{np.round(((tokarska_q95[idx_short_scenario]-tokarska_q05[idx_short_scenario])-(q95_ensemble[idx_short_scenario]-q05_ensemble[idx_short_scenario]))/(tokarska_q95[idx_short_scenario]-tokarska_q05[idx_short_scenario])*100).astype(int)}%')
    print(f'\tIPCC WG1 AR6:\t{np.round(((ipcc_wg1_q95[idx_short_scenario]-ipcc_wg1_q05[idx_short_scenario])-(q95_ensemble[idx_short_scenario]-q05_ensemble[idx_short_scenario]))/(ipcc_wg1_q95[idx_short_scenario]-ipcc_wg1_q05[idx_short_scenario])*100).astype(int)}%')
    print(f'\tfair-calibrate v1.4.1:\t{np.round(((smith_fair_q95_141[idx_short_scenario]-smith_fair_q05_141[idx_short_scenario])-(q95_ensemble[idx_short_scenario]-q05_ensemble[idx_short_scenario]))/(smith_fair_q95_141[idx_short_scenario]-smith_fair_q05_141[idx_short_scenario])*100).astype(int)}%')
    print(f'\tfair-calibrate v1.4.0:\t{np.round(((smith_fair_q95_140[idx_short_scenario]-smith_fair_q05_140[idx_short_scenario])-(q95_ensemble[idx_short_scenario]-q05_ensemble[idx_short_scenario]))/(smith_fair_q95_140[idx_short_scenario]-smith_fair_q05_140[idx_short_scenario])*100).astype(int)}%')
    print(f'\tCMIP6 ESMs:\t{np.round(((q95_simulations[idx_short_scenario]-q05_simulations[idx_short_scenario])-(q95_ensemble[idx_short_scenario]-q05_ensemble[idx_short_scenario]))/(q95_simulations[idx_short_scenario]-q05_simulations[idx_short_scenario])*100).astype(int)}%')