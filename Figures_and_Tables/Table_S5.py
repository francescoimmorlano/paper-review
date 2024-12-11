"""
Author: Francesco Immorlano

Script for reproducing Table comparison with OSCAR
"""

import sys
sys.path.insert(1, './..')
from lib import *

# Load predictions made by the DNNs after transfer learning on observative data
predictions = read_tl_obs_predictions(n_BEST_datasets_per_model_scenario, compute_figures_tables_paper, 'Transfer_learning_obs_2024-11-20_01-04-11')

# Compute baseline in 1995-2014
predictions_baseline = np.mean(predictions[:,:,:,1995-1979:2014-1979+1,:,:], axis=3) # (5, 22, 3 64, 128)

# Compute anomaly
warming_predictions = predictions[:,:,:,:,:,:] - predictions_baseline[:,:,:,np.newaxis,:,:] # (5, 22, 3, 120, 64, 128) 

# Add 0.85 to get baseline in 1850-1900
warming_predictions += refperiod_conversion

# Compute latitude-weighted spatial averages warming
warming_predictions_means = ((warming_predictions * area_cella).sum(axis=(-1,-2)))/total_earth_area # (5, 22, 3, 120)

stdev_2041_2050 = np.std(warming_predictions_means[:,:,:,2041-1979:2050-1979+1], axis=(0,1))
stdev_2091_2098 = np.std(warming_predictions_means[:,:,:,2091-1979:], axis=(0,1))

ensemble_2041_2050 = np.mean(warming_predictions_means[:,:,:,2041-1979:2050-1979+1], axis=(0,1))
ensemble_2091_2098 = np.mean(warming_predictions_means[:,:,:,2091-1979:], axis=(0,1))

avg_stdev_2041_2050 = np.mean(stdev_2041_2050, axis=1)
avg_stdev_2091_2098 = np.mean(stdev_2091_2098, axis=1)

avg_ensemble_2041_2050 = np.mean(ensemble_2041_2050, axis=1)
avg_ensemble_2091_2098 = np.mean(ensemble_2091_2098, axis=1)

print('2041-2050')
print(f'\tSSP2-4.5: {np.round(avg_ensemble_2041_2050[0],2)} +- {np.round(avg_stdev_2041_2050[0],2)}')
print(f'\tSSP3-7.0: {np.round(avg_ensemble_2041_2050[1],2)} +- {np.round(avg_stdev_2041_2050[1],2)}')
print(f'\tSSP5-8.5: {np.round(avg_ensemble_2041_2050[2],2)} +- {np.round(avg_stdev_2041_2050[2],2)}')

print('\n2091-2098')
print(f'\tSSP2-4.5: {np.round(avg_ensemble_2091_2098[0],2)} +- {np.round(avg_stdev_2091_2098[0],2)}')
print(f'\tSSP3-7.0: {np.round(avg_ensemble_2091_2098[1],2)} +- {np.round(avg_stdev_2091_2098[1],2)}')
print(f'\tSSP5-8.5: {np.round(avg_ensemble_2091_2098[2],2)} +- {np.round(avg_stdev_2091_2098[2],2)}')