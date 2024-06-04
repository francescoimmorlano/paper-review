"""
Author: Francesco Immorlano

Script for parititon uncertainty according to https://doi.org/10.1029/2022EF002963
"""

import sys
sys.path.insert(1, './..')
from lib import *

""" Load CMIP6 simulations """
simulations = read_all_cmip6_simulations()

""" Load predictions made by the DNNs after transfer learning on observational data """
predictions = read_tl_obs_predictions(n_BEST_datasets_per_model_scenario, compute_figures_tables)

""" Compute 10-year running mean by grid point """
running_avg = np.zeros((len(short_scenarios_list), len(models_list), 240, 64, 128))
for idx_scenario_short, scenario_short in enumerate(short_scenarios_list):
    for idx_model, model in enumerate(models_list):
        for rows in range(simulations.shape[3]):
            for columns in range(simulations.shape[4]):
                running_avg[idx_scenario_short, idx_model, :, rows, columns] = moving_average(simulations[idx_model, idx_scenario_short, :, rows, columns])

""" Compute spatial average per year """
annual_running_avg = ((running_avg * area_cella).sum(axis=(-1,-2)))/total_earth_area
predictions_avg = ((predictions * area_cella).sum(axis=(-1,-2)))/total_earth_area

""" Partition uncertainty """
F, F_M, F_V, F_S, T, M, V, S = uncertainty_partition(annual_running_avg[:,:,1979-1855:], 1979, 2094, 1980, 1999)
F_predictions, F_M_predictions, F_V_predictions, F_S_predictions, T_predictions, M_predictions, V_predictions, S_predictions = uncertainty_partition_predictions(predictions_avg[:,:,:,:2094-1979+1], 1979, 2094, 1980, 1999)

V_arr = np.ones((2094-2000+1)) * V
V_arr_predictions = np.ones((2094-2000+1)) * V_predictions

""" Fig. S11 """
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14,5))
ax1.plot(np.arange(2000,2094+1), V_arr, color='red', linewidth=2.0, linestyle='--', label='Internal variability')
ax1.plot(np.arange(2000,2094+1), M[2000-1979:], color='blue', linewidth=3.0, linestyle='--', label='Model uncertainty')
ax1.plot(np.arange(2000,2094+1), S[2000-1979:], color='green', linewidth=3.0, linestyle='--', label='Scenario uncertainty')
ax1.plot(np.arange(2000,2094+1), T[2000-1979:], color='black', linewidth=3.0, linestyle='--', label='Total uncertainty')
ax1.set_title('CMIP6')
x_ticks = [2000,2020,2040,2060,2080,2094]
ax1.set_xticks(x_ticks)
ax1.tick_params(axis='x', labelsize=11)
ax1.tick_params(axis='y', labelsize=11)
ax1.set_xlabel('Years', size=12)
ax1.set_ylabel('Uncertainty', size=12)
ax1.legend(loc='upper left')
ax1.grid()
ax2.plot(np.arange(2000,2094+1), V_arr_predictions, color='red', linewidth=2.0, linestyle='--', label='Internal variability')
ax2.plot(np.arange(2000,2094+1), M_predictions[2000-1979:], color='blue', linewidth=3.0, linestyle='--', label='Model uncertainty')
ax2.plot(np.arange(2000,2094+1), S_predictions[2000-1979:], color='green', linewidth=3.0, linestyle='--', label='Scenario uncertainty')
ax2.plot(np.arange(2000,2094+1), T_predictions[2000-1979:], color='black', linewidth=3.0, linestyle='--', label='Total uncertainty')
ax2.set_title('Deep Neural Networks')
x_ticks = [2000,2020,2040,2060,2080,2094]
ax2.set_xticks(x_ticks)
ax2.tick_params(axis='x', labelsize=11)
ax2.tick_params(axis='y', labelsize=11)
ax2.set_xlabel('Years', size=12)
ax2.legend(loc='upper left')
ax2.grid()
plt.text(x=0.08, y=0.92, s='A', fontweight='bold',
        fontsize=20, transform=fig.transFigure)
plt.text(x=0.505, y=0.92, s='B', fontweight='bold',
        fontsize=20, transform=fig.transFigure)
plt.savefig('Fig_S11.png', dpi=300, bbox_inches='tight')

""" Fig. S12 """
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14,5))
ax1.plot(np.arange(2000,2094+1), F_V[2000-1979:], color='red', linewidth=2.0, linestyle='--', label='Internal variability')
ax1.plot(np.arange(2000,2094+1), F_M[2000-1979:], color='blue', linewidth=3.0, linestyle='--', label='Model uncertainty')
ax1.plot(np.arange(2000,2094+1), F_S[2000-1979:], color='green', linewidth=3.0, linestyle='--', label='Scenario uncertainty')
ax1.plot(np.arange(2000,2094+1), F[2000-1979:], color='black', linewidth=3.0, linestyle='--', label='Total uncertainty')
ax1.set_title('CMIP6')
ax1.set_xlabel('Years', size=12)
ax1.set_ylabel('Fractional uncertainty', size=12)
x_ticks = [2000,2020,2040,2060,2080,2094]
ax1.set_xticks(ticks=x_ticks)
ax1.tick_params(axis='x', labelsize=11)
ax1.tick_params(axis='y', labelsize=11)
ax1.legend(loc='upper left')
ax1.grid()
ax2.plot(np.arange(2000,2094+1), F_V_predictions[2000-1979:], color='red', linewidth=2.0, linestyle='--', label='Internal variability')
ax2.plot(np.arange(2000,2094+1), F_M_predictions[2000-1979:], color='blue', linewidth=3.0, linestyle='--', label='Model uncertainty')
ax2.plot(np.arange(2000,2094+1), F_S_predictions[2000-1979:], color='green', linewidth=3.0, linestyle='--', label='Scenario uncertainty')
ax2.plot(np.arange(2000,2094+1), F_predictions[2000-1979:], color='black', linewidth=3.0, linestyle='--', label='Total uncertainty')
ax2.set_title('Deep Neural Networks')
ax2.set_xlabel('Years', size=12)
x_ticks = [2000,2020,2040,2060,2080,2094]
ax2.set_xticks(x_ticks)
ax2.tick_params(axis='x', labelsize=11)
ax2.tick_params(axis='y', labelsize=11)
ax2.legend(loc='upper left')
ax2.grid()
plt.text(x=0.08, y=0.92, s='A', fontweight='bold',
        fontsize=20, transform=fig.transFigure)
plt.text(x=0.505, y=0.92, s='B', fontweight='bold',
        fontsize=20, transform=fig.transFigure)
plt.savefig('Fig_S12.png', dpi=300, bbox_inches='tight')

""" Fig. S13 """
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14,5))
ax1.plot(np.arange(2000,2094+1), F_M[2000-1979:], color='blue', linewidth=3.0, linestyle='--', label='CMIP6')
ax1.plot(np.arange(2000,2094+1), F_M_predictions[2000-1979:], color='blue', linewidth=3.0, linestyle='-', label='DNNs')
ax1.set_title('Fractional model uncertainty')
ax1.legend(loc='lower right')
x_ticks = [2000,2020,2040,2060,2080,2094]
ax1.set_xticks(x_ticks)
ax1.tick_params(axis='x', labelsize=11)
ax1.tick_params(axis='y', labelsize=11)
ax1.set_xlabel('Years', size=12)
ax1.grid()
ax2.plot(np.arange(2000,2094+1), M[2000-1979:], color='blue', linewidth=3.0, linestyle='--', label='CMIP6')
ax2.plot(np.arange(2000,2094+1), M_predictions[2000-1979:], color='blue', linewidth=3.0, linestyle='-', label='DNNs')
ax2.legend(loc='lower right')
ax2.set_title('Model uncertainty')
x_ticks = [2000,2020,2040,2060,2080,2094]
ax2.set_xticks(x_ticks)
ax2.tick_params(axis='x', labelsize=11)
ax2.tick_params(axis='y', labelsize=11)
ax2.set_xlabel('Years', size=12)
ax2.grid()
plt.text(x=0.08, y=0.92, s='A', fontweight='bold',
        fontsize=20, transform=fig.transFigure)
plt.text(x=0.505, y=0.92, s='B', fontweight='bold',
        fontsize=20, transform=fig.transFigure)
plt.savefig('Fig_S13.png', dpi=300, bbox_inches='tight')

M_near_term = M[2030-1979:2039-1979+1]
M_predictions_near_term = M_predictions[2030-1979:2039-1979+1]
print('\nTable S6 — Model uncertainty reduction in the near term (2030–2039):')
print(np.round(np.mean((M_near_term - M_predictions_near_term) / M_near_term * 100),1))

M_long_term = M[2085-1979:2094-1979+1]
M_predictions_long_term = M_predictions[2085-1979:2094-1979+1]
print('\nTable S7 — Model uncertainty reduction in the long term (2085–2094):')
print(np.round(np.mean((M_long_term - M_predictions_long_term) / M_long_term * 100),1))