"""
Author: Francesco Immorlano

Script for reproducing images used in Figure S2
"""

import sys
sys.path.insert(1, '..')
from lib import *

# Load DNNs predictions after pre-training
predictions = read_first_train_predictions(compute_figures_tables_paper, 'First_Training_')

# Load CMIP6 simulations
simulations = read_all_cmip6_simulations()

# Convert from K to Celsius degrees
predictions_C = predictions - 273.15
simulations_C = simulations - 273.15

# Compute average global surface air temperature
annual_predictions_means_C = ((predictions_C * area_cella).sum(axis=(-1,-2)))/total_earth_area
annual_simulations_means_C = ((simulations_C * area_cella).sum(axis=(-1,-2)))/total_earth_area

''' Plot '''
fig, axs = plt.subplots(len(short_scenarios_list), 2, figsize=(40,30))
plt.subplots_adjust(wspace=0.1, hspace=0.4)
for scenario_short_idx, scenario_short in enumerate(short_scenarios_list):
    for model_idx, model in enumerate(models_list):
        axs[scenario_short_idx,0].plot(np.arange(1850, 2099), annual_simulations_means_C[model_idx,scenario_short_idx,:], linewidth=1, label=f'{model}')
        axs[scenario_short_idx,1].plot(np.arange(1850, 2099), annual_predictions_means_C[model_idx, scenario_short_idx,:], linewidth=2, linestyle='--', label=f'{model}')

    scenario = f'SSP{scenario_short[-3]}-{scenario_short[-2]}.{scenario_short[-1]}'
    
    axs[scenario_short_idx,0].set_title(f'CMIP6 — {scenario}', size=35, pad=15)
    axs[scenario_short_idx,1].set_title(f'DNNs — {scenario}', size=35, pad=15)

    axs[scenario_short_idx,0].tick_params(axis='both', which='major', labelsize=25)
    axs[scenario_short_idx,0].set_xticks([1850, 1900, 1950, 2000, 2050, 2098])

    axs[scenario_short_idx,1].tick_params(axis='both', which='major', labelsize=25)
    axs[scenario_short_idx,1].set_xticks([1850, 1900, 1950, 2000, 2050, 2098])
    
fig.add_subplot(1, 1, 1, frame_on=False)
plt.tick_params(labelcolor="none", bottom=False, left=False)
plt.xlabel('Years', fontsize=35, labelpad=50)
plt.ylabel('Near surface air temperature [°C]', fontsize=35, labelpad=50)
plt.text(x=0.11, y=0.9, s=f'A', fontweight='bold',
         fontsize=45, ha="center", transform=fig.transFigure)
plt.text(x=0.11, y=0.62, s=f'B', fontweight='bold',
         fontsize=45, ha="center", transform=fig.transFigure)
plt.text(x=0.11, y=0.34, s=f'C', fontweight='bold',
         fontsize=45, ha="center", transform=fig.transFigure)
plt.text(x=0.515, y=0.9, s=f'D', fontweight='bold',
         fontsize=45, ha="center", transform=fig.transFigure)
plt.text(x=0.515, y=0.62, s=f'E', fontweight='bold',
         fontsize=45, ha="center", transform=fig.transFigure)
plt.text(x=0.515, y=0.34, s=f'F', fontweight='bold',
         fontsize=45, ha="center", transform=fig.transFigure)
plt.savefig(f'Fig_S2.png', dpi=300, bbox_inches='tight')
plt.close()

