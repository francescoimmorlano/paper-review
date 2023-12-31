import os
from netCDF4 import Dataset
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd

"""
Script for reproducing Figure 2
"""

models_list = [
        'ACCESS-CM2',
        'AWI-CM-1-1-MR',
        'BCC-CSM2-MR',
        'CAMS-CSM1-0',
        'CanESM5-CanOE',
        'CMCC-CM2-SR5',
        'CNRM-CM6-1',
        'CNRM-ESM2-1',
        'FGOALS-f3-L',
        'FGOALS-g3',
        'GFDL-ESM4',
        'IITM-ESM',
        'INM-CM4-8',
        'INM-CM5-0',
        'IPSL-CM6A-LR',
        'KACE-1-0-G',
        'MIROC6',
        'MPI-ESM1-2-LR',
        'MRI-ESM2-0',
        'NorESM2-MM',
        'TaiESM1',
        'UKESM1-0-LL'
        ]

short_scenarios_list = ['ssp245', 'ssp370', 'ssp585']
variable_short = 'tas'

ROOT_DATA = '../Source_data'
SIMULATIONS_DIRECTORY = f'{ROOT_DATA}/CMIP6_data/near_surface_air_temperature/Annual_uniform_remapped'
PATH_BEST_DATA = f'{ROOT_DATA}/BEST_data/BEST_regridded_annual_1979-2022.nc'
PATH_BEST_DATA_UNCERTAINTY = f'{ROOT_DATA}/BEST_data/Land_and_Ocean_global_average_annual.txt'

total_earth_area = 5.1009974e+14
# Avg global surface temperature in 1850-1900
global_mean_temp_1850_1900 = 13.798588235294114
with open('../area_cella.csv', newline='') as csvfile:
    area_cella = np.genfromtxt(csvfile, delimiter=',')

n_BEST_datasets_per_model_scenario = 5

start_year_training = 1979
end_year_training = 2022
n_training_years = end_year_training-start_year_training+1
start_year_test = end_year_training+1
end_year_test = 2098
n_test_years = end_year_test-start_year_test+1

""" Load predictions made by the DNNs after transfer learning on observational data """
predictions = np.zeros((n_BEST_datasets_per_model_scenario, len(models_list), len(short_scenarios_list), n_training_years+n_test_years, 64, 128))
for model_idx, model in tqdm(enumerate(models_list), total=len(models_list)):
    for scenario_idx, scenario_short in enumerate(short_scenarios_list):
        for i in range(n_BEST_datasets_per_model_scenario):
            TRAIN_SET_PREDICTIONS_DIRECTORY = f'{ROOT_DATA}/Transfer_Learning_on_Observations/Training_set_predictions/tas_{model}_{scenario_short}_{i+1}'
            TEST_SET_PREDICTIONS_DIRECTORY = f'{ROOT_DATA}/Transfer_Learning_on_Observations/Test_set_predictions/tas_{model}_{scenario_short}_{i+1}'
            # Training set predictions
            model_train_set_predictions_filenames_list = os.listdir(TRAIN_SET_PREDICTIONS_DIRECTORY)
            model_train_set_predictions_filenames_list = [fn for fn in model_train_set_predictions_filenames_list if (fn.endswith('.csv'))]
            model_train_set_predictions_filenames_list.sort()
            model_train_set_prediction_array = np.zeros((n_training_years, 64, 128))
            for mp_idx, mp_filename in enumerate(model_train_set_predictions_filenames_list):
                if (not mp_filename.endswith('.csv')):
                    continue
                file = open(f'{TRAIN_SET_PREDICTIONS_DIRECTORY}/{mp_filename}')
                model_train_set_prediction_array[mp_idx,:,:] = np.loadtxt(file, delimiter=',')
            predictions[i,model_idx,scenario_idx,:n_training_years,:,:] = model_train_set_prediction_array
            # Test set predictions
            model_predictions_filenames_list = os.listdir(TEST_SET_PREDICTIONS_DIRECTORY)
            model_predictions_filenames_list = [fn for fn in model_predictions_filenames_list if (fn.endswith('.csv'))]
            model_predictions_filenames_list.sort()
            # Even if end_year_test=2098, observational data are read until 2100 and then in array predictions only those until 2098 are saved.
            # This must be done to avoid errors
            model_prediction_array = np.zeros((n_test_years+2, 64, 128))
            for mp_idx, mp_filename in enumerate(model_predictions_filenames_list):
                if (not mp_filename.endswith('.csv')):
                    continue
                file = open(f'{TEST_SET_PREDICTIONS_DIRECTORY}/{mp_filename}')
                model_prediction_array[mp_idx,:,:] = np.loadtxt(file, delimiter=',')
            predictions[i,model_idx,scenario_idx,n_training_years:,:,:] = model_prediction_array[:-2]

""" Load CMIP6 simulations """
simulations_files_list = os.listdir(SIMULATIONS_DIRECTORY)
simulations_files_list.sort()
simulations = np.zeros((len(short_scenarios_list),len(models_list), 2098-1850+1, 64, 128))
for short_scenario_idx, short_scenario in enumerate(short_scenarios_list):
    for model_idx, model in enumerate(models_list):
        matching_simulations = [simulation_file for simulation_file in simulations_files_list if ((model in simulation_file and 'historical' in simulation_file)
                                                                                                or (model in simulation_file and short_scenario in simulation_file))]
        # maching_simuations[0] is the historical and matching_simulations[1] is the SSP simulation because of the sort operation
        # (for each model, the first simulation is the historical and then the SSP) 
        nc_historical_data = Dataset(f'{SIMULATIONS_DIRECTORY}/{matching_simulations[0]}', mode='r+', format='NETCDF3_CLASSIC')
        nc_ssp_data = Dataset(f'{SIMULATIONS_DIRECTORY}/{matching_simulations[1]}', mode='r+', format='NETCDF3_CLASSIC')
        n_historical_years = nc_historical_data[variable_short].shape[0]
        n_ssp_years = nc_ssp_data[variable_short].shape[0]
        n_lats = nc_ssp_data['lat'].shape[0]
        n_lons = nc_ssp_data['lon'].shape[0]
        simulations[short_scenario_idx,model_idx,:n_historical_years,:,:] = nc_historical_data[variable_short][:]
        if (n_ssp_years == 84):
            simulations[short_scenario_idx,model_idx,n_historical_years:,:,:] = nc_ssp_data[variable_short][:,:,:]
        elif (n_ssp_years == 85):
            simulations[short_scenario_idx,model_idx,n_historical_years:,:,:] = nc_ssp_data[variable_short][:-1,:,:]
        elif (n_ssp_years == 86):
            simulations[short_scenario_idx,model_idx,n_historical_years:,:,:] = nc_ssp_data[variable_short][:-2,:,:]
        nc_historical_data.close()
        nc_ssp_data.close()

""" Load BEST observational data """
nc_BEST_data = Dataset(f'{PATH_BEST_DATA}', mode='r+', format='NETCDF3_CLASSIC')
n_BEST_years = nc_BEST_data['st'].shape[0]
n_lats = nc_BEST_data['lat'].shape[0]
n_lons = nc_BEST_data['lon'].shape[0]
BEST_data_array = np.zeros((n_BEST_years, n_lats, n_lons))
BEST_data_array[:,:,:] = nc_BEST_data['st'][:,:,:]
nc_BEST_data.close()

""" Load BEST observational data uncertainty """
uncertainty_df = pd.read_csv(f'{PATH_BEST_DATA_UNCERTAINTY}', header=None, delim_whitespace=True)
annual_uncertainties_list = list(uncertainty_df[uncertainty_df[0].between(start_year_training, end_year_training)][2])

annual_uncertainties_list.append(0.045) # 2019
annual_uncertainties_list.append(0.045) # 2020
annual_uncertainties_list.append(0.045) # 2021
if end_year_training == 2022:
    annual_uncertainties_list.append(0.045) # 2022

# Convert from K to Celsius degrees
simulations_C = simulations - 273.15
predictions_C = predictions - 273.15
BEST_data_array_C = BEST_data_array - 273.15

# Compute average global surface air temperature
annual_simulations_means = ((simulations_C * area_cella).sum(axis=(-1,-2)))/total_earth_area
annual_predictions_means = ((predictions_C * area_cella).sum(axis=(-1,-2)))/total_earth_area
annual_BEST_data_means = ((BEST_data_array_C * area_cella).sum(axis=(-1,-2)))/total_earth_area

# Compute average across CMIP6 ESMs and DNNs predictions
ensemble_simulations_means = np.mean(annual_simulations_means, axis=(1))
ensemble_predictions_means = np.mean(annual_predictions_means, axis=(0,1))

# Compute warming wrt pre-industrial period
warming_ensemble_simulations_means = ensemble_simulations_means - global_mean_temp_1850_1900
warming_ensemble_predictions_means = ensemble_predictions_means - global_mean_temp_1850_1900
warming_annual_simulations_means = annual_simulations_means - global_mean_temp_1850_1900
warming_annual_predictions_means = annual_predictions_means - global_mean_temp_1850_1900
warming_annual_BEST_data_means = annual_BEST_data_means - global_mean_temp_1850_1900

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
font = {'fontname':'Arial'}
fig, axs = plt.subplots(len(short_scenarios_list), figsize=(16,18))
plt.subplots_adjust(hspace=0.4)
for scenario_short_idx, scenario_short in enumerate(short_scenarios_list):
    # BEST
    axs[scenario_short_idx].scatter(np.arange(start_year_training, end_year_training+1), warming_annual_BEST_data_means, linewidth=1, label=f'BEST observational data', color='black', zorder=7)
    # BEST uncertainty shading
    axs[scenario_short_idx].fill_between(np.arange(start_year_training, end_year_training+1), warming_annual_BEST_data_means-annual_uncertainties_list, warming_annual_BEST_data_means+annual_uncertainties_list, facecolor='#FF5733', zorder=8)
    # training set shading
    axs[scenario_short_idx].fill_between(np.arange(start_year_training, end_year_training+1), -2, warming_ensemble_predictions_means[0,:end_year_training-start_year_training+1], color='red', alpha=0.1, zorder = 0)
    # DNNs predictions ensemble
    axs[scenario_short_idx].plot(np.arange(start_year_training, end_year_test+1), warming_ensemble_predictions_means[scenario_short_idx,:], linewidth=4, label=f'DNNs multi-model mean', color='#1d73b3', zorder=6)
    # predictions 5-95% range uncertainty shading
    axs[scenario_short_idx].fill_between(np.arange(start_year_training, end_year_test+1), q05_predictions[scenario_short_idx,:], q95_predictions[scenario_short_idx,:], facecolor='#7EFDFF', zorder=3)
    # CMIP6 ensemble
    axs[scenario_short_idx].plot(np.arange(start_year_training, end_year_test+1), warming_ensemble_simulations_means[scenario_short_idx,1979-1850:], linewidth=4, label=f'CMIP6 multi-model mean', color='#F56113', zorder=5)
    # CMIP6 5-95% range uncertainty shading
    axs[scenario_short_idx].fill_between(np.arange(start_year_training, end_year_test+1), q05_simulations[scenario_short_idx,1979-1850:], q95_simulations[scenario_short_idx,1979-1850:], facecolor='#FFD67E', zorder=1)

for scenario_short_idx, scenario_short in enumerate(short_scenarios_list):
    scenario = f'SSP{scenario_short[-3]}-{scenario_short[-2]}.{scenario_short[-1]}'
    axs[scenario_short_idx].set_title(f'Scenario {scenario} — Temperature in 2098: {round(warming_ensemble_predictions_means[scenario_short_idx,-1],2)} °C [{np.round(q05_predictions[scenario_short_idx,-1],2)}–{np.round(q95_predictions[scenario_short_idx,-1],2)} °C]',
                                      size=22, **font)
    axs[scenario_short_idx].axhline(y=warming_ensemble_predictions_means[scenario_short_idx,-1],xmin=0,xmax=0.955,c="black",linewidth=1.5, zorder=9, linestyle='--')
    axs[scenario_short_idx].annotate('%0.2f' % warming_ensemble_predictions_means[scenario_short_idx,-1], xy=(-0.005, warming_ensemble_predictions_means[scenario_short_idx,-1]+0.2), xytext=(8, 0), 
                    xycoords=('axes fraction', 'data'), textcoords='offset points', size=17, fontweight='bold', **font)
    axs[scenario_short_idx].set_ylim([-1, 10])
    if scenario_short_idx == 0:
        axs[scenario_short_idx].hlines(y=2, xmin=1979, xmax=2051, color= 'r', zorder=9)
        axs[scenario_short_idx].vlines(x=2051, ymin=-2, ymax=warming_ensemble_predictions_means[scenario_short_idx,2051-1979], color= 'r', zorder=9)
        axs[scenario_short_idx].set_xticks([1979, 2000, 2022, 2040, 2051, 2060, 2080, 2098])
    if scenario_short_idx == 1:
        axs[scenario_short_idx].hlines(y=2, xmin=1979, xmax=2044, color= 'r', zorder=9)
        axs[scenario_short_idx].vlines(x=2044, ymin=-2, ymax=warming_ensemble_predictions_means[scenario_short_idx,2044-1979], color= 'r', zorder=9)
        axs[scenario_short_idx].set_xticks([1979, 2000, 2022, 2044, 2060, 2080, 2098])
    if scenario_short_idx == 2:
        axs[scenario_short_idx].hlines(y=2, xmin=1979, xmax=2042, color= 'r', zorder=9)
        axs[scenario_short_idx].vlines(x=2042, ymin=-2, ymax=warming_ensemble_predictions_means[scenario_short_idx,2043-1979], color= 'r', zorder=9)
        axs[scenario_short_idx].set_xticks([1979, 2000, 2022, 2042, 2060, 2080, 2098])
    plt.xlim(left=1979)
    plt.sca(axs[scenario_short_idx])
    plt.yticks(fontname='Arial', fontsize=17)
    plt.xticks(fontname='Arial', fontsize=17)
    axs[scenario_short_idx].legend(loc='upper right', prop={"family":"Arial",'size':14})

fig.add_subplot(1, 1, 1, frame_on=False)
plt.tick_params(labelcolor="none", bottom=False, left=False)
plt.xlabel('Years', fontsize=20, labelpad=30, **font)
plt.ylabel('Global average near surface air temperature anomaly [°C]\nBase period: 1850–1900', fontsize=22, labelpad=30, **font)

plt.savefig(f'Fig_2.png', dpi=300, bbox_inches='tight')
plt.close()
