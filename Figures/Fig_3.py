import os
from netCDF4 import Dataset
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

"""
Script for reproducing Figure 3
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
# Avg global surface temperature in 1995-2014
global_mean_temp_1995_2014 = 14.711500000000001
with open('../area_cella.csv', newline='') as csvfile:
    area_cella = np.genfromtxt(csvfile, delimiter=',')

n_BEST_datasets_per_model_scenario = 5

start_year_training = 1979
end_year_training = 2022
n_training_years = end_year_training-start_year_training+1
start_year_test = end_year_training+1
end_year_test = 2098
n_test_years = end_year_test-start_year_test+1

# settings for modern reference time period and proxy for pre-industrial time period
refperiod_start = 1995
refperiod_end   = 2014
piperiod_start  = 1850
piperiod_end    = 1900

# historical warming estimate based on cross-chapter box 2.3 (https://www.ipcc.ch/report/ar6/wg1/downloads/report/IPCC_AR6_WGI_Chapter02.pdf)
refperiod_conversion = 0.85

""" Load predictions made by the DNNs and CMIP6 ESMs simulations"""
predictions = np.zeros((n_BEST_datasets_per_model_scenario, len(models_list), len(short_scenarios_list), n_training_years+n_test_years, 64, 128))
simulation_array = np.zeros((len(models_list), len(short_scenarios_list), 249, 64, 128))
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
            model_test_set_predictions_filenames_list = os.listdir(TEST_SET_PREDICTIONS_DIRECTORY)
            model_test_set_predictions_filenames_list = [fn for fn in model_test_set_predictions_filenames_list if (fn.endswith('.csv'))]
            model_test_set_predictions_filenames_list.sort()
            model_test_set_prediction_array = np.zeros((n_test_years, 64, 128))
            for mp_idx, mp_filename in enumerate(model_test_set_predictions_filenames_list):
                if (not mp_filename.endswith('.csv')):
                    continue
                file = open(f'{TEST_SET_PREDICTIONS_DIRECTORY}/{mp_filename}')
                model_test_set_prediction_array[mp_idx,:,:] = np.loadtxt(file, delimiter=',')
                predictions[i,model_idx,scenario_idx,n_training_years:,:,:] = model_test_set_prediction_array[:,:,:]
        # CMIP6 ESMs simulations
        simulations_files_list = os.listdir(SIMULATIONS_DIRECTORY)
        simulations_files_list.sort()
        matching_simulations = [simulation_file for simulation_file in simulations_files_list if ((model in simulation_file and 'historical' in simulation_file)
                                                                                                or (model in simulation_file and scenario_short in simulation_file))]
        # maching_simuations[0] is the historical and matching_simulations[1] is the SSP simulation because of the sort operation
        # (for each model, the first simulation is the historical and then the SSP) 
        nc_historical_data = Dataset(f'{SIMULATIONS_DIRECTORY}/{matching_simulations[0]}', mode='r+', format='NETCDF3_CLASSIC')
        nc_ssp_data = Dataset(f'{SIMULATIONS_DIRECTORY}/{matching_simulations[1]}', mode='r+', format='NETCDF3_CLASSIC')
        n_historical_years = nc_historical_data[variable_short].shape[0]
        n_ssp_years = nc_ssp_data[variable_short].shape[0]
        n_lats = nc_ssp_data['lat'].shape[0]
        n_lons = nc_ssp_data['lon'].shape[0]
        simulation_array[model_idx, scenario_idx,:n_historical_years] = nc_historical_data[variable_short][:]
        if (n_ssp_years == 86):
            simulation_array[model_idx,scenario_idx,n_historical_years:] = nc_ssp_data[variable_short][:-2]
        elif (n_ssp_years == 85):
            simulation_array[model_idx,scenario_idx,n_historical_years:] = nc_ssp_data[variable_short][:-1]
        elif (n_ssp_years == 84):
            simulation_array[model_idx,scenario_idx,n_historical_years:] = nc_ssp_data[variable_short][:]
        nc_historical_data.close()
        nc_ssp_data.close()

# Convert from K to Celsius degrees
predictions_C = predictions - 273.15
simulation_array_C = simulation_array - 273.15

# Compute average global surface air temperature
annual_predictions_means = ((predictions_C * area_cella).sum(axis=(-1,-2)))/total_earth_area
annual_simulations_means = ((simulation_array_C * area_cella).sum(axis=(-1,-2)))/total_earth_area

# Compute warming wrt pre-industrial period
warming_predictions_means = annual_predictions_means - global_mean_temp_1995_2014
warming_simulations_means = annual_simulations_means - global_mean_temp_1995_2014

# Compute avg warming in 2081-2098
warming_predictions_means_2081_2098 = warming_predictions_means[:,:,:,2081-1979:]
warming_simulations_means_2081_2098 = warming_simulations_means[:,:,2081-1850:]

# Compute median, 5% and 95%
median_predictions_means_2081_2098 = np.zeros((len(short_scenarios_list),18))
median_simulations_means_2081_2098 = np.zeros((len(short_scenarios_list),18))
q05_predictions_means_2081_2098 = np.zeros((len(short_scenarios_list),18))
q05_simulations_means_2081_2098 = np.zeros((len(short_scenarios_list),18))
q95_predictions_means_2081_2098 = np.zeros((len(short_scenarios_list),18))
q95_simulations_means_2081_2098 = np.zeros((len(short_scenarios_list),18))
for short_scenario_idx, short_scenario in enumerate(short_scenarios_list):
    # DNNs predictions
    for i in range(18):
        median_predictions_means_2081_2098[short_scenario_idx,i] = np.median(np.ravel(warming_predictions_means_2081_2098[:,:,short_scenario_idx,i]))
        q05_predictions_means_2081_2098[short_scenario_idx,i] = np.percentile(warming_predictions_means_2081_2098[:,:,short_scenario_idx,i],5)
        q95_predictions_means_2081_2098[short_scenario_idx,i] = np.percentile(warming_predictions_means_2081_2098[:,:,short_scenario_idx,i],95)
    # CMIP6 ESMs simulations
    for i in range(18):
        median_simulations_means_2081_2098[short_scenario_idx,i] = np.median(np.ravel(warming_simulations_means_2081_2098[:,short_scenario_idx,i]))
        q05_simulations_means_2081_2098[short_scenario_idx,i] = np.percentile(warming_simulations_means_2081_2098[:,short_scenario_idx,i],5)
        q95_simulations_means_2081_2098[short_scenario_idx,i] = np.percentile(warming_simulations_means_2081_2098[:,short_scenario_idx,i],95)
 
# Compute avg median, 5% and 95% in 2081-2098
# DNNs predictions
avg_median_ensemble = median_predictions_means_2081_2098.mean(axis=1)
q05_ensemble = q05_predictions_means_2081_2098.mean(axis=1)
q95_ensemble = q95_predictions_means_2081_2098.mean(axis=1)
# CMIP6 ESMs simulations
avg_median_simulations = median_simulations_means_2081_2098.mean(axis=1)
q05_simulations = q05_simulations_means_2081_2098.mean(axis=1)
q95_simulations = q95_simulations_means_2081_2098.mean(axis=1)

"""
5%, median, 95% average temperature values in 2081–2100 wrt 1995-2014
by Ribes, Tokarska, Liang and IPCC WGI
"""
ribes_q05 = [1.23, 2.09, 2.43]
ribes_mean = [1.85, 2.81, 3.52]
ribes_q95 = [2.47, 3.52, 4.61]

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

""" Plot """
font = {'fontname':'Arial'}
fig, axes = plt.subplots(1, figsize=(10,10))
xpos = [1,2,3]
xlabel = ['SSP2-4.5','SSP3-7.0','SSP5-8.5']
barwidth = 0.17
barwidth_constrained = 0.2
shift_dist = 0

# left and right borders of bars
l1 = -2.2*barwidth
r1 = -1.4*barwidth
l2 = r1
r2 = -0.6*barwidth
l3 = r2
r3 = +0.2*barwidth
l4 = r3
r4 = +1*barwidth
l5 = r4
r5 = +1.8*barwidth
l6 = r5
r6 = +2.6*barwidth

""" SSP2-4.5 """
# NN ensemble
left = xpos[0]+l1
right = xpos[0]+r1
upper = q95_ensemble[0]
mid = avg_median_ensemble[0]
lower = q05_ensemble[0]
axes.fill([left,right,right,left],
               [lower,lower,upper,upper],
               facecolor='lightblue',edgecolor='black',linewidth=0.3,label='This work')
axes.plot([left+0.02,right-0.02],[mid,mid],color='white',linewidth=2)

# Liang
left = xpos[0]+l2
right = xpos[0]+r2
upper = yongxiao_q95[0]
mid = yongxiao_median[0]
lower = yongxiao_q05[0]
axes.fill([left,right,right,left],
               [lower,lower,upper,upper],
               facecolor='pink',edgecolor='black',linewidth=0.3,label='Liang et al.')
axes.plot([left+0.02,right-0.02],[mid,mid],color='white',linewidth=2)

# Ribes
left = xpos[0]+l3
right = xpos[0]+r3
upper = ribes_q95[0]
mid = ribes_mean[0]
lower = ribes_q05[0]
axes.fill([left,right,right,left],
               [lower,lower,upper,upper],
               facecolor='purple',edgecolor='black',linewidth=0.3,label='Ribes et al.')
axes.plot([left+0.02,right-0.02],[mid,mid],color='white',linewidth=2)

# Tokarska
left = xpos[0]+l4
right = xpos[0]+r4
upper = tokarska_q95[0]
mid = tokarska_median[0]
lower = tokarska_q05[0]
axes.fill([left,right,right,left],
               [lower,lower,upper,upper],
               facecolor='blue',edgecolor='black',linewidth=0.3,label='Tokarska et al.')
axes.plot([left+0.02,right-0.02],[mid,mid],color='white',linewidth=2)

# IPCC WGI
left = xpos[0]+l5
right = xpos[0]+r5
upper = ipcc_wg1_q95[0]
mid = ipcc_wg1_median[0]
lower = ipcc_wg1_q05[0]
axes.fill([left,right,right,left],
               [lower,lower,upper,upper],
               facecolor='orange',edgecolor='black',linewidth=0.3,label='IPCC WGI')
axes.plot([left+0.02,right-0.02],[mid,mid],color='white',linewidth=2)

# Unconstrained CMIP6
left = xpos[0]+l6
right = xpos[0]+r6
upper = q95_simulations[0]
mid = avg_median_simulations[0]
lower = q05_simulations[0]
axes.fill([left,right,right,left],
               [lower,lower,upper,upper],
               facecolor='red',edgecolor='black',linewidth=0.3,label='Unconstrained CMIP6')
axes.plot([left+0.02,right-0.02],[mid,mid],color='white',linewidth=2)

""" SSP3-7.0 """
# NN Ensemble
left = xpos[1]+l1
right = xpos[1]+r1
upper = q95_ensemble[1]
mid = avg_median_ensemble[1]
lower = q05_ensemble[1]
axes.fill([left,right,right,left],
               [lower,lower,upper,upper],
               facecolor='lightblue',edgecolor='black',linewidth=0.3,label='')
axes.plot([left+0.02,right-0.02],[mid,mid],color='white',linewidth=2)

# Lian
left = xpos[1]+l2
right = xpos[1]+r2
upper = yongxiao_q95[1]
mid = yongxiao_median[1]
lower = yongxiao_q05[1]
axes.fill([left,right,right,left],
               [lower,lower,upper,upper],
               facecolor='pink',edgecolor='black',linewidth=0.3,label='')
axes.plot([left+0.02,right-0.02],[mid,mid],color='white',linewidth=2)

# Ribes
left = xpos[1]+l3
right = xpos[1]+r3
upper = ribes_q95[1]
mid = ribes_mean[1]
lower = ribes_q05[1]
axes.fill([left,right,right,left],
               [lower,lower,upper,upper],
               facecolor='purple',edgecolor='black',linewidth=0.3,label='')
axes.plot([left+0.02,right-0.02],[mid,mid],color='white',linewidth=2)

# Tokarska
left = xpos[1]+l4
right = xpos[1]+r4
upper = tokarska_q95[1]
mid = tokarska_median[1]
lower = tokarska_q05[1]
axes.fill([left,right,right,left],
               [lower,lower,upper,upper],
               facecolor='blue',edgecolor='black',linewidth=0.3,label='')
axes.plot([left+0.02,right-0.02],[mid,mid],color='white',linewidth=2)

# IPCC WGI
left = xpos[1]+l5
right = xpos[1]+r5
upper = ipcc_wg1_q95[1]
mid = ipcc_wg1_median[1]
lower = ipcc_wg1_q05[1]
axes.fill([left,right,right,left],
               [lower,lower,upper,upper],
               facecolor='orange',edgecolor='black',linewidth=0.3,label='')
axes.plot([left+0.02,right-0.02],[mid,mid],color='white',linewidth=2)

# Unconstrained CMIP6
left = xpos[1]+l6
right = xpos[1]+r6
upper = q95_simulations[1]
mid = avg_median_simulations[1]
lower = q05_simulations[1]
axes.fill([left,right,right,left],
               [lower,lower,upper,upper],
               facecolor='red',edgecolor='black',linewidth=0.3,label='')
axes.plot([left+0.02,right-0.02],[mid,mid],color='white',linewidth=2)

""" SSP5-8.5 """
# NN ensemble
left = xpos[2]+l1
right = xpos[2]+r1
upper = q95_ensemble[2]
mid = avg_median_ensemble[2]
lower = q05_ensemble[2]
axes.fill([left,right,right,left],
               [lower,lower,upper,upper],
               facecolor='lightblue',edgecolor='black',linewidth=0.3,label='')
axes.plot([left+0.02,right-0.02],[mid,mid],color='white',linewidth=2)

# Lian
left = xpos[2]+l2
right = xpos[2]+r2
upper = yongxiao_q95[2]
mid = yongxiao_median[2]
lower = yongxiao_q05[2]
axes.fill([left,right,right,left],
               [lower,lower,upper,upper],
               facecolor='pink',edgecolor='black',linewidth=0.3,label='')
axes.plot([left+0.02,right-0.02],[mid,mid],color='white',linewidth=2)

# Ribes
left = xpos[2]+l3
right = xpos[2]+r3
upper = ribes_q95[2]
mid = ribes_mean[2]
lower = ribes_q05[2]
axes.fill([left,right,right,left],
               [lower,lower,upper,upper],
               facecolor='purple',edgecolor='black',linewidth=0.3,label='')
axes.plot([left+0.02,right-0.02],[mid,mid],color='white',linewidth=2)

# Tokarska
left = xpos[2]+l4
right = xpos[2]+r4
upper = tokarska_q95[2]
mid = tokarska_median[2]
lower = tokarska_q05[2]
axes.fill([left,right,right,left],
               [lower,lower,upper,upper],
               facecolor='blue',edgecolor='black',linewidth=0.3,label='')
axes.plot([left+0.02,right-0.02],[mid,mid],color='white',linewidth=2)

# IPCC WGI
left = xpos[2]+l5
right = xpos[2]+r5
upper = ipcc_wg1_q95[2]
mid = ipcc_wg1_median[2]
lower = ipcc_wg1_q05[2]
axes.fill([left,right,right,left],
               [lower,lower,upper,upper],
               facecolor='orange',edgecolor='black',linewidth=0.3,label='')
axes.plot([left+0.02,right-0.02],[mid,mid],color='white',linewidth=2)

# Unconstrained CMIP6
left = xpos[2]+l6
right = xpos[2]+r6
upper = q95_simulations[2]
mid = avg_median_simulations[2]
lower = q05_simulations[2]
axes.fill([left,right,right,left],
               [lower,lower,upper,upper],
               facecolor='red',edgecolor='black',linewidth=0.3,label='')
axes.plot([left+0.02,right-0.02],[mid,mid],color='white',linewidth=2)

axes.set_xlim([0.5,3.56])
axes.set_ylim([0,6])
axes.set_xticks(xpos)
axes.set_xticklabels(xlabel, rotation=30)
plt.xticks(fontname='Arial', fontsize=15)
plt.yticks(fontname='Arial', fontsize=15)

axes.set_ylabel('Surface Air Temperature 2081-2100 relative to '+str(refperiod_start)+'-'+str(refperiod_end)+' ($^\circ$C)',fontsize=17, labelpad=15, **font)

legend = axes.legend(loc='upper left', shadow=False, fontsize='small',ncol=1,frameon=True,facecolor='white', framealpha=1,prop={"family":"Arial", 'size':13})    

for yval in range(1,7):
    axes.plot([0.5,5.5],[yval-refperiod_conversion,yval-refperiod_conversion], color='black', dashes=(2, 10),linewidth=0.7)
ax2 = axes.twinx()
mn, mx = axes.get_ylim()
ax2.set_ylim(mn + refperiod_conversion, mx + refperiod_conversion) 
ax2.set_ylabel('relative to '+str(piperiod_start)+'-'+str(piperiod_end)+' ($^\circ$C)', fontsize=17, labelpad=15, **font)

plt.yticks(fontname='Arial', fontsize=15)

axes.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=True,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=True) # labels along the bottom edge are off

axes.tick_params(
    axis='y',          # changes apply to the y-axis
    which='both',      # both major and minor ticks are affected
    left=True,      # ticks along the bottom edge are off
    right=False,         # ticks along the top edge are off
    labelleft=True) # labels along the bottom edge are off

plt.savefig(f'Fig_3.png', bbox_extra_artists=(legend,), bbox_inches='tight', dpi=300)
plt.close()

