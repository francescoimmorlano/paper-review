"""
Author: Francesco Immorlano

Script for reproducing images used in Extended Data Figure 4
"""

import os
from netCDF4 import Dataset
import numpy as np
import pickle
import matplotlib.pyplot as plt

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

total_earth_area = 5.1009974e+14
with open('../area_cella.csv', newline='') as csvfile:
    area_cella = np.genfromtxt(csvfile, delimiter=',')

# settings for modern reference time period and proxy for pre-industrial time period
refperiod_start = 1995
refperiod_end   = 2014
piperiod_start  = 1850
piperiod_end    = 1900

# historical warming estimate based on cross-chapter box 2.3 (https://www.ipcc.ch/report/ar6/wg1/downloads/report/IPCC_AR6_WGI_Chapter02.pdf)
refperiod_conversion = 0.85

global_mean_temp_1850_1900 = 13.798588235294114
global_mean_temp_1995_2014 = 14.711500000000001

n_lat_points = 64
n_lon_points = 128

start_year_training = 1850
end_year_training = 2022
n_training_years = end_year_training-start_year_training+1
start_year_test = end_year_training+1
end_year_test = 2098
n_test_years = end_year_test-start_year_test+1

ROOT_DATA = '../Source_data'
SIMULATIONS_DIRECTORY = f'{ROOT_DATA}/CMIP6_data/near_surface_air_temperature/Annual_uniform_remapped'

""" Load CMIP6 ESMs simulations """
simulations = np.zeros((len(models_list), 3, end_year_test-start_year_training+1, n_lat_points, n_lon_points))
for idx_model, model in enumerate(models_list):
    for idx_scenario_short, scenario_short in enumerate(short_scenarios_list):
        scenario = f'SSP{scenario_short[-3]}-{scenario_short[-2]}.{scenario_short[-1]}'

        simulations_files_list = os.listdir(SIMULATIONS_DIRECTORY)
        simulations_files_list.sort()
        matching_simulations = [simulation_file for simulation_file in simulations_files_list if ((model in simulation_file and 'historical' in simulation_file)
                                                                                               or (model in simulation_file and scenario_short in simulation_file))]

        # maching_simulations[0] is the historical and matching_simulations[1] is the SSP simulation because of the sort operation
        # (for each model, the first simulation is the historical and then the SSP)  
        nc_historical_data = Dataset(f'{SIMULATIONS_DIRECTORY}/{matching_simulations[0]}', mode='r+', format='NETCDF3_CLASSIC')
        nc_ssp_data = Dataset(f'{SIMULATIONS_DIRECTORY}/{matching_simulations[1]}', mode='r+', format='NETCDF3_CLASSIC')

        n_historical_years = nc_historical_data[variable_short].shape[0]
        n_ssp_years = nc_ssp_data[variable_short].shape[0]
        simulations[idx_model,idx_scenario_short,:n_historical_years,:,:] = nc_historical_data[variable_short][:,:,:]
        if (n_ssp_years == 86):
            simulations[idx_model,idx_scenario_short,n_historical_years:,:,:] = nc_ssp_data[variable_short][:-2,:,:]
        elif (n_ssp_years == 85):
            simulations[idx_model,idx_scenario_short,n_historical_years:,:,:] = nc_ssp_data[variable_short][:-1,:,:]
        else:
            simulations[idx_model,idx_scenario_short,n_historical_years:,:,:] = nc_ssp_data[variable_short][:,:,:]
    nc_historical_data.close()
    nc_ssp_data.close()

ensemble_statistics = np.zeros((3,3,22))            # scenarios, (median, q05, q95), models
simulations_statistics = np.zeros((3,4,22))         # scenarios, (median, q05, q95, avg_taken_out), models
first_training_statistics = np.zeros((3,3,22))      # scenarios, (median, q05, q95), models

for idx_shuffle in range(22):
    models_list_remaining = models_list.copy()

    if idx_shuffle < 9:
        shuffle_number = f'0{idx_shuffle+1}'
    else:
        shuffle_number = f'{idx_shuffle+1}'

    model_taken_out = models_list[idx_shuffle]
    """ Remove current take-out model  """
    models_list_remaining.remove(model_taken_out)

    """ Load DNNs predictions after TL on the take-out model"""
    predictions_tl_on_simulations = np.zeros((len(models_list_remaining), len(short_scenarios_list), n_training_years+n_test_years, 64, 128)) # (21,3,249,64,128)
    pickle_in = open(f'{ROOT_DATA}/Transfer_Learning_on_Simulations/Predictions_shuffle-{shuffle_number}.pickle', 'rb')
    predictions_tl_on_simulations = pickle.load(pickle_in)
    pickle_in.close()

    remaining_models_idx = []
    for i in range(22):
        if i == idx_shuffle:
            continue
        remaining_models_idx.append(i)

    simulations_remaining = simulations[remaining_models_idx,:,:,:]

    simulation_takeout = simulations[idx_shuffle,:,:,:]

    # Convert from K to Celsius degrees
    predictions_C = predictions_tl_on_simulations - 273.15                  # (21,3,249,64,128)
    simulation_takeout_C = simulation_takeout - 273.15                      # (3,249,64,128)
    simulations_remaining_C = simulations_remaining - 273.15                # (21,3,249,64,128)

    # Compute average global surface air temperature
    annual_predictions_means = ((predictions_C * area_cella).sum(axis=(-1,-2)))/total_earth_area                                    # (21,3,249)
    annual_simulation_takeout_means = ((simulation_takeout_C * area_cella).sum(axis=(-1,-2)))/total_earth_area                      # (3,249)
    annual_simulations_remaining_means = ((simulations_remaining_C * area_cella).sum(axis=(-1,-2)))/total_earth_area                # (21,3,249)

    # Compute warming wrt pre-industrial period
    warming_predictions_means = annual_predictions_means - global_mean_temp_1995_2014                                       # (21,3,249)
    warming_simulation_takeout_means = annual_simulation_takeout_means - global_mean_temp_1995_2014                         # (3,249)
    warming_simulations_remaining_means = annual_simulations_remaining_means - global_mean_temp_1995_2014                   # (21,3,249)

    # Select warming values in 2081-2098
    warming_predictions_means_2081_2098 = warming_predictions_means[:,:,2081-1850:]                                     # (21,3,18)
    warming_simulation_takeout_means_2081_2098 = warming_simulation_takeout_means[:,2081-1850:]                         # (3,18)
    warming_simulations_remaining_means_2081_2098 = warming_simulations_remaining_means[:,:,2081-1850:]                 # (21,3,18)

    # Compute median, 5% and 95%
    median_predictions_means_2081_2098 = np.zeros((len(short_scenarios_list),18))
    median_simulations_remaining_2081_2098 = np.zeros((len(short_scenarios_list),18))

    q05_predictions_means_2081_2098 = np.zeros((len(short_scenarios_list),18))
    q05_simulations_remaining_means_2081_2098 = np.zeros((len(short_scenarios_list),18))

    q95_predictions_means_2081_2098 = np.zeros((len(short_scenarios_list),18))
    q95_simulations_remaining_means_2081_2098 = np.zeros((len(short_scenarios_list),18))

    for short_scenario_idx, short_scenario in enumerate(short_scenarios_list):
        for i in range(18):
            # DNNs predictions after TL on simulations
            median_predictions_means_2081_2098[short_scenario_idx,i] = np.median(warming_predictions_means_2081_2098[:,short_scenario_idx,i])
            q05_predictions_means_2081_2098[short_scenario_idx,i] = np.percentile(warming_predictions_means_2081_2098[:,short_scenario_idx,i],5)
            q95_predictions_means_2081_2098[short_scenario_idx,i] = np.percentile(warming_predictions_means_2081_2098[:,short_scenario_idx,i],95)
            # 21 remaining CMIP6 simulations
            median_simulations_remaining_2081_2098[short_scenario_idx,i] = np.median(warming_simulations_remaining_means_2081_2098[:,short_scenario_idx,i])
            q05_simulations_remaining_means_2081_2098[short_scenario_idx,i] = np.percentile(warming_simulations_remaining_means_2081_2098[:,short_scenario_idx,i],5)
            q95_simulations_remaining_means_2081_2098[short_scenario_idx,i] = np.percentile(warming_simulations_remaining_means_2081_2098[:,short_scenario_idx,i],95)
            
        ensemble_statistics[short_scenario_idx,0,idx_shuffle] = median_predictions_means_2081_2098[short_scenario_idx,:].mean()
        ensemble_statistics[short_scenario_idx,1,idx_shuffle] = q05_predictions_means_2081_2098[short_scenario_idx,:].mean()
        ensemble_statistics[short_scenario_idx,2,idx_shuffle] = q95_predictions_means_2081_2098[short_scenario_idx,:].mean()
        
        simulations_statistics[short_scenario_idx,0,idx_shuffle] = median_simulations_remaining_2081_2098[short_scenario_idx,:].mean()
        simulations_statistics[short_scenario_idx,1,idx_shuffle] = q05_simulations_remaining_means_2081_2098[short_scenario_idx,:].mean()
        simulations_statistics[short_scenario_idx,2,idx_shuffle] = q95_simulations_remaining_means_2081_2098[short_scenario_idx,:].mean()
        simulations_statistics[short_scenario_idx,3,idx_shuffle] = warming_simulation_takeout_means_2081_2098[short_scenario_idx,:].mean()

""" Plot """
fig, axes = plt.subplots(3, figsize=(35,30))
plt.rcParams.update({'font.sans-serif': 'Arial'})

xpos = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23]

size_x_y_ticks = 23
size_x_y_labels = 23
size_legend = 19
size_axis_title = 30
size_letters = 35

xlabel = ['',
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
        'UKESM1-0-LL',''
        ]

barwidth = 0.27
barwidth_constrained = 0.2
shift_dist = 0

# left and right borders of bars
l1 = -1 * barwidth
r1 = 0 * barwidth
l2 = r1
r2 = +1 * barwidth

for idx_short_scenario, short_scenario in enumerate(short_scenarios_list):
    
    scenario = f'SSP{short_scenario[-3]}-{short_scenario[-2]}.{short_scenario[-1]}'
    upper_lim = int(np.ceil(np.max([np.max(ensemble_statistics[idx_short_scenario,2,:]),
                        np.max(simulations_statistics[idx_short_scenario,2,:])])))
    for pos in xpos:
        if pos==0:
            continue
        if pos==23:
            break
        # 21 remaining CMIP6 + model_takeout median
        left = pos+l1
        right = pos+r1
        upper = simulations_statistics[idx_short_scenario,2,pos-1]
        median = simulations_statistics[idx_short_scenario,0,pos-1]
        avg_takeout = simulations_statistics[idx_short_scenario,3,pos-1]
        lower = simulations_statistics[idx_short_scenario,1,pos-1]
        if pos==1:
            red_label = '21 CMIP6 ESMs'
            axes[idx_short_scenario].fill([left,right,right,left],
                        [lower,lower,upper,upper],
                        facecolor='red',edgecolor='black',linewidth=0.3,label=red_label)
            axes[idx_short_scenario].plot([left+0.01,right-0.01],[avg_takeout,avg_takeout],color='black',linewidth=2,label='Taken out CMIP6 ESM')
            
        else:
            axes[idx_short_scenario].fill([left,right,right,left],
                        [lower,lower,upper,upper],
                        facecolor='red',edgecolor='black',linewidth=0.3)
            axes[idx_short_scenario].plot([left+0.01,right-0.01],[avg_takeout,avg_takeout],color='black',linewidth=2)
            
        axes[idx_short_scenario].plot([left+0.01,right-0.01],[median,median],color='white',linewidth=2)

        # DNNs predictions after TL on simulations
        left = pos+l2
        right = pos+r2
        upper = ensemble_statistics[idx_short_scenario,2,pos-1]
        median_ensemble = ensemble_statistics[idx_short_scenario,0,pos-1]
        lower = ensemble_statistics[idx_short_scenario,1,pos-1]
        if pos==1:
            axes[idx_short_scenario].fill([left,right,right,left],
                        [lower,lower,upper,upper],
                        facecolor='lightblue',edgecolor='black',linewidth=0.3, label='21 DNNs')
        else:
            axes[idx_short_scenario].fill([left,right,right,left],
                        [lower,lower,upper,upper],
                        facecolor='lightblue',edgecolor='black',linewidth=0.3)
        axes[idx_short_scenario].plot([left+0.01,right-0.01],[median_ensemble,median_ensemble],color='white',linewidth=2)
        axes[idx_short_scenario].plot([left+0.01,right-0.01],[avg_takeout,avg_takeout],color='black',linewidth=2, linestyle=':')

    axes[idx_short_scenario].set_xlim([0.5,3.56])
    axes[idx_short_scenario].set_ylim([0,upper_lim+0.5])
    axes[idx_short_scenario].set_xticks(xpos)
    axes[idx_short_scenario].set_xticklabels(xlabel, rotation=45)


    axes[idx_short_scenario].set_ylabel('Surface Air Temperature 2081-2098\nrelative to '+str(refperiod_start)+'-'+str(refperiod_end)+' ($^\circ$C)',fontsize=size_x_y_labels, labelpad=18)

    legend = axes[idx_short_scenario].legend(loc='upper left', shadow=False, fontsize='small',ncol=1,frameon=True,facecolor='white', framealpha=1,prop={'size':size_legend})    

    for yval in range(1,upper_lim+2):
        axes[idx_short_scenario].plot([0,23],[yval-refperiod_conversion,yval-refperiod_conversion], color='black', dashes=(2, 10),linewidth=0.6)
    ax2 = axes[idx_short_scenario].twinx()
    mn, mx = axes[idx_short_scenario].get_ylim()
    ax2.set_ylim(mn + refperiod_conversion, mx + refperiod_conversion) 
    ax2.set_ylabel('relative to '+str(piperiod_start)+'-'+str(piperiod_end)+' ($^\circ$C)', fontsize=size_x_y_labels, labelpad=18)
    ax2.tick_params(labelsize=size_x_y_ticks)

    axes[idx_short_scenario].tick_params(
        axis='x',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom=True,      # ticks along the bottom edge are off
        top=False,         # ticks along the top edge are off
        labelbottom=True,
        labelsize=size_x_y_ticks) # labels along the bottom edge are off

    axes[idx_short_scenario].tick_params(
        axis='y',          # changes apply to the y-axis
        which='both',      # both major and minor ticks are affected
        left=True,      # ticks along the bottom edge are off
        right=False,         # ticks along the top edge are off
        labelleft=True,
        labelsize=size_x_y_ticks) # labels along the bottom edge are off

    plt.title(f'Scenario {scenario}', size=25, pad=size_axis_title)

fig.tight_layout(pad=5)

plt.savefig(f'Ext_Data_Fig_4.png', bbox_inches='tight', dpi=300)
plt.close()

