"""
Author: Francesco Immorlano

Script for reproducing Figure S6
"""

import sys
sys.path.insert(1, './..')
from lib import *

AM_families = True
baseline_years = '1995-2014'

# Load smoothed CMIP6 simulations
pickle_in = open(f'{PATH_SMOOTHED_CMIP6_SIMULATIONS_DIRECTORY}/smooth_splines_dof-{n_dof}_CMIP6_warming_{baseline_years}.pickle','rb')
smooth_warming_simulations = pickle.load(pickle_in)

ensemble_statistics = np.zeros((len(short_scenarios_list),3,22))            # scenarios, (median, q05, q95), models
simulations_statistics = np.zeros((len(short_scenarios_list),4,22))         # scenarios, (median, q05, q95, avg_taken_out), models
first_training_statistics = np.zeros((len(short_scenarios_list),3,22))      # scenarios, (median, q05, q95), models

for idx_take_out, (take_out, take_out_family) in enumerate(atmospheric_model_families_dict.items()):
    models_list_remaining = models_list.copy()

    predictions_tl_on_simulations = read_tl_simulations_predictions_shuffle(idx_take_out, compute_figures_tables_paper, None, AM_families, take_out_family)

    if take_out_family:
        take_out_family_idx = []
        for member in take_out_family:
            take_out_family_idx.append(models_list.index(member))

    remaining_models_idx = []
    for i in range(22):
        if i == idx_take_out:
            continue
        if take_out_family and i in take_out_family_idx:
            continue
        remaining_models_idx.append(i)

    smooth_warming_simulations_remaining = smooth_warming_simulations[remaining_models_idx,:,:,:]
    smooth_warming_simulation_takeout = smooth_warming_simulations[idx_take_out,:,:,:]

    # Convert from K to Celsius degrees 
    predictions_C = predictions_tl_on_simulations - 273.15

    # Compute climatologies in 1995-2014 
    predictions_baseline = np.mean(predictions_tl_on_simulations[:,:,1995-1850:2014-1850+1,:,:], axis=2) # (21, 3, 64, 128)

    # Compute warming wrt 1995-2014 
    warming_predictions = predictions_tl_on_simulations[:,:,:,:,:] - predictions_baseline[:,:, np.newaxis,:,:] # (21, 3, 249, 64, 128) # np.newaxis is needed to add a new axis and let the difference be broadcasted

    # Compute spatial avg warming
    warming_predictions_means = ((warming_predictions * area_cella).sum(axis=(-1,-2)))/total_earth_area # (21, 3, 249)
    smooth_warming_simulation_takeout_means = ((smooth_warming_simulation_takeout * area_cella).sum(axis=(-1,-2)))/total_earth_area # (3, 249)
    smooth_warming_simulations_remaining_means = ((smooth_warming_simulations_remaining * area_cella).sum(axis=(-1,-2)))/total_earth_area # (21, 3, 249)

    # Select warming values in 2081-2098
    warming_predictions_means_2081_2098 = warming_predictions_means[:,:,2081-1850:]
    smooth_warming_simulation_takeout_means_2081_2098 = smooth_warming_simulation_takeout_means[:,2081-1850:]
    smooth_warming_simulations_remaining_means_2081_2098 = smooth_warming_simulations_remaining_means[:,:,2081-1850:]

    # Compute median, 5% and 95%
    median_predictions_means_2081_2098 = np.zeros((len(short_scenarios_list),2098-2081+1))
    median_simulations_remaining_2081_2098 = np.zeros((len(short_scenarios_list),2098-2081+1))

    q05_predictions_means_2081_2098 = np.zeros((len(short_scenarios_list),2098-2081+1))
    q05_simulations_remaining_means_2081_2098 = np.zeros((len(short_scenarios_list),2098-2081+1))

    q95_predictions_means_2081_2098 = np.zeros((len(short_scenarios_list),2098-2081+1))
    q95_simulations_remaining_means_2081_2098 = np.zeros((len(short_scenarios_list),2098-2081+1))
    for short_scenario_idx, short_scenario in enumerate(short_scenarios_list):
        for i in range(2098-2081+1):
            # DNNs predictions after TL on simulations
            median_predictions_means_2081_2098[short_scenario_idx,i] = np.median(warming_predictions_means_2081_2098[:,short_scenario_idx,i])
            q05_predictions_means_2081_2098[short_scenario_idx,i] = np.percentile(warming_predictions_means_2081_2098[:,short_scenario_idx,i],5)
            q95_predictions_means_2081_2098[short_scenario_idx,i] = np.percentile(warming_predictions_means_2081_2098[:,short_scenario_idx,i],95)
            # 21 remaining CMIP6 simulations
            median_simulations_remaining_2081_2098[short_scenario_idx,i] = np.median(smooth_warming_simulations_remaining_means_2081_2098[:,short_scenario_idx,i])
            q05_simulations_remaining_means_2081_2098[short_scenario_idx,i] = np.percentile(smooth_warming_simulations_remaining_means_2081_2098[:,short_scenario_idx,i],5)
            q95_simulations_remaining_means_2081_2098[short_scenario_idx,i] = np.percentile(smooth_warming_simulations_remaining_means_2081_2098[:,short_scenario_idx,i],95)
            
        ensemble_statistics[short_scenario_idx,0,idx_take_out] = median_predictions_means_2081_2098[short_scenario_idx,:].mean()
        ensemble_statistics[short_scenario_idx,1,idx_take_out] = q05_predictions_means_2081_2098[short_scenario_idx,:].mean()
        ensemble_statistics[short_scenario_idx,2,idx_take_out] = q95_predictions_means_2081_2098[short_scenario_idx,:].mean()

        simulations_statistics[short_scenario_idx,0,idx_take_out] = median_simulations_remaining_2081_2098[short_scenario_idx,:].mean()
        simulations_statistics[short_scenario_idx,1,idx_take_out] = q05_simulations_remaining_means_2081_2098[short_scenario_idx,:].mean()
        simulations_statistics[short_scenario_idx,2,idx_take_out] = q95_simulations_remaining_means_2081_2098[short_scenario_idx,:].mean()
        simulations_statistics[short_scenario_idx,3,idx_take_out] = smooth_warming_simulation_takeout_means_2081_2098[short_scenario_idx,:].mean()

''' Plot '''
fig, axes = plt.subplots(3, figsize=(35,30))
fig.subplots_adjust(hspace=0.6)
xpos = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23]

size_x_y_ticks = 23
size_x_y_labels = 23
size_legend = 19
size_axis_title = 30
size_letters = 40


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
            red_label = 'CMIP6 ESMs'
            axes[idx_short_scenario].fill([left,right,right,left],
                        [lower,lower,upper,upper],
                        facecolor='#00FFFF',edgecolor='black',linewidth=0.3,label=red_label)
            axes[idx_short_scenario].plot([left+0.01,right-0.01],[avg_takeout,avg_takeout],color='black',linewidth=3,label='Taken-out CMIP6 ESM')
            
        else:
            axes[idx_short_scenario].fill([left,right,right,left],
                        [lower,lower,upper,upper],
                        facecolor='#00FFFF',edgecolor='black',linewidth=0.3)
            axes[idx_short_scenario].plot([left+0.01,right-0.01],[avg_takeout,avg_takeout],color='black',linewidth=3)
            
        axes[idx_short_scenario].plot([left+0.01,right-0.01],[median,median],color='red',linewidth=3)

        # DNNs predictions after TL on simulations
        left = pos+l2
        right = pos+r2
        upper = ensemble_statistics[idx_short_scenario,2,pos-1]
        median_ensemble = ensemble_statistics[idx_short_scenario,0,pos-1]
        lower = ensemble_statistics[idx_short_scenario,1,pos-1]
        if pos==1:
            axes[idx_short_scenario].fill([left,right,right,left],
                        [lower,lower,upper,upper],
                        facecolor='#007FEA',edgecolor='black',linewidth=0.3, label='DNNs after TL')
        else:
            axes[idx_short_scenario].fill([left,right,right,left],
                        [lower,lower,upper,upper],
                        facecolor='#007FEA',edgecolor='black',linewidth=0.3)
        axes[idx_short_scenario].plot([left+0.01,right-0.01],[median_ensemble,median_ensemble],color='white',linewidth=3)
        axes[idx_short_scenario].plot([left+0.01,right-0.01],[avg_takeout,avg_takeout],color='black',linewidth=3, linestyle=':')

    axes[idx_short_scenario].set_xlim([0.5,3.56])
    axes[idx_short_scenario].set_ylim([0,upper_lim+0.5])
    axes[idx_short_scenario].set_xticks(xpos)
    axes[idx_short_scenario].set_xticklabels(xlabel, rotation=45)

    axes[idx_short_scenario].set_ylabel('Surface Air Temperature 2081-2098\nrelative to '+str(refperiod_start)+'-'+str(refperiod_end)+' ($^\circ$C)',fontsize=size_x_y_labels, labelpad=18)

    if idx_short_scenario == 0:
        legend = axes[idx_short_scenario].legend(loc='upper left', shadow=False, fontsize='small',ncol=1,frameon=True,facecolor='white', framealpha=1,prop={'size':size_legend})    

    for yval in range(1,upper_lim+2):
        axes[idx_short_scenario].plot([0,23],[yval-refperiod_conversion,yval-refperiod_conversion], color='black', dashes=(2, 10),linewidth=0.7)
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

    plt.title(f'Scenario {scenario}', size=size_axis_title, pad=20)

plt.text(x=0.08, y=0.89, s='A', fontweight='bold',
        fontsize=size_letters, transform=fig.transFigure)
plt.text(x=0.08, y=0.60, s='B', fontweight='bold',
        fontsize=size_letters, transform=fig.transFigure)
plt.text(x=0.08, y=0.305, s='C', fontweight='bold',
        fontsize=size_letters, transform=fig.transFigure)

plt.savefig('Fig_S6.png', bbox_inches='tight', dpi=300)
plt.close()