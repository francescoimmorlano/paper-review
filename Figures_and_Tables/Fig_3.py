"""
Author: Francesco Immorlano

Script for reproducing Figure 3
"""

import sys
sys.path.insert(1, './..')
from lib import *

""" Load DNNs predictions """
predictions = read_tl_obs_predictions(n_BEST_datasets_per_model_scenario, compute_figures_tables)

""" Load CMIP6 ESMs simulations """
simulation_array = read_all_cmip6_simulations()

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
median_predictions_means_2081_2098 = np.zeros((len(short_scenarios_list),2098-2081+1))
median_simulations_means_2081_2098 = np.zeros((len(short_scenarios_list),2098-2081+1))
q05_predictions_means_2081_2098 = np.zeros((len(short_scenarios_list),2098-2081+1))
q05_simulations_means_2081_2098 = np.zeros((len(short_scenarios_list),2098-2081+1))
q95_predictions_means_2081_2098 = np.zeros((len(short_scenarios_list),2098-2081+1))
q95_simulations_means_2081_2098 = np.zeros((len(short_scenarios_list),2098-2081+1))
for short_scenario_idx, short_scenario in enumerate(short_scenarios_list):
    # DNNs predictions
    for i in range(2098-2081+1):
        median_predictions_means_2081_2098[short_scenario_idx,i] = np.median(np.ravel(warming_predictions_means_2081_2098[:,:,short_scenario_idx,i]))
        q05_predictions_means_2081_2098[short_scenario_idx,i] = np.percentile(warming_predictions_means_2081_2098[:,:,short_scenario_idx,i],5)
        q95_predictions_means_2081_2098[short_scenario_idx,i] = np.percentile(warming_predictions_means_2081_2098[:,:,short_scenario_idx,i],95)
    # CMIP6 ESMs simulations
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

q05_ensemble = np.round(q05_ensemble,2)
q95_ensemble = np.round(q95_ensemble,2)

print('\nDNNs ensemble warming in 2081–2098 wrt 1995–2014')
print(f'{np.round(q05_ensemble[0],2)} — {np.round(avg_median_ensemble[0],2)} — {np.round(q95_ensemble[0],2)}')
print(f'{np.round(q05_ensemble[1],2)} — {np.round(avg_median_ensemble[1],2)} — {np.round(q95_ensemble[1],2)}')
print(f'{np.round(q05_ensemble[2],2)} — {np.round(avg_median_ensemble[2],2)} — {np.round(q95_ensemble[2],2)}\n')

print('Uncertainty reduction')
for idx_short_scenario, scenario_short in enumerate(short_scenarios_list):
    print(f'SSP{scenario_short[-3]}-{scenario_short[-2]}.{scenario_short[-1]}')
    print(f'\tRibes:\t\t{np.round(((ribes_q95[idx_short_scenario]-ribes_q05[idx_short_scenario])-(q95_ensemble[idx_short_scenario]-q05_ensemble[idx_short_scenario]))/(ribes_q95[idx_short_scenario]-ribes_q05[idx_short_scenario])*100).astype(int)}%')
    print(f'\tLiang:\t\t{np.round(((yongxiao_q95[idx_short_scenario]-yongxiao_q05[idx_short_scenario])-(q95_ensemble[idx_short_scenario]-q05_ensemble[idx_short_scenario]))/(yongxiao_q95[idx_short_scenario]-yongxiao_q05[idx_short_scenario])*100).astype(int)}%')
    print(f'\tTokarska:\t{np.round(((tokarska_q95[idx_short_scenario]-tokarska_q05[idx_short_scenario])-(q95_ensemble[idx_short_scenario]-q05_ensemble[idx_short_scenario]))/(tokarska_q95[idx_short_scenario]-tokarska_q05[idx_short_scenario])*100).astype(int)}%')
    print(f'\tIPCC WG1 AR6:\t{np.round(((ipcc_wg1_q95[idx_short_scenario]-ipcc_wg1_q05[idx_short_scenario])-(q95_ensemble[idx_short_scenario]-q05_ensemble[idx_short_scenario]))/(ipcc_wg1_q95[idx_short_scenario]-ipcc_wg1_q05[idx_short_scenario])*100).astype(int)}%')


""" Plot """
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
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)

axes.set_ylabel('Surface Air Temperature 2081-2100 relative to '+str(refperiod_start)+'-'+str(refperiod_end)+' ($^\circ$C)',fontsize=17, labelpad=15)

legend = axes.legend(loc='upper left', shadow=False, fontsize='small',ncol=1,frameon=True,facecolor='white', framealpha=1,prop={'size':13})    

for yval in range(1,7):
    axes.plot([0.5,5.5],[yval-refperiod_conversion,yval-refperiod_conversion], color='black', dashes=(2, 10),linewidth=0.7)
ax2 = axes.twinx()
mn, mx = axes.get_ylim()
ax2.set_ylim(mn + refperiod_conversion, mx + refperiod_conversion) 
ax2.set_ylabel('relative to '+str(piperiod_start)+'-'+str(piperiod_end)+' ($^\circ$C)', fontsize=17, labelpad=15)

plt.yticks(fontsize=15)

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

