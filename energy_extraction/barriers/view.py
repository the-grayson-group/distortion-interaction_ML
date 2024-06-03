'''
#####################################################################################
#                                     view.py                                       #
#                                                                                   #
#   Code to save figures of the spread of targets for each dataset. Code requires   #
#   the user to suggest the path. This will require changing to accomodate the      #
#   users paths.
#                                                                                   #
#####################################################################################
#                                       Authors                                     #
#                                                                                   #
#--------------------------------- Samuel G. Espley --------------------------------#
#                                                                                   #
#####################################################################################
'''
import math
import numpy as np
import pandas as pd
from mplcursors import cursor
import matplotlib.pyplot as plt

dataset = input('Which dataset: da, ma, tt or mal? ')

def _load_data(dataset):
    '''
    Function to load all the data for the three datasets.

    Arguments:
    dataset (String): A string of either da, ma, mal, and tt to load the dataset.

    Returns:
    lst (List): A list of DataFrames containing the energies.
    '''

    pth = '../../feature_selection/_f_selection/'
    if dataset == 'da':
        df1 = pd.read_pickle(f'{pth}da/manual_da_endo.pkl')
        df2 = pd.read_pickle(f'{pth}da/manual_da_exo.pkl')
        lst = [df1, df2]
        for df in lst:
            _remove_features(df)

    elif dataset == 'ma':
        df1 = pd.read_pickle(f'{pth}ma/manual_ma.pkl')
        lst = [df1]
        for df in lst:
            _remove_features(df)
            df['sum_distortion_energies_am1'] = df['distortion_energy_1_am1'] + df['distortion_energy_0_am1']
    
    elif dataset == 'tt':
        df1 = pd.read_pickle(f'{pth}/tt/manual_tt_gas.pkl')
        df2 = pd.read_pickle(f'{pth}/tt/manual_tt_spe.pkl')
        df3 = pd.read_pickle(f'{pth}/tt/manual_tt_solvent.pkl')
        lst = [df1, df2, df3]
        for df in lst:
            _remove_features(df)

    elif dataset == 'mal':
        df1 = pd.read_pickle(f'{pth}mal/manual_mal.pkl')
        lst = [df1]
        for df in lst:
            _remove_features(df)
    
    return lst

def _remove_features(df):
    '''
    Function to get only the required features.
    
    Arguments:
    df (DataFrame): A DataFrame containing all the features.

    '''
    remove = []
    for col in df.columns:
        if 'barrier' in col or 'energy' in col or 'energies' in col or 'reaction_number' in col or 'am1' in col:
            pass
        else:
            remove.append(col)
    for col in remove:
        del df[col]

def _get_hists(lst):

    '''
    Function to get histograms of all the data in lst.

    Arguments:
    lst (List): A list of DataFrames containing the energies.
    '''

    col_d = {
        'e_barrier_dft': '$\Delta$E', 
        'q_barrier_dft':'$\Delta$G$^{‡}$', 
        'sum_distortion_energies_dft':'$\Sigma_{Distortion Energy}$', 
        'interaction_energies_dft':'Interaction Energies',
    }


    if dataset == 'ma':
        col_d.update({'distortion_energy_0_dft':'Nucleophile Distortion Energy', 'distortion_energy_1_dft':'Michael Acceptor Distortion Energy'})
        for col in list(lst[0].columns):
            if col == 'distortion_contributions' or col == 'reaction_number':
                pass
            elif '_am1' in col:
                pass
            else:
                sqm = str(col).replace('_dft', '_am1')
                if col == 'interaction_energies_dft':
                    lst[0][sqm] = lst[0][sqm]*-1
                    lst[0][col] = lst[0][col]*-1
                fig, ax = plt.subplots(1, 1, figsize=(8, 8))
                lst[0][sqm].hist(ax=ax,
                            bins=50,
                            alpha=0.8,
                            histtype='step',
                            fill=True,
                            edgecolor='darkblue',
                            color='cornflowerblue',
                            linewidth=3,
                            zorder=10
                            )
                lst[0][col].hist(ax=ax,
                            bins=50,
                            histtype='step',
                            fill=True,
                            edgecolor='maroon',
                            color='lightcoral',
                            linewidth=3,
                            zorder=1
                                )
                plt.ylabel('Count', fontsize=22, fontweight='semibold')
                plt.xlabel(f'{col_d[col]}\n/ kcal mol⁻¹', fontsize=22, fontweight='semibold')
                plt.xticks(fontsize=16)
                plt.yticks(fontsize=16)
                plt.legend(['AM1', 'DFT'], prop={'weight':'semibold', 'size':17}, loc='upper right', shadow=True)
                fig.tight_layout()
                plt.savefig(f'ma/figures/ma_{col}.png', dpi=800)
                # plt.show()

    elif dataset == 'mal':
        col_d.update({'distortion_energy_0_dft':'Nucleophile Distortion Energy', 'distortion_energy_1_dft':'Michael Acceptor Distortion Energy'})
        for col in list(lst[0].columns):
            if col == 'distortion_contributions' or col == 'reaction_number':
                pass
            elif '_am1' in col:
                pass
            else:
                sqm = str(col).replace('_dft', '_am1')
                if col == 'interaction_energies_dft':
                    lst[0][sqm] = lst[0][sqm]*-1
                    lst[0][col] = lst[0][col]*-1
                fig, ax = plt.subplots(1, 1, figsize=(8, 8))
                lst[0][sqm].hist(ax=ax,
                            bins=30,
                            alpha=0.8,
                            histtype='step',
                            fill=True,
                            edgecolor='darkblue',
                            color='cornflowerblue',
                            linewidth=3,
                            zorder=10
                            )
                lst[0][col].hist(ax=ax,
                            bins=30,
                            histtype='step',
                            fill=True,
                            edgecolor='maroon',
                            color='lightcoral',
                            linewidth=3,
                            zorder=1
                                )
                # ax.set_title('Michael Addition Malonate - ' + col_d[col], fontsize=15)
                #ax.set_xlabel('Reaction Barrier / $kcal mol^{-1}$')
                plt.ylabel('Count', fontsize=22, fontweight='semibold')
                plt.xlabel(f'{col_d[col]}\n/ kcal mol⁻¹', fontsize=22, fontweight='semibold')
                plt.xticks(fontsize=16)
                plt.yticks(fontsize=16)
                plt.legend(['AM1', 'DFT'], prop={'weight':'semibold', 'size':17}, loc='upper right', shadow=True)
                fig.tight_layout()
                plt.savefig(f'mal/figures/mal_{col}.png', dpi=800)
                # plt.show()

    elif dataset == 'da':
        col_d.update({'distortion_energy_di_dft':'Diene Distortion Energy', 'distortion_energy_dp_dft':'Dienophile Distortion Energy'})
        for col in list(lst[0].columns):
            if col == 'distortion_contributions' or col == 'reaction_number':
                pass
            elif '_am1' in col:
                pass
            else: 
                sqm = str(col).replace('_dft', '_am1')
                if col == 'interaction_energies_dft':
                    lst[0][sqm] = lst[0][sqm]*-1
                    lst[0][col] = lst[0][col]*-1
                fig, ax = plt.subplots(1, 1, figsize=(8, 8))
                lst[0][sqm].hist(ax=ax,
                            bins=50,
                            alpha=0.8,
                            histtype='step',
                            fill=True,
                            edgecolor='darkblue',
                            color='cornflowerblue',
                            linewidth=3,
                            zorder=10
                            )
                lst[0][col].hist(ax=ax,
                            bins=50,
                            histtype='step',
                            fill=True,
                            edgecolor='maroon',
                            color='lightcoral',
                            linewidth=3,
                            zorder=1
                                )
                # lst[1][sqm].hist(ax=ax[1],
                #             #bins=50,
                #             histtype='step',
                #             fill=True,
                #             edgecolor='darkblue',
                #             color='cornflowerblue'
                #             )
                # lst[1][col].hist(ax=ax[1],
                #             #bins=50,
                #             alpha=0.8,
                #             histtype='step',
                #             fill=True,
                #             edgecolor='maroon',
                #             color='lightcoral'
                #                 )
                # fig.suptitle('Diels-Alder - ' + col_d[col])
                # ax[0].set_title('Endo')
                # ax[1].set_title('Exo')
                plt.ylabel('Count', fontsize=22, fontweight='semibold')
                plt.xlabel(f'{col_d[col]}\n/ kcal mol⁻¹', fontsize=22, fontweight='semibold')
                plt.xticks(fontsize=16)
                plt.yticks(fontsize=16)
                plt.legend(['AM1', 'DFT'], prop={'weight':'semibold', 'size':17}, loc='upper right', shadow=True)
                fig.tight_layout()
                plt.savefig(f'da/figures/da_{col}.png', dpi=800)
                # plt.show()
    
    elif dataset == 'tt':
         col_d.update({'distortion_energy_1_dft':'Dipole Distortion Energy', 'distortion_energy_2_dft':'Dipolarophile Distortion Energy'})
         for col in list(lst[0].columns):
            if col == 'distortion_contributions' or col == 'reaction_number':
                pass
            elif '_am1' in col:
                pass
            else:
                sqm = str(col).replace('_dft', '_am1')
                if col == 'interaction_energies_dft':
                    lst[1][sqm] = lst[1][sqm]*-1
                    lst[1][col] = lst[1][col]*-1
                fig, ax = plt.subplots(1, 1, figsize=(8, 8))
                lst[1][sqm].hist(ax=ax,
                            bins=50,
                            alpha=0.8,
                            histtype='step',
                            fill=True,
                            edgecolor='darkblue',
                            color='cornflowerblue',
                            linewidth=3,
                            zorder=10
                            )
                lst[1][col].hist(ax=ax,
                            bins=50,
                            histtype='step',
                            fill=True,
                            edgecolor='maroon',
                            color='lightcoral',
                            linewidth=3,
                            zorder=1
                            )
                # lst[1][sqm].hist(ax=ax[1],
                #             bins=50,
                #             histtype='step',
                #             fill=True,
                #             edgecolor='darkblue',
                #             color='cornflowerblue',
                #             linewidth=3
                #             )
                # lst[1][col].hist(ax=ax[1],
                #             bins=50,
                #             alpha=0.8,
                #             histtype='step',
                #             fill=True,
                #             edgecolor='maroon',
                #             color='lightcoral',
                #             linewidth=3
                #             )
                # lst[2][sqm].hist(ax=ax[2],
                #             bins=50,
                #             histtype='step',
                #             fill=True,
                #             edgecolor='darkblue',
                #             color='cornflowerblue',
                #             linewidth=3
                #             )
                # lst[2][col].hist(ax=ax[2],
                #             bins=50,
                #             alpha=0.8,
                #             histtype='step',
                #             fill=True,
                #             edgecolor='maroon',
                #             color='lightcoral',
                #             linewidth=3
                #             )
                # ax[0].set_title('Gas')
                # ax[1].set_title('SPE')
                # ax[2].set_title('Solvent')
                plt.ylabel('Count', fontsize=22, fontweight='semibold')
                plt.xlabel(f'{col_d[col]}\n/ kcal mol⁻¹', fontsize=22, fontweight='semibold')
                plt.xticks(fontsize=16)
                plt.yticks(fontsize=16)
                plt.legend(['AM1', 'DFT'], prop={'weight':'semibold', 'size':17}, loc='upper right', shadow=True)
                fig.tight_layout()
                # plt.savefig(f'tt_gas_{col}.png', dpi=800)
                plt.savefig(f'tt_solvent_{col}.png', dpi=800)
                # plt.show()

def func(cr, comb):
    '''
    Function for adding the highlighting aspect for each graph to look at rxn id's.
    '''
    @cr.connect("add")
    def _(sel):
            sel.annotation.set_text((pd.DataFrame(comb).loc[sel.index, ['reaction_number']].to_string()))
            sel.annotation.set_fontweight('bold')
            sel.annotation.get_bbox_patch().set_boxstyle('round')
            sel.annotation.get_bbox_patch().set(fc='white')

def _get_metrics(comb):

    '''
    Function to get RMSE, MAE and Pearson r for the data in a DataFrame.

    Arguments:
    comb (DataFrame): A combined DataFrame of energies for AM1 and DFT.
    '''

    text_d = {}
    metric_d = {}
    if dataset == 'ma':
        lst =  ['e_barrier_', 'q_barrier_', 'sum_distortion_energies_', 'interaction_energies_', 'distortion_energy_0_', 'distortion_energy_1_']
    elif dataset == 'mal':
        lst =  ['e_barrier_', 'q_barrier_', 'sum_distortion_energies_', 'interaction_energies_', 'distortion_energy_0_', 'distortion_energy_1_']
    elif dataset == 'da':
        lst = ['e_barrier_', 'q_barrier_', 'sum_distortion_energies_', 'interaction_energies_', 'distortion_energy_di_', 'distortion_energy_dp_']
    elif dataset == 'tt':
        lst = ['e_barrier_', 'q_barrier_', 'sum_distortion_energies_', 'interaction_energies_', 'distortion_energy_1_', 'distortion_energy_2_']
    
    for col in lst:
        rmse = np.round(math.sqrt(np.square(list(comb[col+'am1'] - comb[col+'dft'])).mean()), 3)
        mae = np.round(np.mean(np.abs(list(comb[col+'am1'] - comb[col+'dft']))), 3)
        pearson_r = np.round(np.corrcoef(comb[col+'am1'], comb[col+'dft'])[0,1], 4)

        t_rmse = 'RMSE: ' + str(np.round(math.sqrt(np.square(list(comb[col+'am1'] - comb[col+'dft'])).mean()), 3)) + ' $kcalmol^{-1}$'
        t_mae = 'MAE: ' + str(np.round(np.mean(np.abs(list(comb[col+'am1'] - comb[col+'dft']))), 3)) + ' $kcalmol^{-1}$'
        t_pearson_r = 'r: ' + str(np.round(np.corrcoef(comb[col+'am1'], comb[col+'dft'])[0,1], 4))
        text = t_rmse+'\n'+'\n'+t_mae+'\n'+'\n'+t_pearson_r

        if col == 'e_barrier_':
            text_d['e'] = text
            metric_d['e'] = {'rmse':rmse, 'mae':mae, 'pearson_r':pearson_r}
        elif col == 'q_barrier_':
            text_d['q'] = text
            metric_d['q'] = {'rmse':rmse, 'mae':mae, 'pearson_r':pearson_r}
        elif col == 'sum_distortion_energies_':
            text_d['d'] = text
            metric_d['d'] = {'rmse':rmse, 'mae':mae, 'pearson_r':pearson_r}
        elif col == 'interaction_energies_':
            text_d['i'] = text
            metric_d['i'] = {'rmse':rmse, 'mae':mae, 'pearson_r':pearson_r}
        elif col == 'distortion_energy_0_':
            text_d['di_0'] = text
            metric_d['di_0'] = {'rmse':rmse, 'mae':mae, 'pearson_r':pearson_r}
        elif col == 'distortion_energy_1_':
            text_d['di_1'] = text
            metric_d['di_1'] = {'rmse':rmse, 'mae':mae, 'pearson_r':pearson_r}
        elif col == 'distortion_energy_2_':
            text_d['di_2'] = text
            metric_d['di_2'] = {'rmse':rmse, 'mae':mae, 'pearson_r':pearson_r}
        elif col == 'distortion_energy_di_':
            text_d['di_di'] = text
            metric_d['di_di'] = {'rmse':rmse, 'mae':mae, 'pearson_r':pearson_r}
        elif col == 'distortion_energy_dp_':
            text_d['di_dp'] = text
            metric_d['di_dp'] = {'rmse':rmse, 'mae':mae, 'pearson_r':pearson_r}

    return metric_d, text_d

def _get_scats(lst):

    '''
    Function to get scatter graphs of all the data in lst.

    Arguments:
    lst (List): A list of DataFrames containing the energies.
    '''

    edge_c = 'darkblue'
    inner_c = 'cornflowerblue'

    if dataset == 'ma':
        comb = lst[0]
        metric_d, text_d = _get_metrics(comb)
        fig, ax = plt.subplots(2, 6, figsize=(18, 10), gridspec_kw={
                        #    'width_ratios': [4, 1, 4, 1],
                        #    'height_ratios': [1, 1],
                       'wspace': 0.3,
                       'hspace': 0.4})
        for axis in [ax[0,1], ax[0,3], ax[1,1], ax[1,3], ax[0,5], ax[1,5]]:
                axis.axis('off')
                axis.set_xlim(-1,2)
                axis.set_ylim(-1,1)
        fig.suptitle('Michael Addition')
        plt.subplots_adjust(left=0.03,
                    bottom=0.1,
                    right=0.98,
                    top=0.9,
                    wspace=0.01,
                    hspace=0.4)
        ax_0 = comb.plot(kind='scatter', ax=ax[0,0], x='q_barrier_am1', y='q_barrier_dft', s=40, edgecolor=edge_c, color=inner_c)
        ax_1 = comb.plot(kind='scatter', ax=ax[0,2], x='e_barrier_am1', y='e_barrier_dft', s=40, edgecolor=edge_c, color=inner_c)
        ax_2 = comb.plot(kind='scatter', ax=ax[1,0], x='sum_distortion_energies_am1',y='sum_distortion_energies_dft', s=40, edgecolor=edge_c, color=inner_c)
        ax_3 = comb.plot(kind='scatter', ax=ax[1,2], x='interaction_energies_am1',y='interaction_energies_dft', s=40, edgecolor=edge_c, color=inner_c)
        ax_4 = comb.plot(kind='scatter', ax=ax[0,4], x='distortion_energy_0_am1',y='distortion_energy_0_dft', s=40, edgecolor=edge_c, color=inner_c)
        ax_5 = comb.plot(kind='scatter', ax=ax[1,4], x='distortion_energy_1_am1',y='distortion_energy_1_dft', s=40, edgecolor=edge_c, color=inner_c)
        
        ax[0,0].set_title(r'$\Delta$G')
        ax[0,2].set_title(r'$\Delta$E')
        ax[1,0].set_title(r'$\Delta$ $E_{Distortion}$')
        ax[1,2].set_title(r'$\Delta$ $E_{Interaction}$')

        ax[0,0].set_xlabel('AM1 Reaction Barrier / $kcal mol^{-1}$')
        ax[0,0].set_ylabel('DFT Reaction Barrier / $kcal mol^{-1}$')
        ax[0,2].set_xlabel('AM1 Reaction Barrier / $kcal mol^{-1}$')
        ax[0,2].set_ylabel('DFT Reaction Barrier / $kcal mol^{-1}$')
        ax[1,0].set_xlabel('AM1 $\Sigma_{Distortion Energy}$ / $kcal mol^{-1}$')
        ax[1,0].set_ylabel('DFT $\Sigma_{Distortion Energy}$ / $kcal mol^{-1}$')  
        ax[1,2].set_xlabel('AM1 Interaction Energy / $kcal mol^{-1}$')
        ax[1,2].set_ylabel('DFT Interaction Energy / $kcal mol^{-1}$')  
        ax[1,4].set_xlabel('AM1 Distortion Energy \nMichael Acceptor / $kcal mol^{-1}$')
        ax[1,4].set_ylabel('DFT Distortion Energy \nMichael Acceptor / $kcal mol^{-1}$')  
        ax[0,4].set_xlabel('AM1 Distortion Energy \nNucleophile / $kcal mol^{-1}$')
        ax[0,4].set_ylabel('DFT Distortion Energy \nNucleophile / $kcal mol^{-1}$')  

        ax[0,1].text(0, 0, text_d['q'], verticalalignment='center', horizontalalignment='center')
        ax[0,3].text(0, 0, text_d['e'], verticalalignment='center', horizontalalignment='center')
        ax[1,1].text(0, 0, text_d['d'], verticalalignment='center', horizontalalignment='center')
        ax[1,3].text(0, 0, text_d['i'], verticalalignment='center', horizontalalignment='center')
        ax[0,5].text(0, 0, text_d['di_0'], verticalalignment='center', horizontalalignment='center')
        ax[1,5].text(0, 0, text_d['di_1'], verticalalignment='center', horizontalalignment='center')

        for axes in [ax[0,0], ax[0,2], ax[1,0], ax[1,2], ax[0,4], ax[1,4]]:
            cr = cursor(axes, hover=2, multiple=True)
            func(cr, comb)
        plt.show()
    
    if dataset == 'mal':
        comb = lst[0]
        metric_d, text_d = _get_metrics(comb)
        fig, ax = plt.subplots(2, 6, figsize=(18, 10), gridspec_kw={
                        #    'width_ratios': [4, 1, 4, 1],
                        #    'height_ratios': [1, 1],
                       'wspace': 0.3,
                       'hspace': 0.4})
        for axis in [ax[0,1], ax[0,3], ax[1,1], ax[1,3], ax[0,5], ax[1,5]]:
                axis.axis('off')
                axis.set_xlim(-1,2)
                axis.set_ylim(-1,1)
        fig.suptitle('Michael Addition - Malonate')
        plt.subplots_adjust(left=0.03,
                    bottom=0.1,
                    right=0.98,
                    top=0.9,
                    wspace=0.01,
                    hspace=0.4)
        ax_0 = comb.plot(kind='scatter', ax=ax[0,0], x='q_barrier_am1', y='q_barrier_dft', s=40, edgecolor=edge_c, color=inner_c)
        ax_1 = comb.plot(kind='scatter', ax=ax[0,2], x='e_barrier_am1', y='e_barrier_dft', s=40, edgecolor=edge_c, color=inner_c)
        ax_2 = comb.plot(kind='scatter', ax=ax[1,0], x='sum_distortion_energies_am1',y='sum_distortion_energies_dft', s=40, edgecolor=edge_c, color=inner_c)
        ax_3 = comb.plot(kind='scatter', ax=ax[1,2], x='interaction_energies_am1',y='interaction_energies_dft', s=40, edgecolor=edge_c, color=inner_c)
        ax_4 = comb.plot(kind='scatter', ax=ax[0,4], x='distortion_energy_0_am1',y='distortion_energy_0_dft', s=40, edgecolor=edge_c, color=inner_c)
        ax_5 = comb.plot(kind='scatter', ax=ax[1,4], x='distortion_energy_1_am1',y='distortion_energy_1_dft', s=40, edgecolor=edge_c, color=inner_c)
        
        ax[0,0].set_title(r'$\Delta$G')
        ax[0,2].set_title(r'$\Delta$E')
        ax[1,0].set_title(r'$\Delta$ $E_{Distortion}$')
        ax[1,2].set_title(r'$\Delta$ $E_{Interaction}$')

        ax[0,0].set_xlabel('AM1 Reaction Barrier / $kcal mol^{-1}$')
        ax[0,0].set_ylabel('DFT Reaction Barrier / $kcal mol^{-1}$')
        ax[0,2].set_xlabel('AM1 Reaction Barrier / $kcal mol^{-1}$')
        ax[0,2].set_ylabel('DFT Reaction Barrier / $kcal mol^{-1}$')
        ax[1,0].set_xlabel('AM1 $\Sigma_{Distortion Energy}$ / $kcal mol^{-1}$')
        ax[1,0].set_ylabel('DFT $\Sigma_{Distortion Energy}$ / $kcal mol^{-1}$')  
        ax[1,2].set_xlabel('AM1 Interaction Energy / $kcal mol^{-1}$')
        ax[1,2].set_ylabel('DFT Interaction Energy / $kcal mol^{-1}$')  
        ax[1,4].set_xlabel('AM1 Distortion Energy \nMichael Acceptor / $kcal mol^{-1}$')
        ax[1,4].set_ylabel('DFT Distortion Energy \nMichael Acceptor / $kcal mol^{-1}$')  
        ax[0,4].set_xlabel('AM1 Distortion Energy \nNucleophile / $kcal mol^{-1}$')
        ax[0,4].set_ylabel('DFT Distortion Energy \nNucleophile / $kcal mol^{-1}$')  

        ax[0,1].text(0, 0, text_d['q'], verticalalignment='center', horizontalalignment='center')
        ax[0,3].text(0, 0, text_d['e'], verticalalignment='center', horizontalalignment='center')
        ax[1,1].text(0, 0, text_d['d'], verticalalignment='center', horizontalalignment='center')
        ax[1,3].text(0, 0, text_d['i'], verticalalignment='center', horizontalalignment='center')
        ax[0,5].text(0, 0, text_d['di_0'], verticalalignment='center', horizontalalignment='center')
        ax[1,5].text(0, 0, text_d['di_1'], verticalalignment='center', horizontalalignment='center')

        for axes in [ax[0,0], ax[0,2], ax[1,0], ax[1,2], ax[0,4], ax[1,4]]:
            cr = cursor(axes, hover=2, multiple=True)
            func(cr, comb)
        plt.show()

    elif dataset == 'da':
        comb_endo = lst[0]
        comb_exo = lst[1]
        
        for comb, title in zip([comb_endo, comb_exo], ['Endo', 'Exo']):
            metric_d, text_d = _get_metrics(comb)
            fig, ax = plt.subplots(2, 6, figsize=(18, 10), gridspec_kw={
                        #    'width_ratios': [4, 1, 4, 1],
                        #    'height_ratios': [1, 1],
                       'wspace': 0.2,
                       'hspace': 0.4})
            for axis in [ax[0,1], ax[0,3], ax[1,1], ax[1,3], ax[0,5], ax[1,5]]:
                axis.axis('off')
                axis.set_xlim(-1,2)
                axis.set_ylim(-1,1)
            fig.suptitle(title)
            plt.subplots_adjust(left=0.03,
                    bottom=0.1,
                    right=0.98,
                    top=0.9,
                    wspace=0.01,
                    hspace=0.4
                    )
            ax_0 = comb.plot(kind='scatter', ax=ax[0,0], x='q_barrier_am1', y='q_barrier_dft', s=40, edgecolor=edge_c, color=inner_c)
            ax_1 = comb.plot(kind='scatter', ax=ax[0,2], x='e_barrier_am1', y='e_barrier_dft', s=40, edgecolor=edge_c, color=inner_c)
            ax_2 = comb.plot(kind='scatter', ax=ax[1,0], x='sum_distortion_energies_am1', y='sum_distortion_energies_dft', s=40, edgecolor=edge_c, color=inner_c)
            ax_3 = comb.plot(kind='scatter', ax=ax[1,2], x='interaction_energies_am1', y='interaction_energies_dft', s=40, edgecolor=edge_c, color=inner_c)
            ax_4 = comb.plot(kind='scatter', ax=ax[0,4], x='distortion_energy_di_am1', y='distortion_energy_di_dft', s=40, edgecolor=edge_c, color=inner_c)
            ax_5 = comb.plot(kind='scatter', ax=ax[1,4], x='distortion_energy_dp_am1', y='distortion_energy_dp_dft', s=40, edgecolor=edge_c, color=inner_c)

            ax[0,0].set_title(r'$\Delta$G')
            ax[0,2].set_title(r'$\Delta$E')
            ax[1,0].set_title(r'$\Delta$ $E_{Distortion}$')
            ax[1,2].set_title(r'$\Delta$ $E_{Interaction}$')

            ax[0,0].set_xlabel('AM1 Reaction Barrier / $kcal mol^{-1}$')
            ax[0,0].set_ylabel('DFT Reaction Barrier / $kcal mol^{-1}$')
            ax[0,2].set_xlabel('AM1 Reaction Barrier / $kcal mol^{-1}$')
            ax[0,2].set_ylabel('DFT Reaction Barrier / $kcal mol^{-1}$')
            ax[1,0].set_xlabel('AM1 $\Sigma_{Distortion Energy}$ / $kcal mol^{-1}$')
            ax[1,0].set_ylabel('DFT $\Sigma_{Distortion Energy}$ / $kcal mol^{-1}$')  
            ax[1,2].set_xlabel('AM1 Interaction Energy / $kcal mol^{-1}$')
            ax[1,2].set_ylabel('DFT Interaction Energy / $kcal mol^{-1}$')  

            ax[0,1].text(0, 0, text_d['q'], verticalalignment='center', horizontalalignment='center')
            ax[0,3].text(0, 0, text_d['e'], verticalalignment='center', horizontalalignment='center')
            ax[1,1].text(0, 0, text_d['d'], verticalalignment='center', horizontalalignment='center')
            ax[1,3].text(0, 0, text_d['i'], verticalalignment='center', horizontalalignment='center')
            ax[0,5].text(0, 0, text_d['di_di'], verticalalignment='center', horizontalalignment='center')
            ax[1,5].text(0, 0, text_d['di_dp'], verticalalignment='center', horizontalalignment='center')

            for axes in [ax[0,0], ax[0,2], ax[1,0], ax[1,2], ax[0,4], ax[1,4]]:
                cr = cursor(axes, hover=2, multiple=True)
                func(cr, comb)
            plt.show()
    
    elif dataset == 'tt':

        comb_gas = lst[0]
        comb_spe = lst[1]
        comb_solvent = lst[2]
        for comb, title in zip([comb_gas, comb_spe, comb_solvent], ['AM1 Gas - DFT', 'AM1 SPE - DFT', 'AM1 Solvent - DFT']):
            metric_d, text_d = _get_metrics(comb)
            fig, ax = plt.subplots(2, 6, figsize=(18, 10), gridspec_kw={
                        #    'width_ratios': [4, 1, 4, 1],
                        #    'height_ratios': [1, 1],
                       'wspace': 0.3,
                       'hspace': 0.4})
            for axis in [ax[0,1], ax[0,3], ax[1,1], ax[1,3], ax[0,5], ax[1,5]]:
                axis.axis('off')
                axis.set_xlim(-1,2)
                axis.set_ylim(-1,1)
            fig.suptitle(title)
            plt.subplots_adjust(left=0.03,
                    bottom=0.1,
                    right=0.98,
                    top=0.9,
                    wspace=0.01,
                    hspace=0.4
                    )
        
            ax_0 = comb.plot(kind='scatter', ax=ax[0,0], x='q_barrier_am1', y='q_barrier_dft', s=40, edgecolor=edge_c, color=inner_c)
            ax_1 = comb.plot(kind='scatter', ax=ax[0,2], x='e_barrier_am1', y='e_barrier_dft', s=40, edgecolor=edge_c, color=inner_c)
            ax_2 = comb.plot(kind='scatter', ax=ax[1,0], x='sum_distortion_energies_am1', y='sum_distortion_energies_dft', s=40, edgecolor=edge_c, color=inner_c)
            ax_3 = comb.plot(kind='scatter', ax=ax[1,2], x='interaction_energies_am1',y='interaction_energies_dft', s=40, edgecolor=edge_c, color=inner_c)
            ax_4 = comb.plot(kind='scatter', ax=ax[0,4], x='distortion_energy_1_am1', y='distortion_energy_1_dft', s=40, edgecolor=edge_c, color=inner_c)
            ax_5 = comb.plot(kind='scatter', ax=ax[1,4], x='distortion_energy_2_am1', y='distortion_energy_2_dft', s=40, edgecolor=edge_c, color=inner_c)

            ax[0,0].set_title(r'$\Delta$G')
            ax[0,2].set_title(r'$\Delta$E')
            ax[1,0].set_title(r'$\Delta$ $E_{Distortion}$')
            ax[1,2].set_title(r'$\Delta$ $E_{Interaction}$')

            ax[0,0].set_xlabel('AM1 Reaction Barrier / $kcal mol^{-1}$')
            ax[0,0].set_ylabel('DFT Reaction Barrier / $kcal mol^{-1}$')
            ax[0,2].set_xlabel('AM1 Reaction Barrier / $kcal mol^{-1}$')
            ax[0,2].set_ylabel('DFT Reaction Barrier / $kcal mol^{-1}$')
            ax[1,0].set_xlabel('AM1 $\Sigma_{Distortion Energy}$ / $kcal mol^{-1}$')
            ax[1,0].set_ylabel('DFT $\Sigma_{Distortion Energy}$ / $kcal mol^{-1}$')  
            ax[1,2].set_xlabel('AM1 Interaction Energy / $kcal mol^{-1}$')
            ax[1,2].set_ylabel('DFT Interaction Energy / $kcal mol^{-1}$')  

            ax[0,1].text(0, 0, text_d['q'], verticalalignment='center', horizontalalignment='center')
            ax[0,3].text(0, 0, text_d['e'], verticalalignment='center', horizontalalignment='center')
            ax[1,1].text(0, 0, text_d['d'], verticalalignment='center', horizontalalignment='center')
            ax[1,3].text(0, 0, text_d['i'], verticalalignment='center', horizontalalignment='center')
            ax[0,5].text(0, 0, text_d['di_1'], verticalalignment='center', horizontalalignment='center')
            ax[1,5].text(0, 0, text_d['di_2'], verticalalignment='center', horizontalalignment='center')

            for axes in [ax[0,0], ax[0,2], ax[1,0], ax[1,2], ax[0,4], ax[1,4]]:
                cr = cursor(axes, hover=2, multiple=True)
                func(cr, comb)
            plt.show()
    

lst = _load_data(dataset)
_get_hists(lst)
# _get_scats(lst)


# df = pd.read_pickle('../../feature_extraction/features/_f_selection/da/sfs_da_exo.pkl')
# #print(df['sum_distortion_energies_am1'])
# print(df.shape)
# x = pd.read_pickle('../../feature_extraction/features/da_exo_features.pkl')[['reaction_number', 'sum_distortion_energies_am1']]
# print(x.shape)
# new_df = df.merge(x, on='reaction_number')
# print(new_df.shape)
# new_df.to_pickle('../../feature_extraction/features/_f_selection/da/sfs_da_exo.pkl')
