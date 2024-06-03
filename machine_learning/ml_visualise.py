"""
#########################################################################################################################
#                                                   ml_visualise.py                                                     #
#                                                                                                                       #
#                        This code takes the ML results for a given dataset and will perform analysis                   #
#                        to generate figures containing the key results.                                                #
#                                                                                                                       #
#                        Depending on the tags used, the files can be printed to screen or saved. The                   #
#                        also allows for the plots to be collated or saved individually.                                #
#                                                                                                                       #
#            Usage:                                                                                                     #
#                                                                                                                       #
#                 python ml_visualise.py -f path/to/ml_results.pkl                                                      #
#                                                                                                                       #
#                 -s  => if used, will save the outputs to files rather than printing them to the screen.               #
#                 -g  => if used, will group the figures together.                                                      #
#                                                                                                                       #
#########################################################################################################################
#                                                      Authors                                                          #
#                                                                                                                       #
#------------------------------------------------  Samuel G. Espley ----------------------------------------------------#
#                                                                                                                       #
#########################################################################################################################
"""


# Imports
import sys
import argparse
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)

class Plotting():

    def __init__(self):
        
        # Pull out path to dataset.
        self.path = options.file
        self.strip_path = self.path.rsplit('/', 1)[0]+'/'
        self.df = pd.read_pickle(self.path)
        self.d_strings = {
                        'mal/':{
                            'title':"Malonate Michael Addition",
                            '2':'NN (2 hidden layers)', 
                            '4':'NN (4 hidden layers)', 
                            'krr':'Kernel Ridge Regression', 
                            'svr':'Support Vector Regression', 
                            'ridge': 'Ridge Regression',
                            'q_barrier_dft':r"$\Delta$G", 
                            'e_barrier_dft':r"$\Delta$E", 
                            'interaction_energies_dft':'Interaction Energy', 
                            'sum_distortion_energies_dft':'Sum Distortion Energy', 
                            'distortion_energy_0_dft':'Malonate Nucleophile Distortion Energy',
                            'distortion_energy_1_dft':'Michael Acceptor Distortion Energy',
                            'd1':'energy_0',
                            'd2':'energy_1'},
                        'ma/':{
                            'title':"Nitro Michael Addition",
                            '2':'NN (2 hidden layers)', 
                            '4':'NN (4 hidden layers)', 
                            'krr':'Kernel Ridge Regression', 
                            'svr':'Support Vector Regression', 
                            'ridge': 'Ridge Regression',
                            'q_barrier_dft':r"$\Delta$G", 
                            'e_barrier_dft':r"$\Delta$E", 
                            'interaction_energies_dft':'Interaction Energy', 
                            'sum_distortion_energies_dft':'Sum Distortion Energy', 
                            'distortion_energy_0_dft':'Nitro Nucleophile Distortion Energy',
                            'distortion_energy_1_dft':'Michael Acceptor Distortion Energy',
                            'd1':'energy_0',
                            'd2':'energy_1'},
                        'da/':{
                            'title':"Diels-Alder",
                            '2':'NN (2 hidden layers)', 
                            '4':'NN (4 hidden layers)', 
                            'krr':'Kernel Ridge Regression', 
                            'svr':'Support Vector Regression', 
                            'ridge': 'Ridge Regression',
                            'q_barrier_dft':r"$\Delta$G", 
                            'e_barrier_dft':r"$\Delta$E", 
                            'interaction_energies_dft':'Interaction Energy', 
                            'sum_distortion_energies_dft':'Sum Distortion Energy', 
                            'distortion_energy_di_dft':'Diene Distortion Energy',
                            'distortion_energy_dp_dft':'Dienophile Distortion Energy',
                            'd1':'energy_di',
                            'd2':'energy_dp'},
                        'da/lit_da/cyalk/':{
                            'title':"Diels-Alder Cycloalkenones Test Set",
                            '2':'NN (2 hidden layers)', 
                            '4':'NN (4 hidden layers)', 
                            'krr':'Kernel Ridge Regression', 
                            'svr':'Support Vector Regression', 
                            'ridge': 'Ridge Regression',
                            'q_barrier_dft':r"$\Delta$G", 
                            'e_barrier_dft':r"$\Delta$E", 
                            'interaction_energies_dft':'Interaction Energy', 
                            'sum_distortion_energies_dft':'Sum Distortion Energy', 
                            'distortion_energy_di_dft':'Diene Distortion Energy',
                            'distortion_energy_dp_dft':'Dienophile Distortion Energy',
                            'd1':'energy_di',
                            'd2':'energy_dp'},
                        'da/lit_da/cypro/':{
                            'title':"Diels-Alder Cyclopropenes Test Set",
                            '2':'NN (2 hidden layers)', 
                            '4':'NN (4 hidden layers)', 
                            'krr':'Kernel Ridge Regression', 
                            'svr':'Support Vector Regression', 
                            'ridge': 'Ridge Regression',
                            'q_barrier_dft':r"$\Delta$G", 
                            'e_barrier_dft':r"$\Delta$E", 
                            'interaction_energies_dft':'Interaction Energy', 
                            'sum_distortion_energies_dft':'Sum Distortion Energy', 
                            'distortion_energy_di_dft':'Diene Distortion Energy',
                            'distortion_energy_dp_dft':'Dienophile Distortion Energy',
                            'd1':'energy_di',
                            'd2':'energy_dp'},
                        'tt/':{
                            'title':"[3+2] Cycloaddition",
                            '2':'NN (2 hidden layers)', 
                            '4':'NN (4 hidden layers)', 
                            'krr':'Kernel Ridge Regression', 
                            'svr':'Support Vector Regression', 
                            'ridge': 'Ridge Regression',
                            'q_barrier_dft':r"$\Delta$G", 
                            'e_barrier_dft':r"$\Delta$E", 
                            'interaction_energies_dft':'Interaction Energy', 
                            'sum_distortion_energies_dft':'Sum Distortion Energy', 
                            'distortion_energy_1_dft':'Dipole Distortion Energy',
                            'distortion_energy_2_dft':'Dipolarophile Acceptor Distortion Energy',
                            'd1':'energy_1',
                            'd2':'energy_2'}
                    }
        # Check if test or external test data.
        if 'y_test_pred' not in self.df.columns.to_list():
            self.ext = 1
        else:
            self.ext = 0

    def _models(self):
        '''
        Function to pull out the best model over the tested models stored in ml_results.pkl
        The function will plot two graphs of the results:
            - The best models for each target.
            - All the models for the target.
        '''
        def _ave_map(l):
            value = sum(l)/len(l)
            return value
        
        def _min_index(l1, l2):
            value = l2[l1.index(min(l1))]
            return value
        
        def _index(l1):
            value = l1.index(min(l1))
            return value

        if self.ext == 0:
            self.df['y_test_pred_average'] = self.df['y_test_pred'].apply(_ave_map)
            self.df['std_error_test_average'] = self.df['std_error_test'].apply(_ave_map)
            self.df['y_test_pred_min'] = self.df['y_test_pred'].apply(min)
            self.df['rs_ind'] = self.df['y_test_pred'].apply(_index)
            self.df['best_random_state'] = self.df.apply(lambda x: _min_index(x.y_test_pred, x.random_state), axis=1)
        elif self.ext == 1:
            self.df['y_ext_pred_average'] = self.df['y_ext_pred'].apply(_ave_map)
            self.df['std_error_ext_average'] = self.df['std_error_ext'].apply(_ave_map)
            self.df['y_ext_pred_min'] = self.df['y_ext_pred'].apply(min)
            self.df['rs_ind'] = self.df['y_ext_pred'].apply(_index)
            self.df['best_random_state'] = self.df.apply(lambda x: _min_index(x.y_ext_pred, x.random_state), axis=1)

        # Pull out different targets within this.
        q_b = self.df[self.df['model_target'].str.contains('_q_barrier')]
        e_b = self.df[self.df['model_target'].str.contains('_e_barrier')]
        inter = self.df[self.df['model_target'].str.contains('interaction')]
        # sum_d = self.df[self.df['model_target'].str.contains('_sum_')]

        r_models = self.df[self.df['model_target'].str.contains('ridge')]
        krr_models = self.df[self.df['model_target'].str.contains('krr')]
        svr_models = self.df[self.df['model_target'].str.contains('svr')]
        nn_2 = self.df[self.df['model_target'].str.contains('2_st_nn')]
        nn_4 = self.df[self.df['model_target'].str.contains('4_st_nn')]

               
        # Pull out the different datasets.
        if 'ma/' in self.path:
            one = self.df[self.df['model_target'].str.contains('energy_0')]
            two = self.df[self.df['model_target'].str.contains('energy_1')]
        elif 'mal/' in self.path:
            one = self.df[self.df['model_target'].str.contains('energy_0')]
            two = self.df[self.df['model_target'].str.contains('energy_1')]
        elif 'da/' in self.path:
            one = self.df[self.df['model_target'].str.contains('energy_di')]
            two = self.df[self.df['model_target'].str.contains('energy_dp')]
        elif 'tt/' in self.path:
            one = self.df[self.df['model_target'].str.contains('energy_1')]
            two = self.df[self.df['model_target'].str.contains('energy_2')]

        model_dfs = [r_models, krr_models, svr_models, nn_2, nn_4]
        dfs = [
            q_b, e_b, inter, 
            #sum_d, 
            one, two]
        best_models = []
        for df in dfs:
            if self.ext == 0:
                best_models.append(df['model_target'][df['y_test_pred_min'] == min(df['y_test_pred_min'])].iloc[0])
            elif self.ext == 1:
                best_models.append(df['model_target'][df['y_ext_pred_min'] == min(df['y_ext_pred_min'])].iloc[0])
        
        Plotting()._plot_best_models(self.df, best_models)
        Plotting()._plot_all_models(dfs, best_models)
        # Plotting()._plot_collation_predictions(model_dfs, best_models)
  
    def _plot_best_models(self, df, best_models):
        '''
        Function to plot the best performing model on the best random state over 5 random states.
        
        Arguments:
        df (DataFrame): A dataframe containing the ML results.
        best_models (List): A list of the best models.
        
        Outputs:
        Will either save or print figures relating to the best performing models.
        '''
        # Determine whether test or external data.
        if self.ext == 0:
            data_name = 'test'
        elif self.ext == 1:
            data_name = 'ext'
        # Check if flag passed for individual figures - If not create subplot.
        if options.grouped == False:
            fig, ax = plt.subplots(2,3, figsize=(14,10))
            for bm, axes, col in zip(best_models, ax.reshape(-1), [['cornflowerblue', 'navy'], ['slateblue', 'indigo'], ['forestgreen', 'darkgreen'], ['lightcoral', 'maroon'], ['lightcoral', 'maroon'], ['lightcoral', 'maroon']]): 
                ind = int(df['rs_ind'][df['model_target'] == bm].iloc[0])
                axes.scatter(df[f'y_{data_name}_pred_values'][df['model_target'] == bm][0][ind], df[f'y_{data_name}_true'][df['model_target'] == bm][0][ind], color=col[0], edgecolor=col[-1], s=80, linewidth=2)
                props = dict(boxstyle='round', facecolor='white', edgecolor='black', alpha=0.7)
                
                # The way NN results are saved is very slightly different. Deal with that here.
                if 'nn' in bm:
                    min_v = min(df[f'y_{data_name}_pred_values'][df['model_target'] == bm][0][ind][0])
                    max_v = max(df[f'y_{data_name}_true'][df['model_target'] == bm][0][ind][0])
                else:
                    min_v = min(df[f'y_{data_name}_pred_values'][df['model_target'] == bm][0][ind])
                    max_v = max(df[f'y_{data_name}_true'][df['model_target'] == bm][0][ind])
            
                axes.text(min_v, max_v, f"{bm} \n{df[f'y_{data_name}_pred_min'][df['model_target'] == bm].iloc[0]} kcal/mol", verticalalignment='top', size=10, fontweight='semibold', bbox=props)
            # Plot labels.
            fig.supylabel('True Values / kcalmol$^{-1}$', size=13, fontweight='semibold')
            fig.supxlabel('Predicted Values / kcalmol$^{-1}$', size=13, fontweight='semibold')
            if options.save == True:    
                plt.savefig(f'{self.strip_path}figures/best_models.png', dpi=800)
                plt.clf()
            elif options.save == False:
                fig.tight_layout()
                plt.show()
        # Create individual plots.
        else:
            for bm, col in zip(best_models, [['cornflowerblue', 'navy'], ['slateblue', 'indigo'], ['forestgreen', 'darkgreen'], ['lightcoral', 'maroon'], ['lightcoral', 'maroon'], ['lightcoral', 'maroon']]): 
                #labels = [self.d_strings[self.strip_path][i.split('_')[0]] for i in df['model_target']]
                name = bm.split('_')[0]
                target = [i.split('_nn_', 2)[-1] if 'nn' in i else i.split('_', 1)[-1] for i in [bm]][0]
                ind = int(df['rs_ind'][df['model_target'] == bm].iloc[0])
                if target == 'interaction_energies_dft':
                    if 'nn' in bm:
                        x = [-i for i in df[f'y_{data_name}_pred_values'][df['model_target'] == bm][0][ind][0]]
                        y = [-i for i in df[f'y_{data_name}_true'][df['model_target'] == bm][0][ind][0]]
                    else:
                        x = [-i for i in df[f'y_{data_name}_pred_values'][df['model_target'] == bm][0][ind]]
                        y = [-i for i in df[f'y_{data_name}_true'][df['model_target'] == bm][0][ind]]
                    plt.scatter(x, y, color=col[0], edgecolor=col[-1], s=80, linewidth=2)
                    xmin, xmax, ymin, ymax = plt.axis()
                    plt.plot([100, -100], [100, -100], linestyle='--', color='black', linewidth=2)
                    plt.fill_between([100, -100], [100+1, -100+1], [100-1, -100-1], color='grey', edgecolor='white', alpha=0.5, label='_nolegend_', zorder=0)
                else:
                    plt.scatter(df[f'y_{data_name}_pred_values'][df['model_target'] == bm][0][ind], df[f'y_{data_name}_true'][df['model_target'] == bm][0][ind], color=col[0], edgecolor=col[-1], s=80, linewidth=2)
                    xmin, xmax, ymin, ymax = plt.axis()
                    plt.plot([-100, 100], [-100, 100], linestyle='--', color='black', linewidth=2)
                    plt.fill_between([-100, 100], [-100+1, 100+1], [-100-1, 100-1], color='grey', edgecolor='white', alpha=0.5, label='_nolegend_', zorder=0)
                
                props = dict(boxstyle='round', facecolor='white', edgecolor='black', alpha=0.7)       
                # Add a solve here but realistically need to ammend ml_analysis.py to not save NN models in too many lists...
                if 'nn' in bm:
                    min_v = min(df[f'y_{data_name}_pred_values'][df['model_target'] == bm][0][ind][0])
                    max_v = max(df[f'y_{data_name}_true'][df['model_target'] == bm][0][ind][0])
                else:
                    min_v = min(df[f'y_{data_name}_pred_values'][df['model_target'] == bm][0][ind])
                    max_v = max(df[f'y_{data_name}_true'][df['model_target'] == bm][0][ind])
                # xmin, xmax, ymin, ymax = plt.axis()
                # plt.plot([-100, 100], [-100, 100], linestyle='--', color='black', linewidth=2)
                # plt.fill_between([-100, 100], [-100+1, 100+1], [-100-1, 100-1], color='grey', edgecolor='white', alpha=0.5, label='_nolegend_', zorder=0)
                plt.title(f"{self.d_strings[self.strip_path][target]}", size=13, fontweight='semibold')            
                plt.ylabel("True DFT Energy / kcal mol$^{-1}$", size=13, fontweight='semibold')
                plt.xlabel("Predicted DFT Energy / kcal mol$^{-1}$", size=13, fontweight='semibold')
                if target == 'interaction_energies_dft':
                    plt.text( min(x), max(y), f"{self.d_strings[self.strip_path][name]} \nTest MAE: {df[f'y_{data_name}_pred_min'][df['model_target'] == bm].iloc[0]:.2f} kcal/mol", verticalalignment='top', size=10, fontweight='semibold', bbox=props)
                else:
                    plt.text(min_v, max_v, f"{self.d_strings[self.strip_path][name]} \nTest MAE: {df[f'y_{data_name}_pred_min'][df['model_target'] == bm].iloc[0]:.2f} kcal/mol", verticalalignment='top', size=10, fontweight='semibold', bbox=props)

                plt.xlim(xmin, xmax)
                plt.ylim(ymin, ymax)
                if options.save == True:
                    plt.savefig(f"{self.strip_path}figures/{target}_best_rs_{df['best_random_state'][df['model_target'] == bm].iloc[0]}.png", dpi=800)
                    plt.clf()
                elif options.save == False:
                    plt.show()
                    plt.clf()

    def _plot_all_models(self, dfs, best_models):
        '''
        Function to plot all models in the ml_results.pkl file.
        
        Arguments:
        df (DataFrame): A dataframe containing the ML results.
        best_models (List): A list of the best models.
        
        Outputs:
        Will either save or print figures for all the models.
        '''
        colours = [
            ['cornflowerblue', 'navy'],
            ['slateblue', 'indigo'], 
            ['forestgreen', 'darkgreen'], 
            ['lightcoral', 'maroon'], 
            ['lightcoral', 'maroon'], 
            ['lightcoral', 'maroon']
        ]
        # Determine whether test or external data.
        if self.ext == 0:
            data_name = 'test'
        elif self.ext == 1:
            data_name = 'ext'
        # Check if flag passed for individual figures - If not create subplot.
        if options.grouped == False:
            fig, ax = plt.subplots(2,3, figsize=(14,10), sharex=True)
            for df, axes, name, col in zip(dfs, ax.reshape(-1), best_models, colours):
                labels = [self.d_strings[self.strip_path][i.split('_')[0]] for i in df['model_target']]
                names = [i.split('_', 1)[-1] for i in df['model_target']]
                names = [i.split('_nn_')[-1] if 'nn' in i else i for i in names]
                df = df.sort_values(by='model_target')
                df.plot.bar(x='model_target', y=f'y_{data_name}_pred_average', ax=axes, color=col[0], edgecolor=col[-1], linewidth=3)
                axes.axhline(1, color='black', linestyle='--')
                axes.set_xlabel('')
                axes.set_title(self.d_strings[self.strip_path][names[0]])
                xt = {i:i.split('_')[0] for i in df['model_target']}
                xt_l = [f'{i} NN' if i.isdigit() == True else i.capitalize() for i in list(xt.values())]
                xt_l = [i.upper() if len(i) == 3 else i for i in xt_l]
                axes.set_xticks(list(range(0, len(xt.keys()))), xt_l)
            if options.save == True:    
                plt.savefig(f'{self.strip_path}figures/all_models.png', dpi=800)
                plt.clf()
            elif options.save == False:
                plt.show()
                plt.clf()
        # Create individual plots.
        else:
            for df, name, col in zip(dfs, best_models, colours):
                df = df.sort_values(by='model_target')
                labels = [self.d_strings[self.strip_path][i.split('_')[0]] for i in df['model_target']]
                names = [i.split('_', 1)[-1] for i in df['model_target']]
                names = [i.split('_nn_')[-1] if 'nn' in i else i for i in names]
                plt.title(f"{self.d_strings[self.strip_path]['title']}\n{self.d_strings[self.strip_path][names[0]]}")
                plt.barh(df['model_target'], df[f'y_{data_name}_pred_average'], tick_label=labels, color=col[0], edgecolor=col[-1], height=0.5, linewidth=2)
                plt.errorbar(df[f'y_{data_name}_pred_average'].tolist(), df['model_target'].tolist(),  xerr=df[f'std_error_{data_name}_average'].tolist(), capsize=4, capthick=1.8, elinewidth=1.8, ls='none', color='black')
                plt.axvline(1, color='black', linestyle='--')
                plt.tight_layout()
                if options.save == True:
                    plt.savefig(f"{self.strip_path}figures/{names[0]}_ave.png", dpi=800)
                    plt.clf()
                elif options.save == False:
                    plt.show()
                    plt.clf()
                plt.clf()

    def _plot_collation_predictions(self, dfs, best_models):
        '''
        Function to plot collation prediction results - this is where multiple models are used to calculate a target.
        
        Arguments:
        dfs (List): A list containing the dataframes of ML results.
        best_models (List): A list of the best models.
        
        Outputs:
        Will either save or print figures for all the models.
        '''
        targets = [i.split('_nn_', 2)[-1] if 'nn' in i else i.split('_', 1)[-1] for i in best_models]
        rss = dfs[0]['random_state'].iloc[0][0]
        # Do this just for random state 23 as this is what the models were tuned on. 
        for df in dfs:
            d1_str = self.d_strings[self.strip_path]['d1']
            d2_str = self.d_strings[self.strip_path]['d2']

            d1 = df['y_test_pred_values'][df['model_target'].str.contains(d1_str)].iloc[0][1]
            d2 = df['y_test_pred_values'][df['model_target'].str.contains(d2_str)].iloc[0][1]
            
            inter = df['y_test_pred_values'][df['model_target'].str.contains('inter')].iloc[0][1]
            de =  df['y_test_true'][df['model_target'].str.contains('_e_barrier')].iloc[0][1]
            pred_de =  df['y_test_pred_values'][df['model_target'].str.contains('_e_barrier')].iloc[0][1]
            # Deal with differences in how NN results where saved.
            if '_nn_' in df['model_target'].iloc[0]:
                d1=d1[0]
                d2=d2[0]
                inter=inter[0]
                de=de[0]
                pred_de=pred_de[0]     
            sum_pred_de = (np.array(d1)+np.array(d2))-np.array(inter) 
            direct_pred = np.round(mean_absolute_error(np.array(de), np.array(pred_de)), 3)
            alt_pred = np.round(mean_absolute_error(sum_pred_de, de), 3)
            min_v = min([min(sum_pred_de), min(de), min(sum_pred_de)])
            max_v = max([max(sum_pred_de), max(de), max(sum_pred_de)])
            # props = dict(boxstyle='round', facecolor='white', edgecolor='black', alpha=0.7)  
            # plt.text(min_v+0.5, max_v-0.5, f"Direct Prediction MAE: {direct_pred}\nCollate Prediction MAE: {alt_pred}", verticalalignment='top', horizontalalignment='left', size=10, fontweight='semibold', bbox=props)
            plt.title(fr"$\Delta$E Standard and Summative Prediction - {self.d_strings[self.strip_path][df['model_target'].iloc[0].split('_')[0]]}")
            plt.scatter(sum_pred_de, de, color='cornflowerblue', edgecolor='darkblue')
            plt.scatter(pred_de, de, color='darkgrey', edgecolor='grey')
            plt.legend([f'Summative Prediction: {alt_pred} kcal/mol', f'Standard Prediction: {direct_pred} kcal/mol'], loc='upper left')
            plt.savefig(f"{self.strip_path}figures/{df['model_target'].iloc[0].split('_')[0]}_sum_pred.png", dpi=800)
            plt.clf()

def main():

    Plotting()._models()

if __name__ == '__main__':

    # Argument parse
    parser = argparse.ArgumentParser(
        prog='ml_visualise',
        description='Python script for visualising ML results.')
    parser.add_argument('-s', dest='save', action='store_true', help='Tag to save files or not.')
    parser.add_argument('-g', dest='grouped', action='store_false', help='Tag to create files as a group instead of individually.')
    parser.add_argument('-f', dest='file', default=None, required=True, help='The path to the results pickle file.')
    
    (options, args) = parser.parse_known_args()

    main()