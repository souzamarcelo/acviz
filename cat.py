#!/usr/bin/python3
import os
import sys
import argparse
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib import lines as mlines
from rpy2 import robjects
from math import log
from math import ceil
from copy import copy

desc = '''
-------------------------------------------------------------------------------
| cat: Configuration Analysis Tools (CAT) for irace                           |
| Version: 1.0                                                                |
| Copyright (C) 2020                                                          |
| Marcelo de Souza     <marcelo.desouza@udesc.br>                             |
| Marcus Ritt          <marcus.ritt@inf.ufrgs.br>                             |
|                                                                             |
| This is free software, and you are welcome to redistribute it under certain |
| conditions.  See the GNU General Public License for details. There is NO    |
| WARRANTY; not even for MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. |
-------------------------------------------------------------------------------
'''

colors = ['#EF9A9A','#1A237E','#B71C1C','#00C853','#F57F17','#424242','#FDD835','#B0BEC5','#004D40','#000000','#F50057',
          '#2196F3','#795548','#76FF03','#9C27B0','#1565C0','#F9A825','#F48FB1','#81C784','#F44336','#A1887F','#1B5E20',
          '#69F0AE','#EC407A','#827717','#607D8B','#D50000','#CE93D8','#43A047','#FFEA00','#18FFFF','#3F51B5','#FF6F00',
          '#757575','#64DD17','#EEFF41','#7C4DFF','#33691E','#90CAF9','#AA00FF','#FFF176','#8BC34A','#009688','#9E9D24']


def plotEvo(data, restarts, showElites, showInstances, showConfigurations, pconfig):
    fig = plt.figure('Plot evolution [cat]')
    ax = fig.add_subplot(1, 1, 1, label = 'plot_evolution')
    ax.set_xlim((1, data['id'].max()))
    
    plt.title('Evolution of the configuration process')
    plt.xlabel('candidate evaluations [from %d to %d]' % (1, data['id'].max()))
    plt.ylabel('solution quality [relative deviation]')
    
    simpleColors = {'regular': '#202020', 'elite': 'blue', 'final': 'red', 'best': 'green'}
    data['color'] = data.apply(lambda x: colors[(x['instance'] - 1) % len(colors)] if showInstances else simpleColors[x['type']] if showElites else 'black', axis = 1)
    data.loc[data['reldev'] == 0, 'reldev'] = data[data['reldev'] > 0]['reldev'].min() / 2

    legendElements = []; legendDescriptions = []
    if showElites:
        legendElements.append(copy(plt.scatter(data[data['type'] == 'regular']['id'], data[data['type'] == 'regular']['reldev'].map(log), alpha = 1, c = data[data['type'] == 'regular']['color'], marker = 'x', linewidth = 0.5, s = 16)))
        legendElements.append(copy(plt.scatter(data[data['type'] == 'elite']['id'], data[data['type'] == 'elite']['reldev'].map(log), alpha = 1, c = data[data['type'] == 'elite']['color'], edgecolors = 'black', marker = 'o', linewidth = 0.7, s = 24)))
        legendElements.append(copy(plt.scatter(data[data['type'] == 'final']['id'], data[data['type'] == 'final']['reldev'].map(log), alpha = 1, c = data[data['type'] == 'final']['color'], edgecolors = 'black', marker = 'D', linewidth = 0.7, s = 22)))
        legendElements.append(copy(plt.scatter(data[data['type'] == 'best']['id'], data[data['type'] == 'best']['reldev'].map(log), alpha = 1, c = data[data['type'] == 'best']['color'], edgecolors = 'black', marker = '*', linewidth = 0.7, s = 70)))
        legendDescriptions.extend(['regular exec.', 'elite config.', 'final elite config.', 'best found config.'])
    else:
        legendElements.append(copy(plt.scatter(data['id'], data['reldev'].map(log), alpha = 1, c = data['color'], marker = 'x', linewidth = 0.5, s = 16)))
        legendDescriptions.append('regular exec.')
    if showInstances:
        for element in legendElements: element.set_edgecolor('black'); element.set_facecolor('grey')
    
    legendRegular = False; legendRestart = False
    iterationPoints = data.groupby('iteration', as_index = False).agg({'id': 'first'})['id'].tolist()
    for point, restart in zip(iterationPoints, restarts):
        color = '#B71C1C' if restart else 'k'
        line = plt.axvline(x = point, color = color, linestyle = '--', linewidth = 1.4)
        if not restart and not legendRegular:
            legendElements.append(line)
            legendDescriptions.append('iteration')
            legendRegular = True
        if restart and not legendRestart:
            legendElements.append(line)
            legendDescriptions.append('iteration with restart')
            legendRestart = True
    ax.set_xticks(iterationPoints)
    
    iterations = data['iteration'].unique().tolist()
    avg = [data[data['iteration'] == iteration]['reldev'].map(log).mean() for iteration in iterations]
    best = [data[(data['iteration'] == iteration) & ((data['type'] == 'elite') | (data['type'] == 'final') | (data['type'] == 'best'))]['reldev'].map(log).mean() for iteration in iterations]
    iterationPoints.append(data['id'].max())
    for i in range(len(iterations)):
        plt.plot([iterationPoints[i], iterationPoints[i + 1]], [avg[i], avg[i]], linestyle = '-', color = '#FF8C00', linewidth = 1.8)
        plt.plot([iterationPoints[i], iterationPoints[i + 1]], [best[i], best[i]], linestyle = '-', color = '#800080', linewidth = 1.8)
    legendElements.append(mlines.Line2D([], [], color='#FF8C00', linewidth = 1.8))
    legendDescriptions.append('performance on iteration')
    legendElements.append(mlines.Line2D([], [], color='#800080', linewidth = 1.8))
    legendDescriptions.append('performance of elites')
    
    if showConfigurations:
        for iteration in iterations:
            amount = ceil(len(data[data['iteration'] == iteration]) * pconfig / 100)
            data.groupby('iteration', as_index = False).agg({'id': 'count'})
            best = data[data['iteration'] == iteration].sort_values('reldev', ascending = True).head(amount)
            names = best['configuration'].tolist()
            x = best['id'].tolist()
            y = best['reldev'].map(log).tolist()
            for i in range(len(x)):
                ax.annotate(names[i], xy = (x[i], y[i]), xytext = (0, -8), textcoords = 'offset pixels', horizontalalignment = 'center', verticalalignment = 'center', fontsize = 5)

    fig.legend(legendElements, legendDescriptions, loc = 'center', bbox_to_anchor = (0.5, 0.05), ncol = 4, handletextpad = 0.5, columnspacing = 1.8)
    ax.tick_params(axis = 'both', which = 'major', labelsize = 9)
    plt.xticks(rotation = 90)

    fig.set_size_inches(10, 6)
    fig.subplots_adjust(top = 0.95)
    fig.subplots_adjust(bottom = 0.21)
    fig.subplots_adjust(right = 0.99)
    fig.subplots_adjust(left = 0.06)


def read(iracelog):
    robjects.r['load'](iracelog)
    iraceExp = np.array(robjects.r('iraceResults$experiments'))
    iraceExpLog = np.array(robjects.r('iraceResults$experimentLog'))

    elites = []
    for i in range(1, int(robjects.r('iraceResults$state$indexIteration')[0])):
        elites.append([int(item) for item in str(robjects.r('iraceResults$allElites[[' + str(i) + ']]')).replace('[1]', '').strip().split()])

    experiments = []; id = 0
    for i in range(len(iraceExpLog)):
        id += 1
        experiment = {}
        experiment['id'] = id
        experiment['iteration'] = int(iraceExpLog[i][0])
        experiment['instance'] = int(iraceExpLog[i][1])
        experiment['configuration'] = int(iraceExpLog[i][2])
        experiment['value'] = iraceExp[experiment['instance'] - 1][experiment['configuration'] - 1]
        experiment['type'] = ('best' if experiment['configuration'] == elites[-1][0] else
                              'final' if experiment['configuration'] in elites[-1] else
                              'elite' if experiment['configuration'] in elites[experiment['iteration'] - 1] else
                              'regular')
        experiments.append(experiment)
    data = pd.DataFrame(experiments)

    data['bkv'] = 'NA'
    for instance in data['instance'].unique().tolist():
        data.loc[data['instance'] == instance, 'bkv'] = data[data['instance'] == instance]['value'].min()
    data['reldev'] = abs(1 - (data['value'] / data['bkv']))

    restarts = [bool(item) for item in np.array(robjects.r('iraceResults$softRestart'))]
    return data, restarts


def main(iracelog, showElites, showInstances, showConfigurations, pconfig, exportData, exportPlot, output):
    print(desc)
    settings = '> Settings:\n  - plot evolution of the configuration process\n'
    if showElites: settings += '  - show elite configurations\n'
    if showInstances: settings += '  - identify instances\n'
    if showConfigurations: settings += '  - show configurations of the best performers\n'
    if showConfigurations: settings += '  - pconfig = %d\n' % pconfig
    if exportData: settings += '  - export data to csv\n'
    if exportPlot: settings += '  - export plot to pdf and png\n'
    if exportData or exportPlot: settings += '  - output file name: %s\n' % output
    print(settings)
    
    data, restarts = read(iracelog)
    plotEvo(data, restarts, showElites, showInstances, showConfigurations, pconfig)
    
    if exportData:
        if not os.path.exists('./export'): os.mkdir('./export')
        file = open('./export/' + output + '.csv', 'w')
        file.write(data.to_csv())
        file.close()
        print('> data exported to export/' + output + '.csv')
    if exportPlot:
        if not os.path.exists('./export'): os.mkdir('./export')
        plt.savefig('./export/' + output + '.pdf', format = 'pdf', dpi = 1000)
        plt.savefig('./export/' + output + '.png', format = 'png', dpi = 1000)
        print('> Plot exported to export/' + output + '.pdf')
        print('> Plot exported to export/' + output + '.png')
    else:
        plt.show()
    print('-------------------------------------------------------------------------------')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser._action_groups.pop()
    required = parser.add_argument_group('required arguments')
    optional = parser.add_argument_group('optional arguments')
    required.add_argument('--iracelog', help = 'input of irace log file (.Rdata)', metavar = '<file>', required = ('--version' not in sys.argv and '-v' not in sys.argv))
    optional.add_argument('-v', '--version', help = 'show description and exit', action = 'store_true')
    optional.add_argument('--elites', help = 'enables identification of elite configurations (disabled by default)', action = 'store_true')
    optional.add_argument('--configurations', help = 'enables identification of configurations (disabled by default)', action = 'store_true')
    optional.add_argument('--pconfig', help = 'when --configurations, show configurations of the p%% best executions [0, 100] (default: 10)', metavar = '<p>', default = 10, type = int)
    optional.add_argument('--instances', help = 'enables identification of instances (disabled by default)', action = 'store_true')
    optional.add_argument('--exportdata', help = 'exports the used data to a csv format file (disabled by default)', action = 'store_true')
    optional.add_argument('--exportplot', help = 'exports the resulting plot to png and pdf files (disabled by default)', action = 'store_true')
    optional.add_argument('--output', help = 'defines a name for the output files', metavar = '<name>', type = str)
    args = parser.parse_args()
    if args.version: print(desc); exit()
    if not args.iracelog: print('Invalid arguments!\nPlease input the irace log file using \'--iracelog <file>\'\n'); parser.print_help(); exit()
    main(args.iracelog, args.elites, args.instances, args.configurations, args.pconfig, args.exportdata, args.exportplot, (args.output if args.output else args.iracelog[args.iracelog.rfind('/')+1:].replace('.Rdata', '')))