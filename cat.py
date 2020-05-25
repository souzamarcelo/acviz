#!/usr/bin/python3
import os
import sys
import math
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

annotations = None
fig = None
ax = None
plotData = None
colors = ['#EF9A9A','#1A237E','#B71C1C','#00C853','#F57F17','#424242','#FDD835','#B0BEC5','#004D40','#000000','#F50057',
          '#2196F3','#795548','#76FF03','#9C27B0','#1565C0','#F9A825','#F48FB1','#81C784','#F44336','#A1887F','#1B5E20',
          '#69F0AE','#EC407A','#827717','#607D8B','#D50000','#CE93D8','#43A047','#FFEA00','#18FFFF','#3F51B5','#FF6F00',
          '#757575','#64DD17','#EEFF41','#7C4DFF','#33691E','#90CAF9','#AA00FF','#FFF176','#8BC34A','#009688','#9E9D24']


def updateAnnotations(ind, pathCollection, data):
    index = ind['ind'][0]
    data = data.iloc[[index]]
    annotations.xy = pathCollection.get_offsets()[index]
    exec = data['id'].unique()[0]
    config = data['configuration'].unique()[0]
    instance = data['instance'].unique()[0]
    instancename = data['instancename'].unique()[0]
    annotations.set_text('execution: %d\ninstance: %d (%s)\nconfiguration: %d' % (exec, instance, instancename, config))
    annotations.get_bbox_patch().set_facecolor(data['color'].unique()[0])
    annotations.get_bbox_patch().set_alpha(0.6)


def hover(event):
    vis = annotations.get_visible()
    if event.inaxes == ax:
        found = False
        for i in range(len(ax.collections)):
            pathCollection = ax.collections[i]
            cont, ind = pathCollection.contains(event)
            if cont:
                updateAnnotations(ind, pathCollection, plotData[i])
                annotations.set_visible(True)
                fig.canvas.draw_idle()
                found = True
                break
        if not found and vis:
            annotations.set_visible(False)
            fig.canvas.draw_idle()


def plotEvo(data, restarts, showElites, showInstances, showConfigurations, pconfig):
    global annotations, ax, fig, plotData
    fig = plt.figure('Plot evolution [cat]')
    ax = fig.add_subplot(1, 1, 1, label = 'plot_evolution')
    ax.set_xlim((data['xaxis'].min(), data['xaxis'].max()))
    
    plt.title('Evolution of the configuration process')
    plt.xlabel('candidate evaluations [from %d to %d]' % (data['xaxis'].min(), data['xaxis'].max()))
    plt.ylabel('solution quality [relative deviation]')

    simpleColors = {'regular': '#202020', 'elite': 'blue', 'final': 'red', 'best': 'green'}
    data['color'] = data.apply(lambda x: colors[(x['instance'] - 1) % len(colors)] if showInstances else simpleColors[x['type']] if showElites else 'black', axis = 1)
    data.loc[data['reldev'] == 0, 'reldev'] = data[data['reldev'] > 0]['reldev'].min() / 2

    legendElements = []; legendDescriptions = []; plotData = []
    if showElites:
        plotData.append(data[data['type'] == 'regular'])
        plotData.append(data[data['type'] == 'elite'])
        plotData.append(data[data['type'] == 'final'])
        plotData.append(data[data['type'] == 'best'])
        legendElements.append(copy(plt.scatter(data[data['type'] == 'regular']['xaxis'], data[data['type'] == 'regular']['reldev'].map(log), alpha = 1, c = data[data['type'] == 'regular']['color'], marker = 'x', linewidth = 0.5, s = 16)))
        legendElements.append(copy(plt.scatter(data[data['type'] == 'elite']['xaxis'], data[data['type'] == 'elite']['reldev'].map(log), alpha = 1, c = data[data['type'] == 'elite']['color'], edgecolors = 'black', marker = 'o', linewidth = 0.7, s = 24)))
        legendElements.append(copy(plt.scatter(data[data['type'] == 'final']['xaxis'], data[data['type'] == 'final']['reldev'].map(log), alpha = 1, c = data[data['type'] == 'final']['color'], edgecolors = 'black', marker = 'D', linewidth = 0.7, s = 22)))
        legendElements.append(copy(plt.scatter(data[data['type'] == 'best']['xaxis'], data[data['type'] == 'best']['reldev'].map(log), alpha = 1, c = data[data['type'] == 'best']['color'], edgecolors = 'black', marker = '*', linewidth = 0.7, s = 70)))
        legendDescriptions.extend(['regular exec.', 'elite config.', 'final elite config.', 'best found config.'])
    else:
        plotData.append(data)
        legendElements.append(copy(plt.scatter(data['xaxis'], data['reldev'].map(log), alpha = 1, c = data['color'], marker = 'x', linewidth = 0.5, s = 16)))
        legendDescriptions.append('regular exec.')
    if showInstances:
        for element in legendElements: element.set_edgecolor('black'); element.set_facecolor('grey')

    annotations = ax.annotate('', xy = (0, 0), xytext = (0, 20), textcoords = 'offset points', bbox = dict(boxstyle = 'round'), ha = 'left')
    annotations.set_visible(False)

    legendRegular = False; legendRestart = False
    iterationPoints = data.groupby('iteration', as_index = False).agg({'xaxis': 'first'})['xaxis'].tolist()
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
    avg = [data[data['iteration'] == iteration]['reldev'].map(log).median() for iteration in iterations]
    best = [data[(data['iteration'] == iteration) & ((data['type'] == 'elite') | (data['type'] == 'final') | (data['type'] == 'best'))]['reldev'].map(log).median() for iteration in iterations]
    iterationPoints.append(data['xaxis'].max())

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
            data.groupby('iteration', as_index = False).agg({'xaxis': 'count'})
            best = data[data['iteration'] == iteration].sort_values('reldev', ascending = True).head(amount)
            names = best['configuration'].tolist()
            x = best['xaxis'].tolist()
            y = best['reldev'].map(log).tolist()
            for i in range(len(x)):
                ax.annotate(names[i], xy = (x[i], y[i]), xytext = (0, -8), textcoords = 'offset pixels', horizontalalignment = 'center', verticalalignment = 'center', fontsize = 5)

    fig.legend(legendElements, legendDescriptions, loc = 'center', bbox_to_anchor = (0.5, 0.05), ncol = 4, handletextpad = 0.5, columnspacing = 1.8)
    ax.tick_params(axis = 'both', which = 'major', labelsize = 9)
    plt.xticks(rotation = 90)

    fig.set_size_inches(10, 6.5)
    fig.subplots_adjust(top = 0.95)
    fig.subplots_adjust(bottom = 0.21)
    fig.subplots_adjust(right = 0.99)
    fig.subplots_adjust(left = 0.06)
    fig.canvas.mpl_connect("motion_notify_event", hover)


def read(iracelog, bkv, overTime):
    robjects.r['load'](iracelog)
    iraceExp = np.array(robjects.r('iraceResults$experiments'))
    iraceExpLog = np.array(robjects.r('iraceResults$experimentLog'))
    iraceInstances = np.array(robjects.r('iraceResults$state$.irace$instancesList'))[0]
    iraceInstanceNames = np.array(robjects.r('iraceResults$scenario$instances'))

    elites = []
    for i in range(1, int(robjects.r('iraceResults$state$indexIteration')[0])):
        elites.append([int(item) for item in str(robjects.r('iraceResults$allElites[[' + str(i) + ']]')).replace('[1]', '').strip().split()])

    experiments = []; id = 0; cumulativeTime = 0
    for i in range(len(iraceExpLog)):
        id += 1
        experiment = {}
        experiment['id'] = id
        experiment['iteration'] = int(iraceExpLog[i][0])
        experiment['instance'] = int(iraceExpLog[i][1])
        experiment['startTime'] = cumulativeTime
        cumulativeTime += float(iraceExpLog[i][3])
        experiment['configuration'] = int(iraceExpLog[i][2])
        experiment['value'] = iraceExp[experiment['instance'] - 1][experiment['configuration'] - 1]
        experiment['type'] = ('best' if experiment['configuration'] == elites[-1][0] else
                              'final' if experiment['configuration'] in elites[-1] else
                              'elite' if experiment['configuration'] in elites[experiment['iteration'] - 1] else
                              'regular')
        experiments.append(experiment)
    data = pd.DataFrame(experiments)
    data['xaxis'] = data['startTime'] if overTime and not math.isnan(cumulativeTime) else data['id']
    if overTime and math.isnan(cumulativeTime): print('  - You are trying to plot over time, but the irace run does not have running time data; setting overtime to false!')
    data['instance'] = data['instance'].map(lambda x: iraceInstances[x - 1])
    data['instancename'] = data['instance'].map(lambda x: iraceInstanceNames[x - 1][iraceInstanceNames[x - 1].rindex('/') + 1:iraceInstanceNames[x - 1].rindex('.')])

    data['bkv'] = float('inf')
    if bkv is not None:
        bkv = pd.read_csv(bkv, sep = ':', header = None, names = ['instancename', 'bkv'])
        bkv['bkv'] = pd.to_numeric(bkv['bkv'], errors = 'raise')
        data['bkv'] = data['instancename'].map(lambda x: bkv[bkv['instancename'] == x]['bkv'].min())

    for instance in data['instance'].unique().tolist():
        data.loc[data['instance'] == instance, 'bkv'] = min(data[data['instance'] == instance]['value'].min(), data[data['instance'] == instance]['bkv'].min())
    
    data['reldev'] = abs(1 - (data['value'] / data['bkv']))

    restarts = [bool(item) for item in np.array(robjects.r('iraceResults$softRestart'))]
    if len(restarts) < len(data['iteration'].unique()): restarts.insert(0, False)
    return data, restarts


def main(iracelog, showElites, showInstances, showConfigurations, pconfig, exportData, exportPlot, output, bkv, overTime):
    print(desc)
    settings = '> Settings:\n'
    settings += '  - plot evolution of the configuration process\n'
    settings += '  - irace log file: ' + iracelog + '\n'
    if bkv is not None: settings += '  - bkv file: ' + str(bkv) + '\n'
    if showElites: settings += '  - show elite configurations\n'
    if showInstances: settings += '  - identify instances\n'
    if showConfigurations: settings += '  - show configurations of the best performers\n'
    if showConfigurations: settings += '  - pconfig = %d\n' % pconfig
    if overTime: settings += '  - plotting over time\n'
    if exportData: settings += '  - export data to csv\n'
    if exportPlot: settings += '  - export plot to pdf and png\n'
    if exportData or exportPlot: settings += '  - output file name: %s\n' % output
    print(settings)

    data, restarts = read(iracelog, bkv, overTime)
    plotEvo(data, restarts, showElites, showInstances, showConfigurations, pconfig)
    
    if exportData:
        if not os.path.exists('./export'): os.mkdir('./export')
        file = open('./export/' + output + '.csv', 'w')
        file.write(data.to_csv())
        file.close()
        print('> data exported to export/' + output + '.csv')
    if exportPlot:
        if not os.path.exists('./export'): os.mkdir('./export')
        plt.savefig('./export/' + output + '.pdf', format = 'pdf')
        plt.savefig('./export/' + output + '.png', format = 'png')
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
    optional.add_argument('--bkv', help = 'file containing best known values for the instances used (null by default)', metavar = '<file>')
    optional.add_argument('--elites', help = 'enables identification of elite configurations (disabled by default)', action = 'store_true')
    optional.add_argument('--configurations', help = 'enables identification of configurations (disabled by default)', action = 'store_true')
    optional.add_argument('--pconfig', help = 'when --configurations, show configurations of the p%% best executions [0, 100] (default: 10)', metavar = '<p>', default = 10, type = int)
    optional.add_argument('--instances', help = 'enables identification of instances (disabled by default)', action = 'store_true')
    optional.add_argument('--overtime', help = 'plot the execution over the accumulated configuration time (disabled by default)', action = 'store_true')
    optional.add_argument('--exportdata', help = 'exports the used data to a csv format file (disabled by default)', action = 'store_true')
    optional.add_argument('--exportplot', help = 'exports the resulting plot to png and pdf files (disabled by default)', action = 'store_true')
    optional.add_argument('--output', help = 'defines a name for the output files', metavar = '<name>', type = str)
    args = parser.parse_args()
    if args.version: print(desc); exit()
    if not args.iracelog: print('Invalid arguments!\nPlease input the irace log file using \'--iracelog <file>\'\n'); parser.print_help(); exit()
    main(args.iracelog, args.elites, args.instances, args.configurations, args.pconfig, args.exportdata, args.exportplot, (args.output if args.output else args.iracelog[args.iracelog.rfind('/')+1:].replace('.Rdata', '')), args.bkv, args.overtime)
