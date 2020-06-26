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


def __updateAnnotations(ind, pathCollection, data):
    index = ind['ind'][0]
    data = data.iloc[[index]]
    annotations.xy = pathCollection.get_offsets()[index]
    exec = data['id'].unique()[0]
    config = data['configuration'].unique()[0]
    instancename = data['instancename'].unique()[0]
    annotations.set_text('execution: %d\ninstance: %s\nconfiguration: %d' % (exec, instancename, config))
    annotations.get_bbox_patch().set_facecolor(data['color'].unique()[0])
    annotations.get_bbox_patch().set_alpha(0.6)


def __hover(event):
    vis = annotations.get_visible()
    if event.inaxes == ax:
        found = False
        for i in range(len(ax.collections)):
            pathCollection = ax.collections[i]
            cont, ind = pathCollection.contains(event)
            if cont:
                __updateAnnotations(ind, pathCollection, plotData[i])
                annotations.set_visible(True)
                fig.canvas.draw_idle()
                found = True
                break
        if not found and vis:
            annotations.set_visible(False)
            fig.canvas.draw_idle()


def __plotEvo(data, restarts, objective, showElites, showInstances, showConfigurations, pconfig, overTime, showToolTips, instancesSoFar, mediansElite, mediansRegular):
    global annotations, ax, fig, plotData
    fig = plt.figure('Plot evolution [cat]')
    ax = fig.add_subplot(1, 1, 1, label = 'plot_evolution')
    ax.set_xlim((data['xaxis'].min(), data['xaxis'].max()))
    ax.set_yscale('log')

    plt.xlabel('candidate evaluations' if not overTime else 'cumulative running time [in seconds]')
    plt.ylabel('solution cost [relative deviation]' if objective == 'cost' else 'running time')

    simpleColors = {'regular': '#202020', 'elite': 'blue', 'final': 'red', 'best': 'green'}
    data['color'] = data.apply(lambda x: colors[(x['instanceseed'] - 1) % len(colors)] if showInstances else simpleColors[x['type']] if showElites else 'black', axis = 1)

    legendElements = []; legendDescriptions = []; plotData = []
    if showElites:
        plotData.append(data[data['type'] == 'regular'])
        plotData.append(data[data['type'] == 'elite'])
        plotData.append(data[data['type'] == 'final'])
        plotData.append(data[data['type'] == 'best'])
        legendElements.append(copy(plt.scatter(data[data['type'] == 'regular']['xaxis'], data[data['type'] == 'regular']['yaxis'], alpha = 1, c = data[data['type'] == 'regular']['color'], marker = 'x', linewidth = 0.5, s = 16)))
        legendElements.append(copy(plt.scatter(data[data['type'] == 'elite']['xaxis'], data[data['type'] == 'elite']['yaxis'], alpha = 1, c = data[data['type'] == 'elite']['color'], edgecolors = 'black', marker = 'o', linewidth = 0.7, s = 24)))
        legendElements.append(copy(plt.scatter(data[data['type'] == 'final']['xaxis'], data[data['type'] == 'final']['yaxis'], alpha = 1, c = data[data['type'] == 'final']['color'], edgecolors = 'black', marker = 'D', linewidth = 0.7, s = 22)))
        legendElements.append(copy(plt.scatter(data[data['type'] == 'best']['xaxis'], data[data['type'] == 'best']['yaxis'], alpha = 1, c = data[data['type'] == 'best']['color'], edgecolors = 'black', marker = '*', linewidth = 0.7, s = 70)))
        legendDescriptions.extend(['regular exec.', 'elite config.', 'final elite config.', 'best found config.'])
    else:
        plotData.append(data)
        legendElements.append(copy(plt.scatter(data['xaxis'], data['yaxis'], alpha = 1, c = data['color'], marker = 'x', linewidth = 0.5, s = 16)))
        legendDescriptions.append('regular exec.')
    if showInstances:
        for element in legendElements: element.set_edgecolor('black'); element.set_facecolor('grey')

    annotations = ax.annotate('', xy = (0, 0), xytext = (0, 20), textcoords = 'offset points', bbox = dict(boxstyle = 'round'), ha = 'left')
    annotations.set_visible(False)

    legendRegular = False; legendRestart = False
    iterationPoints = data.groupby('iteration', as_index = False).agg({'xaxis': 'first'})['xaxis'].tolist()
    indexPoint = 0
    for point, restart in zip(iterationPoints, restarts):
        color = '#B71C1C' if restart else 'k'
        line = plt.axvline(x = point, color = color, linestyle = '--', linewidth = 1.5)
        indexPoint += 1
        if not restart and not legendRegular:
            legendElements.append(line)
            legendDescriptions.append('iteration')
            legendRegular = True
        if restart and not legendRestart:
            legendElements.append(line)
            legendDescriptions.append('iteration with restart')
            legendRestart = True
    ax.set_xticks(iterationPoints + [ax.get_xlim()[1]])

    iterations = data['iteration'].unique().tolist()
    iterationPoints.append(data['xaxis'].max())
    for i in range(len(iterations)):
        medianRegular = mediansRegular[mediansRegular['iteration'] == iterations[i]]['median'].unique()[0]
        medianElite = mediansElite[mediansElite['iteration'] == iterations[i]]['median'].unique()[0]
        plt.plot([iterationPoints[i], iterationPoints[i + 1]], [medianRegular, medianRegular], linestyle = '-', color = '#FF8C00', linewidth = 1.8)
        plt.plot([iterationPoints[i], iterationPoints[i + 1]], [medianElite, medianElite], linestyle = '-', color = '#800080', linewidth = 1.8)
    legendElements.append(mlines.Line2D([], [], color='#FF8C00', linewidth = 1.8))
    legendDescriptions.append('median iteration')
    legendElements.append(mlines.Line2D([], [], color='#800080', linewidth = 1.8))
    legendDescriptions.append('median elites')

    if showConfigurations:
        for iteration in iterations:
            amount = ceil(len(data[data['iteration'] == iteration]) * pconfig / 100)
            data.groupby('iteration', as_index = False).agg({'xaxis': 'count'})
            best = data[data['iteration'] == iteration].sort_values('yaxis', ascending = True).head(amount)
            names = best['configuration'].tolist()
            x = best['xaxis'].tolist()
            y = best['yaxis'].tolist()
            for i in range(len(x)):
                ax.annotate(names[i], xy = (x[i], y[i]), xytext = (0, -8), textcoords = 'offset pixels', horizontalalignment = 'center', verticalalignment = 'center', fontsize = 6)

    fig.legend(legendElements, legendDescriptions, loc = 'center', bbox_to_anchor = (0.5, 0.06), ncol = 4, handletextpad = 0.5, columnspacing = 1.8)
    ax.tick_params(axis = 'both', which = 'major', labelsize = 9)
    plt.xticks(rotation = 90)

    ax2 = ax.twiny()
    ax2.set_xticks(iterationPoints[1:] + [ax.get_xlim()[1]])
    ax2.set_xticklabels(instancesSoFar)
    ax2.set_xlabel('instances evaluated so far')
    ax2.set_xlim((data['xaxis'].min(), data['xaxis'].max()))
    ax2.tick_params(axis = 'both', which = 'major', labelsize = 9)
    ax.set_zorder(ax2.get_zorder() + 1)

    fig.set_size_inches(10, 6.5)
    fig.subplots_adjust(top = 0.93)
    fig.subplots_adjust(bottom = 0.21)
    fig.subplots_adjust(right = 0.99)
    fig.subplots_adjust(left = 0.07)
    if showToolTips: fig.canvas.mpl_connect('motion_notify_event', __hover)


def __plotTest(testData):
    results = []
    instances = testData['instancename'].unique()    
    configurations = testData['configuration'].unique()
    for i in range(len(instances)):
        results.append([])
        for j in range(len(configurations)):
            results[i].append(testData[(testData['instancename'] == instances[i]) & (testData['configuration'] == configurations[j])]['rank'].median())
    eliteIterations = []
    eliteIterationsData = testData.groupby('configuration', as_index = False).agg({'iterationelite': 'first'})
    for config in configurations:
        eliteIterations.append(eliteIterationsData[eliteIterationsData['configuration'] == config]['iterationelite'].unique()[0])
    
    fig = plt.figure('Plot testing data [cat]')
    ax = fig.add_subplot(1, 1, 1, label = 'plot_test')
    im = ax.imshow(results, cmap = 'RdYlGn_r')
    
    cbar = ax.figure.colorbar(im, ax = ax)
    cbar.ax.set_ylabel('rank', rotation = -90, va = 'bottom')

    ax.set_xticks(np.arange(len(configurations)))
    ax.set_yticks(np.arange(len(instances)))
    ax.set_xticklabels([str(configurations[i]) + ' (' + str(eliteIterations[i]) + ')' for i in range(len(configurations))])
    ax.set_yticklabels(instances)
    ax.set_xticks(np.arange(len(results[1]) + 1) - .5, minor = True)
    ax.set_yticks(np.arange(len(results) + 1) - .5, minor = True)
    ax.tick_params(top = False, bottom = True, labeltop = False, labelbottom = True, left = True, right = False, labelleft = True, labelright = False)
    ax.grid(which = 'minor', color = 'w', linestyle = '-', linewidth = 3)
    ax.tick_params(which = 'minor', bottom = False, left = False)
    plt.setp(ax.get_xticklabels(), rotation = 90, va = 'center', ha = 'right', rotation_mode = 'anchor')
    ax.set_xlabel('elite configurations (corresponding iterations)')
    ax.set_ylabel('train (gray) and test (blue) instances')
    [label.set_color('#5D6D7E' if 'train' in label.get_text() else '#0000FF') for label in plt.gca().get_yticklabels()]

    texts = []
    for i in range(len(results)):
        for j in range(len(results[0])):
            kw = dict(horizontalalignment = 'center', verticalalignment = 'center')
            text = im.axes.text(j, i, int(results[i][j]), **kw)
            texts.append(text)
    
    fig.tight_layout()


def __read(iracelog, objective, bkv, overTime, imputation, testing):
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
    if overTime and math.isnan(cumulativeTime): print('  - You are trying to plot over time, but the irace log file does not have running time data; setting overtime to false!'); overTime = False
    data['instanceseed'] = data['instance']
    data['instance'] = data['instance'].map(lambda x: iraceInstances[x - 1])
    data['instancename'] = data['instance'].map(lambda x: iraceInstanceNames[x - 1][iraceInstanceNames[x - 1].rindex('/') + 1:iraceInstanceNames[x - 1].rindex('.')])

    data['bkv'] = float('inf')
    if bkv is not None:
        bkv = pd.read_csv(bkv, sep = ':', header = None, names = ['instancename', 'bkv'])
        bkv['bkv'] = pd.to_numeric(bkv['bkv'], errors = 'raise')
        data['bkv'] = data['instancename'].map(lambda x: bkv[bkv['instancename'] == x]['bkv'].min())

    for instance in data['instance'].unique().tolist():
        data.loc[data['instance'] == instance, 'bkv'] = min(data[data['instance'] == instance]['value'].min(), data[data['instance'] == instance]['bkv'].min())
    
    data['yaxis'] = abs(1 - (data['value'] / data['bkv'])) if objective == 'cost' else data['value']
    data.loc[data['yaxis'] == 0, 'yaxis'] = data[data['yaxis'] > 0]['yaxis'].min() / 2
    
    restarts = [bool(item) for item in np.array(robjects.r('iraceResults$softRestart'))]
    if len(restarts) < len(data['iteration'].unique()): restarts.insert(0, False)
    
    instancesSoFar = []
    usedInstances = data.groupby('iteration', as_index = False).agg({'instanceseed': 'unique'})
    for iteration in usedInstances['iteration'].tolist():
        instancesSoFar.append([])
        for instanceList in usedInstances[usedInstances['iteration'] <= iteration]['instanceseed'].tolist():
            instancesSoFar[-1].extend(instanceList)
        instancesSoFar[-1] = np.unique(instancesSoFar[-1])
    instancesSoFar = [len(item) for item in instancesSoFar]

    mediansEliteDict = {'iteration': [], 'median': []}
    mediansRegularDict = {'iteration': [], 'median': []}
    iterations = data['iteration'].unique()
    for iteration in iterations:
        eliteConfs = data[(data['iteration'] == iteration) & (data['type'] != 'regular')]['configuration'].unique()
        nonEliteConfs = data[(data['iteration'] == iteration) & (data['type'] == 'regular')]['configuration'].unique()
        instancesOfIteration = data[data['iteration'] <= iteration]['instanceseed'].unique()
        execElite = data[(data['iteration'] <= iteration) & (data['configuration'].isin(eliteConfs)) & (data['instanceseed'].isin(instancesOfIteration))]
        execElite = execElite.groupby(['configuration', 'instanceseed'], as_index = False).agg({'yaxis': 'median'})
        execNonElite = data[(data['iteration'] <= iteration) & (data['configuration'].isin(nonEliteConfs)) & (data['instanceseed'].isin(instancesOfIteration))]
        execNonElite = execNonElite.groupby(['configuration', 'instanceseed'], as_index = False).agg({'yaxis': 'median'})
        for instanceSeed in instancesOfIteration:
            execElitesInstance = execElite[execElite['instanceseed'] == instanceSeed]
            execNonElitesInstance = execNonElite[execNonElite['instanceseed'] == instanceSeed]
            if imputation == 'elite': imputationValue = data[(data['configuration'].isin(eliteConfs)) & (data['instanceseed'] == instanceSeed) & (data['iteration'] <= iteration)]['yaxis'].max()
            elif imputation == 'alive': imputationValue = data[((data['configuration'].isin(eliteConfs)) | (data['configuration'].isin(nonEliteConfs))) & (data['instanceseed'] == instanceSeed) & (data['iteration'] <= iteration)]['yaxis'].max()
            if not math.isnan(imputationValue):
                for conf in eliteConfs:
                    if len(execElitesInstance[execElitesInstance['configuration'] == conf]) == 0:
                        execElite.loc[len(execElite)] = [conf, instanceSeed, imputationValue]
                for conf in nonEliteConfs:
                    if len(execNonElitesInstance[execNonElitesInstance['configuration'] == conf]) == 0:
                        execNonElite.loc[len(execNonElite)] = [conf, instanceSeed, imputationValue]
        execElite = execElite.groupby('configuration', as_index = False).agg({'yaxis': 'median'})
        execNonElite = execNonElite.groupby('configuration', as_index = False).agg({'yaxis': 'median'})
        mediansEliteDict['iteration'].append(iteration)
        mediansEliteDict['median'].append(execElite['yaxis'].median())
        mediansRegularDict['iteration'].append(iteration)
        mediansRegularDict['median'].append(execNonElite['yaxis'].median())
    mediansElite = pd.DataFrame.from_dict(mediansEliteDict)
    mediansRegular = pd.DataFrame.from_dict(mediansRegularDict)

    testData = None
    if testing:
        testInstanceIds = list(np.array(robjects.r('names(iraceResults$scenario$testInstances)')))
        testInstanceNames = list([instance[instance.rindex('/') + 1:instance.rindex('.')] for instance in np.array(robjects.r('iraceResults$scenario$testInstances'))])
        testInstances = np.array(robjects.r('rownames(iraceResults$testing$experiments)'))
        testConfigurations = np.array(robjects.r('colnames(iraceResults$testing$experiments)'))
        testResults = np.array(robjects.r('iraceResults$testing$experiments'))
        
        trainInstanceNames = data['instancename'].unique()
        testData = {'configuration': [], 'instanceid': [], 'instancename': [], 'result': []}
        for i in range(len(testInstances)):
            for j in range(len(testConfigurations)):
                testData['instanceid'].append(testInstances[i])
                testData['instancename'].append(testInstanceNames[testInstanceIds.index(testInstances[i])])
                testData['configuration'].append(int(testConfigurations[j]))
                testData['result'].append(testResults[i][j])
        testData = pd.DataFrame.from_dict(testData)
        testData = testData.groupby(['configuration', 'instancename'], as_index = False).agg({'result': 'median'})
        testData['instancetype'] = testData['instancename'].map(lambda x: 'train' if x in trainInstanceNames else 'test')
        testData['iterationelite'] = -1
        configs = [int(config) for config in testConfigurations]
        for config in configs:
            iteration = ''
            for i in range(len(elites)):
                #if config in elites[i]:
                if config == elites[i][0]:
                    iteration += str(i + 1) + ';'
            iteration = iteration[:-1]
            testData.loc[testData['configuration'] == config, 'iterationelite'] = iteration
        
        testData['rank'] = 'NA'
        instanceNames = testData['instancename'].unique()
        for instanceName in instanceNames:
            testData.loc[testData['instancename'] == instanceName, 'rank'] = testData[testData['instancename'] == instanceName]['result'].rank(method = 'min')

    return data, restarts, instancesSoFar, overTime, mediansRegular, mediansElite, testData
  

def getPlot(iracelog, objective = 'cost', showElites = False, showInstances = False, showConfigurations = False, pconfig = 10, showPlot = False, exportData = False, exportPlot = False, output = 'output', bkv = None, overTime = False, userPlt = None, showToolTips = True, imputation = 'elite', testing = False):
    global plt
    if userPlt is not None: plt = userPlt 
    data, restarts, instancesSoFar, overTime, mediansRegular, mediansElite, testData = __read(iracelog, objective, bkv, overTime, imputation, testing)
    if testing: __plotTest(testData)
    else: __plotEvo(data, restarts, objective, showElites, showInstances, showConfigurations, pconfig, overTime, showToolTips, instancesSoFar, mediansElite, mediansRegular)
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
        if showPlot: plt.show()
        else: return plt


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser._action_groups.pop()
    required = parser.add_argument_group('required arguments')
    optional = parser.add_argument_group('optional arguments')
    required.add_argument('--iracelog', help = 'input of irace log file (.Rdata)', metavar = '<file>', required = ('--version' not in sys.argv and '-v' not in sys.argv))
    optional.add_argument('-v', '--version', help = 'show description and exit', action = 'store_true')
    optional.add_argument('--objective', help = 'performance measure used by irace [cost or time] (default: cost)', metavar = '<obj>', default = 'cost', type = str)
    optional.add_argument('--overtime', help = 'plot the execution over the accumulated configuration time (disabled by default)', action = 'store_true')
    optional.add_argument('--bkv', help = 'file containing best known values for the instances used (null by default)', metavar = '<file>')
    optional.add_argument('--elites', help = 'enables identification of elite configurations (disabled by default)', action = 'store_true')
    optional.add_argument('--configurations', help = 'enables identification of configurations (disabled by default)', action = 'store_true')
    optional.add_argument('--pconfig', help = 'when --configurations, show configurations of the p%% best executions [0, 100] (default: 10)', metavar = '<p>', default = 10, type = int)
    optional.add_argument('--instances', help = 'enables identification of instances (disabled by default)', action = 'store_true')
    optional.add_argument('--imputation', help = 'imputation strategy for computing medians [elite, alive] (default: elite)', metavar = '<imputation>', type = str, default = 'elite')
    optional.add_argument('--testing', help = 'plots the testing data instead of the configuration process (disabled by default)', action = 'store_true')
    optional.add_argument('--exportdata', help = 'exports the used data to a csv format file (disabled by default)', action = 'store_true')
    optional.add_argument('--exportplot', help = 'exports the resulting plot to png and pdf files (disabled by default)', action = 'store_true')
    optional.add_argument('--output', help = 'defines a name for the output files (default: export)', metavar = '<name>', type = str, default = 'export')
    args = parser.parse_args()
    if args.version: print(desc); exit()
    if not args.iracelog: print('Invalid arguments!\nPlease input the irace log file using \'--iracelog <file>\'\n'); parser.print_help(); exit()
    if args.objective.lower() not in ('cost', 'time'): print('Invalid objective. Use either COST or TIME.'); parser.print_help(); exit()
    
    print(desc)
    settings = '> Settings:\n'
    settings += '  - plot evolution of the configuration process\n'
    settings += '  - irace log file: ' + args.iracelog + '\n'
    settings += '  - imputation strategy: ' + args.imputation + '\n'
    if args.bkv is not None: settings += '  - bkv file: ' + str(args.bkv) + '\n'
    if args.elites: settings += '  - show elite configurations\n'
    if args.instances: settings += '  - identify instances\n'
    if args.configurations: settings += '  - show configurations of the best performers\n'
    if args.configurations: settings += '  - pconfig = %d\n' % args.pconfig
    if args.overtime: settings += '  - plotting over time\n'
    if args.testing: settings += '  - plotting test data\n'
    if args.exportdata: settings += '  - export data to csv\n'
    if args.exportplot: settings += '  - export plot to pdf and png\n'
    if args.exportdata or args.exportplot: settings += '  - output file name: %s\n' % args.output
    print(settings)
    
    getPlot(
        iracelog = args.iracelog,
        objective = args.objective,
        showElites = args.elites,
        showInstances = args.instances,
        showConfigurations = args.configurations,
        pconfig = args.pconfig,
        showPlot = True,
        exportData = args.exportdata,
        exportPlot = args.exportplot,
        output = args.output,
        bkv = args.bkv,
        overTime = args.overtime,
        userPlt = None,
        showToolTips = True,
        imputation = args.imputation,
        testing = args.testing
    )
    print('-------------------------------------------------------------------------------')