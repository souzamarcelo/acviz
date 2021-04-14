    #!/usr/bin/python3
import os
import sys
import math
import argparse
import time
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib import lines as mlines
from rpy2 import robjects
from math import log
from math import ceil
from copy import copy
from natsort import natsorted
from statistics import median
from io import BytesIO
from PIL import Image

desc = '''
-------------------------------------------------------------------------------
| acviz: Algorithm Configuration Visualizations for irace                     |
| Version: 1.0                                                                |
| Copyright (C) 2020                                                          |
| Marcelo de Souza         <marcelo.desouza@udesc.br>                         |
| Marcus Ritt              <marcus.ritt@inf.ufrgs.br>                         |
| Manuel Lopez-Ibanez      <manuel.lopez-ibanez@uma.es>                       |
| Leslie Perez Caceres     <leslie.perez@pucv.cl>                             |
|                                                                             |
| This is free software, and you are welcome to redistribute it under certain |
| conditions.  See the GNU General Public License for details. There is NO    |
| WARRANTY; not even for MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. |
-------------------------------------------------------------------------------
'''

# Global variables
annotations = None
fig = None
ax = None
plotData = None
colors = ['#EF9A9A','#1A237E','#B71C1C','#00C853','#F57F17','#424242','#FDD835','#B0BEC5','#004D40','#000000','#F50057',
          '#2196F3','#795548','#76FF03','#9C27B0','#1565C0','#F9A825','#F48FB1','#81C784','#F44336','#A1887F','#1B5E20',
          '#69F0AE','#EC407A','#827717','#607D8B','#D50000','#CE93D8','#43A047','#FFEA00','#18FFFF','#3F51B5','#FF6F00',
          '#757575','#64DD17','#EEFF41','#7C4DFF','#33691E','#90CAF9','#AA00FF','#FFF176','#8BC34A','#009688','#9E9D24']


# Function __updateAnnotations
def __updateAnnotations(ind, pathCollection, data):
    """
    Updates the tool tip box according to the position and data received.
    Arguments:
        - ind: contains the index of the point inside data
        - pathCollection: contains the position of the cursor
        - data: contains the data of the configuration process
    """

    # Get the index and the corresponding data object
    index = ind['ind'][0]
    data = data.iloc[[index]]

    # Set the position of the tool tip and get the desired information
    annotations.xy = pathCollection.get_offsets()[index]
    exec = data['id'].unique()[0]
    config = data['configuration'].unique()[0]
    instancename = data['instancename'].unique()[0]

    # Set the text, color and opacity of the tool tip box
    annotations.set_text('execution: %d\ninstance: %s\nconfiguration: %d' % (exec, instancename, config))
    annotations.get_bbox_patch().set_facecolor(data['color'].unique()[0])
    annotations.get_bbox_patch().set_alpha(0.6)


# Function __hover
def __hover(event):
    """
    Called when user moves the cursor over the plot.
    If the cursor is over a point, finds the corresponding data and
    calls __updateAnnotations to update the tool tip box; hiddes the
    tool tip box otherwise.
    Arguments:
        - event: object given by matplotlib, containing the cursor data
    """

    # If cursor moves on ax, find the corresponding data and update tool tip box
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
        # Otherwise hidde box, if already visible
        if not found and vis:
            annotations.set_visible(False)
            fig.canvas.draw_idle()


# Function __plotTraining
def __plotTraining(data, typeResult, restarts, showElites, showInstances, pconfig, overTime, showToolTips, instancesSoFar, mediansElite, mediansRegular, alpha, reverse, logScale, timeLimit):
    """
    Plots the evolution of the configuration process.
    Arguments:
        - data: all data from the irace log file
        - typeResult: type of results to show
        - restarts: list that tells whether a restart was made in each iteration
        - showElites: indicates if executions of elite configurations must be highlighted
        - showInstances: indicates if executions on different instances must be highlighted
        - pconfig: percentage of the executions whose configuration is presented
        - overTime: enables the execution time in the x-axis
        - showToolTips: enables the tool tip boxes with information about the executions
        - instancesSoFar: list with instances evaluated until each iteration
        - mediansElite: list with the median performance of elite configurations in each iteration
        - mediansRegular: list with the median performance of non-elite configurations in each iteration
        - alpha: opacity of the points
        - logScale: enables logscale
        - timeLimit: defines the time limit when plotting running times
    """

    # Definition of global variables for this function
    global annotations, ax, fig, plotData
    # Create figure; add unique subplot; set limits and log scale; set axis labels
    fig = plt.figure('Plot evolution [acviz]')
    ax = fig.add_subplot(1, 1, 1, label = 'plot_evolution')
    ax.set_xlim((data['xaxis'].min(), data['xaxis'].max()))
    if logScale: ax.set_yscale('log')
    plt.xlabel('Candidate evaluations' if not overTime else 'Cumulative running time [in seconds]')
    
    # Define y label according to the type of results
    if typeResult == 'rdev': plt.ylabel('Relative deviation')
    elif typeResult == 'adev': plt.ylabel('Absolute deviation')
    else: plt.ylabel('Absolute value')
    
    # Create colors for instances
    simpleColors = {'regular': '#202020', 'elite': 'blue', 'final': 'red', 'best': 'green'}
    data['color'] = data.apply(lambda x: colors[(x['instanceseed'] - 1) % len(colors)] if showInstances else simpleColors[x['type']] if showElites else 'black', axis = 1)

    #Reverse y-axis
    if reverse:
        plt.gca().invert_yaxis()
        data['yaxis'] = 1 - data['yaxis']
        mediansRegular['median'] = 1 - mediansRegular['median']
        mediansElite['median'] = 1 - mediansElite['median']

    # Check whether plotting runnning times with time limits
    plotLimit = (typeResult == 'aval' and not logScale and timeLimit > 0)
    if plotLimit:
        # If so, configure executions exceeding the time limit
        data['yaxis'] = data['yaxis'].map(lambda x : float('inf') if x >= timeLimit else x)
        minv = data['yaxis'].min()
        maxv = data[data['yaxis'] < float('inf')]['yaxis'].max()
        cutv = maxv + 0.3 * (maxv - minv)
        data['yaxis'] = data['yaxis'].map(lambda x : cutv if x == float('inf') else x)

    # Create list of data and populate
    legendElements = []; legendDescriptions = []; plotData = []
    if showElites:
        # Use different markers for executions of elite configurations, if needed
        plotData.append(data[data['type'] == 'regular'])
        plotData.append(data[data['type'] == 'elite'])
        plotData.append(data[data['type'] == 'final'])
        plotData.append(data[data['type'] == 'best'])
        legendElements.append(copy(plt.scatter(data[data['type'] == 'regular']['xaxis'], data[data['type'] == 'regular']['yaxis'], alpha = alpha, c = data[data['type'] == 'regular']['color'], marker = 'x', linewidth = 0.5, s = 16, zorder = 3, clip_on = False)))
        legendElements.append(copy(plt.scatter(data[data['type'] == 'elite']['xaxis'], data[data['type'] == 'elite']['yaxis'], alpha = min(alpha + 0.2, 1), c = data[data['type'] == 'elite']['color'], edgecolors = 'black', marker = 'o', linewidth = 0.7, s = 24, zorder = 3, clip_on = False)))
        legendElements.append(copy(plt.scatter(data[data['type'] == 'final']['xaxis'], data[data['type'] == 'final']['yaxis'], alpha = min(alpha + 0.2, 1), c = data[data['type'] == 'final']['color'], edgecolors = 'black', marker = 'D', linewidth = 0.7, s = 22, zorder = 3, clip_on = False)))
        legendElements.append(copy(plt.scatter(data[data['type'] == 'best']['xaxis'], data[data['type'] == 'best']['yaxis'], alpha = min(alpha + 0.2, 1), c = data[data['type'] == 'best']['color'], edgecolors = 'black', marker = '*', linewidth = 0.7, s = 70, zorder = 3, clip_on = False)))
        legendDescriptions.extend(['regular config.', 'elite config.', 'final elite config.', 'best found config.'])
    else:
        # Otherwise, all executions have the same marker
        plotData.append(data)
        legendElements.append(copy(plt.scatter(data['xaxis'], data['yaxis'], alpha = alpha, c = data['color'], marker = 'x', linewidth = 0.5, s = 16, zorder = 3, clip_on = False)))
        legendDescriptions.append('execution')
    if showInstances:
        for element in legendElements: element.set_edgecolor('black'); element.set_facecolor('grey')

    # Add label and adjust y limits; exceeding executions are positioned in the top of the plotting area
    if plotLimit:
        ymin, _ = ax.get_ylim()
        yticks = list(plt.yticks()[0])
        if all([(x == 0) or (x % max(int(x), 1) == 0) for x in yticks]):
            yticks = [int(x) for x in yticks]
        ylabels = [str(x) for x in yticks]
        yticks.append(cutv)
        ylabels.append('$\\geq$TL')
        plt.yticks(yticks, ylabels)
        ax.set_ylim(bottom = ymin, top = cutv)

    # Create hidden annotations object for being used in the tool tip boxes
    annotations = ax.annotate('', xy = (0, 0), xytext = (0, 20), textcoords = 'offset points', bbox = dict(boxstyle = 'round'), ha = 'left')
    annotations.set_visible(False)

    # Get the beginning of each iteration
    iterationPoints = data.groupby('iteration', as_index = False).agg({'xaxis': 'first'})['xaxis'].tolist()
    legendRegular = False; legendRestart = False
    indexPoint = 0
    # Merge the beginning of each iteration with the information about restarts
    for point, restart in zip(iterationPoints, restarts):
        # Black for regular and red for restarting iterations
        color = '#B71C1C' if restart else 'k'
        # Create vertical line for each iteration
        line = plt.axvline(x = point, color = color, linestyle = '--', linewidth = 1.5, zorder = 1)
        indexPoint += 1
        # Add legend entries
        if not restart and not legendRegular:
            legendElements.append(line)
            legendDescriptions.append('iteration')
            legendRegular = True
        if restart and not legendRestart:
            legendElements.append(line)
            legendDescriptions.append('iteration (restart)')
            legendRestart = True
    # Set the xticks according to the beginning of each iteration
    ax.set_xticks(iterationPoints + [ax.get_xlim()[1]])

    iterations = data['iteration'].unique().tolist()
    iterationPoints.append(data['xaxis'].max())
    
    # Create horizontal lines for the median performance of elite and non-elite configurations in each iteration
    for i in range(len(iterations)):
        medianRegular = mediansRegular[mediansRegular['iteration'] == iterations[i]]['median'].unique()[0]
        medianElite = mediansElite[mediansElite['iteration'] == iterations[i]]['median'].unique()[0]
        plt.plot([iterationPoints[i], iterationPoints[i + 1]], [medianRegular, medianRegular], linestyle = '-', color = '#FF8C00', linewidth = 1.8, zorder = 5)
        plt.plot([iterationPoints[i], iterationPoints[i + 1]], [medianElite, medianElite], linestyle = '-', color = '#800080', linewidth = 1.8, zorder = 5)
    # Add legends for both types of horizontal lines
    legendElements.append(mlines.Line2D([], [], color='#FF8C00', linewidth = 1.8))
    legendDescriptions.append('median iteration')
    legendElements.append(mlines.Line2D([], [], color='#800080', linewidth = 1.8))
    legendDescriptions.append('median elites')
    
    # Identify configurations of the pconfig% best executions of each iteration
    if pconfig > 0:
        for iteration in iterations:
            amount = ceil(len(data[data['iteration'] == iteration]) * pconfig / 100)
            data.groupby('iteration', as_index = False).agg({'xaxis': 'count'})
            best = data[data['iteration'] == iteration].sort_values('yaxis', ascending = True).head(amount)
            names = best['configuration'].tolist()
            x = best['xaxis'].tolist()
            y = best['yaxis'].tolist()
            for i in range(len(x)):
                ax.annotate(names[i], xy = (x[i], y[i]), xytext = (0, -8), textcoords = 'offset pixels', horizontalalignment = 'center', verticalalignment = 'center', fontsize = 6)

    # Adjust legend, tick parameters and rotation
    fig.legend(legendElements, legendDescriptions, loc = 'center', bbox_to_anchor = (0.5, 0.055), ncol = 4, handletextpad = 0.5, columnspacing = 1.8)
    ax.tick_params(axis = 'both', which = 'major', labelsize = 9)
    plt.xticks(rotation = 90)

    # Create secondary x-axis for the instances evaluated so far; adjust parameters
    ax2 = ax.twiny()
    ax2.set_xticks(iterationPoints[1:])
    ax2.set_xticklabels(instancesSoFar)
    ax2.set_xlabel('Instances evaluated')
    ax2.set_xlim((data['xaxis'].min(), data['xaxis'].max()))
    ax2.tick_params(axis = 'both', which = 'major', labelsize = 9)
    ax.set_zorder(ax2.get_zorder() + 1)

    # Adjust figure size and margins
    fig.set_size_inches(10, 6.5)
    fig.subplots_adjust(top = 0.93)
    fig.subplots_adjust(bottom = 0.21)
    fig.subplots_adjust(right = 0.99)
    fig.subplots_adjust(left = 0.07)
    # Add hover listener function, if showToolTips = True
    if showToolTips: fig.canvas.mpl_connect('motion_notify_event', __hover)


# Function __plotTest
def __plotTest(testData, typeResult, firstElites, finalElites, testConfigurations, testColors):
    """
    Plots the performance of the best found configurations on the test instances, if test data available.
    Arguments:
        - testData: the test data from the irace log file
        - typeResult: type of results to show
        - firstElites: the first ranked elite configurations of each iteration
        - finalElites: all elite configurations of the final iteration
        - testConfigurations: list with the IDs of the tested configurations
        - testColors: color scheme
    """

    # Get list of training and test instances
    trainInstances = natsorted(list(testData[testData['instancetype'] == 'train']['instancename'].unique()), key = lambda x: x.lower())
    instances = natsorted(list(testData[testData['instancetype'] == 'test']['instancename'].unique()), key = lambda x: x.lower()) + trainInstances

    # Build a list of elite configurations
    elites = []
    # First the first-ranked elites of each iteration
    for elite in firstElites[:-1]:
        if elite in testConfigurations:
            if elite in elites:
                elites.remove(elite)
            elites.append(elite)
    # Then, all elites of the final iteration
    for elite in finalElites:
        if elite in testConfigurations:
            if elite not in elites:
                elites.append(elite)

    # Create a label for each elite configuration
    elitesLabels = []
    for elite in elites:
        label = '$'
        for j in range(len(firstElites) - 1):
            # Add the number of the iteration
            if elite == firstElites[j]: label += str(j + 1) + ','
        # Add the ranking position, in case of elite configurations of the final iteration
        label += str(len(firstElites)) + '_{' + str(finalElites.index(elite) + 1) + '}' if elite in finalElites else ''
        label = label[:-1] if label[-1] == ',' else label
        label += '$'
        elitesLabels.append(label)
    
    # Create a rank for each instance
    testData['rank'] = 'NA'
    for instanceName in instances:
        testData.loc[testData['instancename'] == instanceName, 'rank'] = testData[testData['instancename'] == instanceName]['yaxis'].rank(method = 'min')

    # Create matrices for the results (data) and normalized results (norm)
    dataPlot = [[], []]
    normPlot = [[], []]
    minValue = [float('inf'), float('inf')]
    maxValue = [float('-inf'), float('-inf')]
    # Index 0 is for the results plot; index 1 is for the ranking plot
    for i in range(len(instances)):
        # Create a line for each instance
        dataPlot[0].append([])
        dataPlot[1].append([])
        normPlot[0].append([])
        normPlot[1].append([])
        for elite in elites:
            # Create columns for each configuration, with the median results or rankings
            dataPlot[0][i].append(testData[(testData['instancename'] == instances[i]) & (testData['configuration'] == elite)]['yaxis'].median())
            dataPlot[1][i].append(testData[(testData['instancename'] == instances[i]) & (testData['configuration'] == elite)]['rank'].median())
            # Update min values
            minValue[0] = min(minValue[0], dataPlot[0][i][-1])
            minValue[1] = min(minValue[1], dataPlot[1][i][-1])
            maxValue[0] = max(maxValue[0], dataPlot[0][i][-1])
            maxValue[1] = max(maxValue[1], dataPlot[1][i][-1])
        addMax = [1 if len(set(dataPlot[x][i])) == 1 else 0 for x in [0, 1]]

        # Create entries for normPlot matrix
        normPlot[0][i] = [(value - min(dataPlot[0][i])) / (max(dataPlot[0][i]) + addMax[0] - min(dataPlot[0][i])) for value in dataPlot[0][i]]
        normPlot[1][i] = [(value - min(dataPlot[1][i])) / (max(dataPlot[1][i]) + addMax[1] - min(dataPlot[1][i])) for value in dataPlot[1][i]]
    # Create titles
    titles = ['Mean absolute values' if typeResult == 'aval' else 'Mean relative deviations' if typeResult == 'rdev' else 'Mean absolute deviations', 'Ranks by instance']
    
    # Create figure and set data according to the color scheme
    fig = plt.figure('Plot testing data [acviz]')
    data = dataPlot if testColors == 'general' else normPlot

    # Execute routines for both subplots
    for index in range(0, 2):
        # Add subplot; create imshow object; set title and tick parameters
        ax = fig.add_subplot(1, 2, index + 1, label = 'plot_test')
        im = ax.imshow(data[index], cmap = 'RdYlGn_r', aspect = 'auto')
        ax.set_title(titles[index], fontsize = 10)
        ax.tick_params(axis = 'both', which = 'both', labelsize = 8)
        ax.tick_params(top = False, bottom = True, labeltop = False, labelbottom = True, left = True, right = False, labelleft = True, labelright = False)
        # Set ticks for each configuration/instance; adjusts x tick labels
        ax.set_xticks(np.arange(len(dataPlot[index][1]) + 1) - .5, minor = True)
        ax.set_yticks(np.arange(len(dataPlot[index]) + 1) - .5, minor = True)
        ax.set_xticks(np.arange(len(elites)))
        ax.set_xticklabels([str(elites[i]) + ' (' + elitesLabels[i] + ')' for i in range(len(elites))])
        
        # Add grid; adjust plot parameters; adjust tick properties
        ax.grid(which = 'minor', color = 'w', linestyle = '-', linewidth = 2)
        ax.tick_params(which = 'minor', bottom = False, left = False)
        plt.setp(ax.get_xticklabels(), rotation = 90, va = 'center', ha = 'right', rotation_mode = 'anchor')
        ax.set_yticks(np.arange(len(instances)))
        ax.set_yticklabels(instances)

        # Set different colors for training and test instances; hidde y tick labels for second subplot
        [label.set_color('#000000' if label.get_text() in trainInstances else '#0000FF') for label in plt.gca().get_yticklabels()]
        if index > 0: ax.set_yticks([])

        # Create normalized matrix for general color scheme; original data otherwise
        normed = [[], []]
        if testColors == 'general':
            for i in range(len(data[index])):
                normed[index].append([])
                for j in range(len(data[index][0])):
                    normed[index][i].append((data[index][i][j] - minValue[index]) / (maxValue[index] - minValue[index]))
        else:
            normed = data
            
        # Create texts for each cell, showing the result/ranking values
        texts = []
        for i in range(len(dataPlot[index])):
            for j in range(len(dataPlot[index][0])):
                # Text properties
                kw = dict(horizontalalignment = 'center', verticalalignment = 'center', fontsize = 8, color = 'white' if normed[index][i][j] < 0.15 or normed[index][i][j] > 0.85 else 'black')
                if not math.isnan(dataPlot[index][i][j]):
                    # Create text object
                    text = im.axes.text(j, i, '{:.3f}'.format(dataPlot[index][i][j]) if index == 0 else int(dataPlot[index][i][j]), **kw)                    
                # Add to the text list
                texts.append(text)

    # Set figure size and margins
    fig.set_size_inches(12, 7)
    fig.tight_layout()


# Function __readTest
def __readTest(iracelog, typeResult, bkvFile):
    """
    Reads the test data from the irace log file.
    Arguments:
        - iracelog: irace log file
        - typeResult: type of results to show
        - bkvFile: file with the best known values for each instance
    """

    # Load the R object from the irace log file
    robjects.r['load'](iracelog)
    # Get the test instances, test and training instance names, tested configurations, test results, elite configurations
    testInstances = list(np.array(robjects.r('rownames(iraceResults$testing$experiments)')))
    testInstanceNames = list([instance[instance.rindex('/') + 1 if '/' in instance else 0:instance.rindex('.') if '.' in instance else len(instance)] for instance in np.array(robjects.r('iraceResults$scenario$testInstances'))])
    trainInstanceNames = [x[x.rindex('/') + 1 if '/' in x else 0:x.rindex('.') if '.' in x else len(x)] for x in list(set(np.array(robjects.r('iraceResults$scenario$instances'))))]
    testConfigurations = [int(x) for x in np.array(robjects.r('colnames(iraceResults$testing$experiments)'))]
    testResultsIrace = np.array(robjects.r('iraceResults$testing$experiments'))
    firstElites = [int(str(robjects.r('iraceResults$allElites[[' + str(i) + ']]')).replace('[1]', '').strip().split()[0]) for i in range(1, int(robjects.r('iraceResults$state$indexIteration')[0]))]
    finalElites = [int(x) for x in str(robjects.r('iraceResults$allElites[[' + str(int(robjects.r('iraceResults$state$indexIteration')[0]) - 1) + ']]')).replace('[1]', '').strip().split()]

    # Create dictionary to read the test data
    testData = {'configuration': [], 'instanceid': [], 'instancename': [], 'result': []}
    # For each instance and configuration, read the result of the execution
    for i in range(len(testInstances)):
        for j in range(len(testConfigurations)):
            testData['instanceid'].append(testInstances[i])
            testData['instancename'].append(testInstanceNames[i])
            testData['configuration'].append(int(testConfigurations[j]))
            testData['result'].append(testResultsIrace[i][j])
    # Create data frame from the dictionary
    testData = pd.DataFrame.from_dict(testData)
    # Group by configuration and instance name; aggregate the results (mean); define the instance type
    testData = testData.groupby(['configuration', 'instancename'], as_index = False).agg({'result': 'mean'})
    testData['instancetype'] = testData['instancename'].map(lambda x: 'train' if x in trainInstanceNames else 'test')
    
    # Read training data to get (or update) the best known values
    iraceExpLog = np.array(robjects.r('iraceResults$experimentLog'))
    iraceExp = np.array(robjects.r('iraceResults$experiments'))
    iraceInstances = np.array(robjects.r('iraceResults$state$.irace$instancesList'))[0]
    iraceInstanceNames = [x[x.rindex('/') + 1 if '/' in x else 0:x.rindex('.') if '.' in x else len(x)] for x in list(np.array(robjects.r('iraceResults$scenario$instances')))]
    # Get all executions performed in the training phase
    experiments = []
    for i in range(len(iraceExpLog)):
        experiment = {}
        experiment['instance'] = int(iraceExpLog[i][1])
        experiment['configuration'] = int(iraceExpLog[i][2])
        experiment['result'] = iraceExp[experiment['instance'] - 1][experiment['configuration'] - 1]
        experiments.append(experiment)
    # Create data frame; get instance IDs and instance names
    trainingData = pd.DataFrame(experiments)
    trainingData['instance'] = trainingData['instance'].map(lambda x: iraceInstances[x - 1])
    trainingData['instancename'] = trainingData['instance'].map(lambda x: iraceInstanceNames[x - 1].replace('/home/msouza/autobqp/autobqp-palubeckis-maxcut', ''))

    # Create column for best known values
    testData['bkv'] = float('inf')
    # If a bkv file is given, read the values for each instance
    if bkvFile is not None:
        bkv = pd.read_csv(bkvFile, sep = ':', header = None, names = ['instancename', 'bkv'])
        bkv['bkv'] = pd.to_numeric(bkv['bkv'], errors = 'raise')
        testData['bkv'] = testData['instancename'].map(lambda x: bkv[bkv['instancename'] == x]['bkv'].min())
    # Also use the values obtained in the training phase
    for instance in testData['instancename'].unique().tolist():
        testData.loc[testData['instancename'] == instance, 'bkv'] = min(testData[testData['instancename'] == instance]['result'].min(), trainingData[trainingData['instancename'] == instance]['result'].min(), testData[testData['instancename'] == instance]['bkv'].min())

    # Remove zeros
    testData['result'] = testData['result'].map(lambda x: x if x != 0 else 0.000001)
    testData['bkv'] = testData['bkv'].map(lambda x: x if x != 0 else 0.000001)
    
    # Calculate values to plot according to result type
    if typeResult == 'rdev': testData['yaxis'] = abs(1 - (testData['result'] / testData['bkv']))
    elif typeResult == 'adev': testData['yaxis'] = abs(testData['result'] - testData['bkv'])
    else: testData['yaxis'] = testData['result']

    return testData, firstElites, finalElites, testConfigurations


# Function __readTraining
def __readTraining(iracelog, typeResult, bkvFile, overTime, imputation, logScale):
    """
    Reads the training data from the irace log file.
    Arguments:
        - iracelog: irace log file
        - typeResult: type of results to show
        - bkvFile: file with the best known values for each instance
        - overTime: enables the execution time in the x-axis
        - imputation: defines the imputation strategy
        - logScale: enables logscale
    """

    # Load the R object from the irace log file
    robjects.r['load'](iracelog)
    # Get the executions, training instances and instance names
    iraceExp = np.array(robjects.r('iraceResults$experiments'))
    iraceExpLog = np.array(robjects.r('iraceResults$experimentLog'))
    iraceInstances = np.array(robjects.r('iraceResults$state$.irace$instancesList'))[0]
    iraceInstanceNames = np.array(robjects.r('iraceResults$scenario$instances'))

    # Create a list with the elite configurations of each iteration
    elites = []
    for i in range(1, int(robjects.r('iraceResults$state$indexIteration')[0])):
        elites.append([int(item) for item in str(robjects.r('iraceResults$allElites[[' + str(i) + ']]')).replace('[1]', '').strip().split()])

    # Create a dictionary for all executions (experiments)
    experiments = []; id = 0; cumulativeTime = 0
    for i in range(len(iraceExpLog)):
        id += 1
        experiment = {}
        experiment['id'] = id
        experiment['iteration'] = max(int(iraceExpLog[i][0]), 1)
        experiment['instance'] = int(iraceExpLog[i][1])
        # Determine the time when the execution started (for overTime option)
        experiment['startTime'] = cumulativeTime
        cumulativeTime += float(iraceExpLog[i][3])
        experiment['configuration'] = int(iraceExpLog[i][2])
        experiment['value'] = iraceExp[experiment['instance'] - 1][experiment['configuration'] - 1]
        # Determine the type of the configuration/execution
        experiment['type'] = ('best' if experiment['configuration'] == elites[-1][0] else
                            'final' if experiment['configuration'] in elites[-1] else
                            'elite' if experiment['configuration'] in elites[experiment['iteration'] - 1] else
                            'regular')
        experiments.append(experiment)
    # Create data frame from the dictionary
    data = pd.DataFrame(experiments)
    # Determine the values for the x-axis
    data['xaxis'] = data['startTime'] if overTime and not math.isnan(cumulativeTime) else data['id']
    if overTime and math.isnan(cumulativeTime): print('  - You are trying to plot over time, but the irace log file does not have running time data; setting overtime to false!'); overTime = False
    # Determine <instance, seed> pairs, instance IDs and instance names
    data['instanceseed'] = data['instance']
    data['instance'] = data['instance'].map(lambda x: iraceInstances[x - 1])
    data['instancename'] = data['instance'].map(lambda x: iraceInstanceNames[x - 1][iraceInstanceNames[x - 1].rindex('/') + 1 if '/' in iraceInstanceNames[x - 1] else 0:iraceInstanceNames[x - 1].rindex('.') if '.' in iraceInstanceNames[x - 1] else len(iraceInstanceNames[x - 1])])

    # Calculate the best known values for each instance
    data['bkv'] = float('inf')
    # Read from file, if given
    if bkvFile is not None:
        bkv = pd.read_csv(bkvFile, sep = ':', header = None, names = ['instancename', 'bkv'])
        bkv['bkv'] = pd.to_numeric(bkv['bkv'], errors = 'raise')
        for instance in data['instancename'].unique().tolist():
            data.loc[data['instancename'] == instance, 'bkv'] = bkv[bkv['instancename'] == instance]['bkv'].min()

    # Also consider the values found during the configuration process
    for instance in data['instance'].unique().tolist():
        data.loc[data['instance'] == instance, 'bkv'] = min(data[data['instance'] == instance]['value'].min(), data[data['instance'] == instance]['bkv'].min())
    
    # Remove zeros
    if typeResult == 'rdev':
        data['value'] = data['value'].map(lambda x: x if x != 0 else 0.000001)
        data['bkv'] = data['bkv'].map(lambda x: x if x != 0 else 0.000001)

    # Calculate values to plot according to result type
    if typeResult == 'rdev': data['yaxis'] = abs(1 - (data['value'] / data['bkv']))
    elif typeResult == 'adev': data['yaxis'] = abs(data['value'] - data['bkv'])
    else: data['yaxis'] = data['value']

    # Avoid zeros
    if logScale: data.loc[data['yaxis'] == 0, 'yaxis'] = data[data['yaxis'] > 0]['yaxis'].min() / 2

    # Check whether iterations performed a restart
    restarts = [bool(item) for item in np.array(robjects.r('iraceResults$softRestart'))]
    if len(restarts) < len(data['iteration'].unique()): restarts.insert(0, False)
    
    # Determine the number of different instances used until each iteration
    instancesSoFar = []
    # First group by iteration and aggregate the <instance, seed> pairs
    usedInstances = data.groupby('iteration', as_index = False).agg({'instanceseed': 'unique'})
    # For each iteration, check the used instances and add to a list
    for iteration in usedInstances['iteration'].tolist():
        instancesSoFar.append([])
        for instanceList in usedInstances[usedInstances['iteration'] <= iteration]['instanceseed'].tolist():
            instancesSoFar[-1].extend(instanceList)
        # Remove duplicates
        instancesSoFar[-1] = np.unique(instancesSoFar[-1])
    # Create a list with amounts
    instancesSoFar = [len(item) for item in instancesSoFar]

    # Calculate the median performances for elite and non-elite configurations
    mediansEliteDict = {'iteration': [], 'median': []}
    mediansRegularDict = {'iteration': [], 'median': []}
    # Medians are calculated for each iteration
    iterations = data['iteration'].unique()
    for iteration in iterations:
        # Get the corresponding data
        dataIt = data[data['iteration'] <= iteration] # Consider only the executions until that iteration
        instancesIt = dataIt['instanceseed'].unique() # Consider only instances used until that iteration
        eliteConfs = dataIt[(dataIt['iteration'] == iteration) & (dataIt['type'] != 'regular')]['configuration'].unique() # Elite configurations
        nonEliteConfs = dataIt[(dataIt['iteration'] == iteration) & (dataIt['type'] == 'regular')]['configuration'].unique() # Non-elite configurations       
        execElite = dataIt[(dataIt['configuration'].isin(eliteConfs)) & (dataIt['instanceseed'].isin(instancesIt))] # Executions of elite configurations
        execElite = execElite.groupby(['configuration', 'instanceseed'], as_index = False).agg({'yaxis': 'median'}) # Group by configuration and <instance, seed>
        execNonElite = dataIt[(dataIt['configuration'].isin(nonEliteConfs)) & (dataIt['instanceseed'].isin(instancesIt))] # Executions of non-elite configurations
        execNonElite = execNonElite.groupby(['configuration', 'instanceseed'], as_index = False).agg({'yaxis': 'median'}) # Group by configuration and <instance, seed>
        
        # Create lists of pairs <configuration, instance>
        pairsElite = [(x, y) for x in eliteConfs for y in instancesIt]
        pairsNonElite = [(x, y) for x in nonEliteConfs for y in instancesIt]
        # Dictionaries to store the results
        resultsElite = {}
        resultsNonElite = {}
        # Get the results of each execution (elite and non-elite), add to dictionary, and remove the pair from the lists above
        for _, row in execElite.iterrows():
            element = (int(row['configuration']), int(row['instanceseed']))
            resultsElite[element] = row['yaxis']
            pairsElite.remove(element)
        for _, row in execNonElite.iterrows():
            element = (int(row['configuration']), int(row['instanceseed']))
            resultsNonElite[element] = row['yaxis']
            pairsNonElite.remove(element)

        # An imputation is performed for missing values
        remInst = [x[1] for x in pairsElite] + [x[1] for x in pairsNonElite] # Remaining instances
        remInst = list(set(remInst)) # Without duplicates
        imputations = {}
        for instance in remInst:
            # Calculate the imputation for each instance, according to the chosen strategy
            if imputation == 'elite': imputationValue = data[(data['configuration'].isin(eliteConfs)) & (data['instanceseed'] == instance) & (data['iteration'] <= iteration)]['yaxis'].max()
            elif imputation == 'alive': imputationValue = data[((data['configuration'].isin(eliteConfs)) | (data['configuration'].isin(nonEliteConfs))) & (data['instanceseed'] == instance) & (data['iteration'] <= iteration)]['yaxis'].max()            
            imputations[instance] = imputationValue
        # Use the calculated imputation value for each missing execution
        for element in pairsElite: # For elite configurations
            resultsElite[element] = imputations[element[1]]
        for element in pairsNonElite: # And for non-elite configurations
            resultsNonElite[element] = imputations[element[1]]
        
        # Calculate medians
        for r in ['elite', 'nonElite']:
            # Get the corresponding data
            results = resultsElite if r == 'elite' else resultsNonElite
            # First aggregate over different replications of configurations
            confResults = {}
            for element in results:
                # If first occurence, add new element
                if element[0] not in confResults:
                    confResults[element[0]] = []
                # Append result
                confResults[element[0]].append(results[element])
            # Aggregate (median)
            for element in confResults:
                confResults[element] = median(confResults[element])
            # Update data structure
            if r == 'elite': resultsElite = confResults
            else: resultsNonElite = confResults
        
        # Finally, aggregate over configurations
        mediansEliteDict['iteration'].append(iteration)        
        mediansEliteDict['median'].append(median(resultsElite.values()))
        mediansRegularDict['iteration'].append(iteration)
        mediansRegularDict['median'].append(median(resultsNonElite.values()))
    # Create data frame from dictionaries
    mediansElite = pd.DataFrame.from_dict(mediansEliteDict)
    mediansRegular = pd.DataFrame.from_dict(mediansRegularDict)

    return data, restarts, instancesSoFar, overTime, mediansRegular, mediansElite


# Function monitor
def __monitor(iracelog, typeResult = 'rdev', showElites = True, showInstances = True, pconfig = 0, showPlot = True, exportData = False, exportPlot = False, output = 'output', bkv = None, overTime = False, userPlt = None, showToolTips = True, imputation = 'elite', testing = False, testColors = 'instance', alpha = 1.0, reverse = False, logScale = True, timeLimit = 0):
    """
    Monitors the irace log file and generates a plot after each iteration;
    plots are exported to a PDF file (name defined by argument 'output');
    the arguments is the same as those of the plot function, but come of them
    are ignored (e.g. exportData, exportPlot and testing options).
    
    Arguments:
        - iracelog: irace log file
        - typeResult: type of results to show
        - showElites: indicates if executions of elite configurations must be highlighted
        - showInstances: indicates if executions on different instances must be highlighted
        - pconfig: percentage of the executions whose configuration is presented
        - showPlot: defines if the plot must be shown
        - exportData: defines if the data must be exported
        - exportPlot: defines if the plot must be exported
        - output: the name of the exported file
        - bkv: file with the best known values for each instance
        - overTime: enables the execution time in the x-axis
        - userPlt: a plt object that can be given by an external program
        - showToolTips: enables the tool tip boxes with information about the executions
        - imputation: defines the imputation stratedy
        - testing: defines if the test plot must be shown
        - testColors: color scheme
        - alpha: opacity of the points
        - reverse: show y-axis reversed
        - logScale: enables logscale
        - timeLimit: defines the time limit when plotting running times
    """
    
    # Definition of global variables for this function 
    global plt

    # Variables to control the monitoring process and store the plot images
    size = -1
    iteration = 0
    images = []

    # Delete the output file, if it already exists
    if os.path.exists('./monitor/' + output + '.pdf'): os.remove('./monitor/' + output + '.pdf')
    if not os.path.exists('./monitor/'): os.mkdir('./monitor')

    # Loop for monitoring the irace log file
    while True:
        # Current size of the log file
        newSize = os.stat(iracelog).st_size
        # Check wether we have modifications (different size)
        if newSize != size:
            size = newSize
            # Compute current iteration
            robjects.r['load'](iracelog)
            newIteration = int(np.array(robjects.r('iraceResults$state$indexIteration'))[0])
            # Check wether a new iteration begins
            if newIteration != iteration:
                iteration = newIteration
                # Generate plot, wether we have at least one finished iteration
                if iteration > 1:
                    # Read data and generate training plot
                    data, restarts, instancesSoFar, overTime, mediansRegular, mediansElite = __readTraining(iracelog, typeResult, bkv, overTime, imputation, logScale)
                    __plotTraining(data, typeResult, restarts, showElites, showInstances, pconfig, overTime, showToolTips, instancesSoFar, mediansElite, mediansRegular, alpha, reverse, logScale, timeLimit)
                    plt.title('ITERATIONS 1 TO ' + str(iteration - 1) if (iteration - 1) != 1 else 'ITERATION ' + str(iteration - 1))
                    fig = plt.gcf()
                    fig.subplots_adjust(top = 0.90)
                    # Save plot in a buffer
                    buf = BytesIO()
                    fig.savefig(buf, format = 'png', dpi = 300)
                    # Read image to store in a PIL image object
                    buf.seek(0)
                    img = Image.open(buf).convert("RGB")
                    # Append to the list of images
                    images.append(img)
                    # Update the PDF with all images
                    images[0].save('./monitor/' + output + '.pdf', save_all = True, append_images = images[1:], quality = 100, optimize = True)
                    # Clean plot object for the next plot
                    plt.clf()
                    # Log progress
                    print('> Update output (./monitor/' + output + '.pdf), including visualization of iteration ' + str(iteration - 1) + '.')
        # Wait 1s until the next iteration
        time.sleep(1)


# Function getPlot
def getPlot(iracelog, typeResult = 'rdev', showElites = True, showInstances = True, pconfig = 0, showPlot = True, exportData = False, exportPlot = False, output = 'output', bkv = None, overTime = False, userPlt = None, showToolTips = True, imputation = 'elite', testing = False, testColors = 'instance', alpha = 1.0, reverse = False, logScale = True, timeLimit = 0):
    """
    Creates a plot object, calls the corresponding functions, export
    the plot (if desired), show the plot or return the plot object.
    
    This is the only public function. It can be called from an external
    program, to communicate with acviz.

    Arguments:
        - iracelog: irace log file
        - typeResult: type of results to show
        - showElites: indicates if executions of elite configurations must be highlighted
        - showInstances: indicates if executions on different instances must be highlighted
        - pconfig: percentage of the executions whose configuration is presented
        - showPlot: defines if the plot must be shown
        - exportData: defines if the data must be exported
        - exportPlot: defines if the plot must be exported
        - output: the name of the exported file
        - bkv: file with the best known values for each instance
        - overTime: enables the execution time in the x-axis
        - userPlt: a plt object that can be given by an external program
        - showToolTips: enables the tool tip boxes with information about the executions
        - imputation: defines the imputation stratedy
        - testing: defines if the test plot must be shown
        - testColors: color scheme
        - alpha: opacity of the points
        - reverse: show y-axis reversed
        - logScale: enables logscale
        - timeLimit: defines the time limit when plotting running times
    """

    # Definition of global variables for this function 
    global plt

    # Use the plt object, if given
    if userPlt is not None: plt = userPlt
    if testing: # Test plot
        testData, firstElites, finalElites, testConfigurations = __readTest(iracelog, typeResult, bkv)
        __plotTest(testData, typeResult, firstElites, finalElites, testConfigurations, testColors)
    else: # Training plot
        data, restarts, instancesSoFar, overTime, mediansRegular, mediansElite = __readTraining(iracelog, typeResult, bkv, overTime, imputation, logScale)
        __plotTraining(data, typeResult, restarts, showElites, showInstances, pconfig, overTime, showToolTips, instancesSoFar, mediansElite, mediansRegular, alpha, reverse, logScale, timeLimit)
    if exportData: # Export data
        if not testing:
            if not os.path.exists('./export'): os.mkdir('./export')
            file = open('./export/' + output + '.csv', 'w')
            file.write(data.to_csv())
            file.close()
            print('> data exported to export/' + output + '.csv')
        else:
            print('> cat only exports training data (remove --testing option)')
    if exportPlot: # Export plot
        if not os.path.exists('./export'): os.mkdir('./export')
        plt.savefig('./export/' + output + '.pdf', format = 'pdf')
        plt.savefig('./export/' + output + '.png', format = 'png')
        print('> Plot exported to export/' + output + '.pdf')
        print('> Plot exported to export/' + output + '.png')
    else: # Show plot
        if showPlot: plt.show()
        else: return plt


# Function main
if __name__ == "__main__":
    # Definition of arguments
    parser = argparse.ArgumentParser()
    parser._action_groups.pop()
    required = parser.add_argument_group('required arguments')
    optional = parser.add_argument_group('optional arguments')
    required.add_argument('--iracelog', help = 'input of irace log file (.Rdata)', metavar = '<file>', required = False)
    optional.add_argument('-v', '--version', help = 'show description and exit', action = 'store_true')
    optional.add_argument('--typeresult', help = 'defines how the results should be presented in training or test plot [aval, adev, rdev] (default: rdev)', default = 'rdev', metavar = '<res>', type = str)
    optional.add_argument('--bkv', help = 'file containing best known values for the instances used (null by default)', metavar = '<file>')
    optional.add_argument('--imputation', help = 'imputation strategy for computing medians [elite, alive] (default: elite)', metavar = '<imp>', type = str, default = 'elite')
    optional.add_argument('--scale', help = 'defines the strategy for the scale of y-axis of the training plot [log, lin] (default: log)', metavar = '<s>', type = str, default = 'log')
    optional.add_argument('--noelites', help = 'disables identification of elite configurations (disabled by default)', action = 'store_false')
    optional.add_argument('--noinstances', help = 'disables identification of instances (disabled by default)', action = 'store_false')
    optional.add_argument('--pconfig', help = 'show configurations of the p%% best executions [0, 100] (default: 0)', metavar = '<p>', default = 0, type = int)
    optional.add_argument('--overtime', help = 'plot the execution over the accumulated configuration time (disabled by default)', action = 'store_true')
    optional.add_argument('--alpha', help = 'opacity of the points, the greater the more opaque [0, 1] (default: 1)', metavar = '<alpha>', type = float, default = 1.0)
    optional.add_argument('--timelimit', help = 'when plotting running times (absolute values and linear scale), executions with value greater than or equal to <tl> will be considered as not solved (NS) and presented accordingly (default: 0 [disabled])', metavar = '<tl>', default = 0, type = int)
    optional.add_argument('--testing', help = 'plots the testing data instead of the configuration process (disabled by default)', action = 'store_true')
    optional.add_argument('--testcolors', help = 'option for how apply the colormap in the test plot [overall, instance] (default: instance)', default = 'instance', metavar = '<col>', type = str)
    optional.add_argument('--exportdata', help = 'exports the used data to a csv format file (disabled by default)', action = 'store_true')
    optional.add_argument('--exportplot', help = 'exports the resulting plot to png and pdf files (disabled by default)', action = 'store_true')
    optional.add_argument('--output', help = 'defines a name for the output files (default: export)', metavar = '<name>', type = str, default = 'export')
    optional.add_argument('--monitor', help = 'monitors the irace log file during irace execution; produces one plot for each iteration (disabled by default)', action = 'store_true')
    optional.add_argument('--reverse', help = 'reverses y-axis (disabled by default)', action = 'store_true')
    args, other = parser.parse_known_args()
    
    # Print version
    if args.version: print(desc); exit()
    
    # If iracelog not given, try to use the first value given, otherwise return error
    if not args.iracelog:
        if len(other) > 0: args.iracelog = other[0]
        else: print('Invalid arguments!\nPlease input the irace log file using \'--iracelog <file>\'\n'); parser.print_help(); exit()
    
    # Print information about acviz
    print(desc)
    # Print settings of this execution
    settings = '> Settings:\n'
    settings += '  - plot evolution of the configuration process\n'
    if args.monitor: settings += '  - executing in monitor mode\n'
    settings += '  - irace log file: ' + args.iracelog + '\n'
    settings += '  - type of results: ' + args.typeresult + '\n'
    settings += '  - imputation strategy: ' + args.imputation + '\n'
    if args.bkv is not None: settings += '  - bkv file: ' + str(args.bkv) + '\n'
    if (not args.testing) and args.scale == 'log': settings += '  - plotting in logscale\n'
    if args.noelites: settings += '  - show elite configurations\n'
    if args.noinstances: settings += '  - identify instances\n'
    if args.pconfig > 0: settings += '  - showing the best configurations (pconfig = %d)\n' % args.pconfig
    if args.timelimit > 0: settings += '  - using time limit: %d\n' % args.timelimit
    if args.overtime: settings += '  - plotting over time\n'
    if args.reverse: settings += '  - plotting reversed y-axis\n'
    if args.testing: settings += '  - plotting test data\n'
    if args.testing: settings += '  - using a %s-based colormap\n' % args.testcolors
    if args.exportdata: settings += '  - export data to csv\n'
    if args.exportplot: settings += '  - export plot to pdf and png\n'
    if args.exportdata or args.exportplot: settings += '  - output file name: %s\n' % args.output

    print(settings)
    
    # Verify which function has to be called
    call = __monitor if args.monitor else getPlot
    # Call function
    call(
        iracelog = args.iracelog,
        typeResult = args.typeresult,
        showElites = args.noelites,
        showInstances = args.noinstances,
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
        testing = args.testing,
        testColors = args.testcolors,
        alpha = args.alpha,
        reverse = args.reverse,
        logScale = (args.scale == 'log'),
        timeLimit = max(args.timelimit, 0)
    )
    print('-------------------------------------------------------------------------------')
