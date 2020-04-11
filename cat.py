#!/usr/bin/python3
import sys
import argparse
from rpy2 import robjects
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from math import log

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

def plot(data, restarts):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlim((1, data['id'].max()))
    
    plt.title('Evolution of the configuration process')
    plt.xlabel('candidate evaluations [from %d to %d]' % (1, data['id'].max()))
    plt.ylabel('solution quality [relative deviation]')
    
    data.loc[data['reldev'] == 0, 'reldev'] = data[data['reldev'] > 0]['reldev'].min() / 2
    regular = plt.scatter(data[data['type'] == 'regular']['id'], data[data['type'] == 'regular']['reldev'].map(log), alpha = 0.7, c = 'black', marker = "x", linewidth = 0.3, s = 13)
    elite = plt.scatter(data[data['type'] == 'elite']['id'], data[data['type'] == 'elite']['reldev'].map(log), alpha = 0.7, c = '#0000FF', edgecolors = 'black', marker = 'o', linewidth = 0.3, s = 25)
    final = plt.scatter(data[data['type'] == 'final']['id'], data[data['type'] == 'final']['reldev'].map(log), alpha = 0.7, c = '#FF0000', edgecolors = 'black', marker = '^', linewidth = 0.3, s = 27)
    best = plt.scatter(data[data['type'] == 'best']['id'], data[data['type'] == 'best']['reldev'].map(log), alpha = 0.8, c = '#006600', edgecolors = '#006600', marker = '*', linewidth = 0.5, s = 32)
    fig.legend([regular, elite, final, best], ['regular execution', 'elite configuration', 'final elite configuration', 'best found configuration'], loc = 'center', bbox_to_anchor = (0.5, 0.03), ncol = 5, handletextpad = -0.3, columnspacing = 0.5)

    iterationPoints = data.groupby('iteration', as_index = False).agg({'id': 'first'})['id'].tolist()
    for point, restart in zip(iterationPoints, restarts):
        color = 'r' if restart else 'k'
        plt.axvline(x = point, color = color, linestyle = ':', linewidth = 1.5)
    ax.set_xticks(iterationPoints)

    iterations = data['iteration'].unique().tolist()
    avg = [data[data['iteration'] == iteration]['reldev'].mean() for iteration in iterations]
    best = [data[data['iteration'] == iteration].groupby('instance', as_index = False).agg({'reldev': 'min'})['reldev'].mean() for iteration in iterations]
    iterationPoints.append(data['id'].max())
    for i in range(len(iterations)):
        plt.plot([iterationPoints[i], iterationPoints[i + 1]], [log(avg[i]), log(avg[i])], linestyle = '-', color = '#FF8C00', linewidth = 1.7)
        plt.plot([iterationPoints[i], iterationPoints[i + 1]], [log(best[i]), log(best[i])], linestyle = '-', color = '#800080', linewidth = 1.7)
    
    ax.tick_params(axis = 'both', which = 'major', labelsize = 9)
    plt.xticks(rotation = 90)

    fig.set_size_inches(10, 6)
    fig.subplots_adjust(top = 0.95)
    fig.subplots_adjust(bottom = 0.2)
    fig.subplots_adjust(right = 0.99)
    fig.subplots_adjust(left = 0.06)

    plt.show()


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


def main(iracelog):
    data, restarts = read(iracelog)
    plot(data, restarts)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--version', help = 'show description and exit', action = 'store_true')
    parser.add_argument('--iracelog', help = 'input of irace log file (.Rdata)', metavar = '<file>')
    args = parser.parse_args()
    if args.version: print(desc); exit()
    if not args.iracelog: print('Invalid arguments!\n'); parser.print_help(); exit()
    main(args.iracelog)