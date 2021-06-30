# acviz: Algorithm Configuration Visualizations for irace

This Python program provides visualizations of the automatic algorithm configuration process with [irace](http://iridia.ulb.ac.be/irace). The following article describes *acviz* in detail and presents a comprehensive experimental evaluation. You can also check the [supplementary material](https://doi.org/10.5281/zenodo.4028904) for further experimental details.

+ + Marcelo De Souza, Marcus Ritt, Manuel López-Ibáñez, and Leslie Pérez Cáceres. **ACVIZ: A Tool for the Visual Analysis of the Configuration of Algorithms with irace**. Operations Research Perspectives, 8:100-186, 2021.<br>
[[link here](https://www.sciencedirect.com/science/article/pii/S2214716021000099) | [supplementary material at Zenodo](https://doi.org/10.5281/zenodo.4028904)]

#### Bibtex
```bibtex
@article{DeSouzaEtAl2021acviz,
   title      = {ACVIZ: A Tool for the Visual Analysis of the Configuration of Algorithms with irace},
   author     = {Marcelo de Souza and Marcus Ritt and Manuel L{\'o}pez-Ib{\'a}{\~n}ez and Leslie P{\'e}rez C{\'a}ceres},
   journal    = {Operations Research Perspectives},
   year       = 2021,
   volume     = 8,
   pages      = 100--186,
   doi        = {10.1016/j.orp.2021.100186},
   supplement = {https://zenodo.org/record/4714582}
}
```

Please, make sure to reference us if you use *acviz* in your research.

***

## People

**Maintainer:** [Marcelo de Souza](https://souzamarcelo.github.io).

**Contributors:** [Marcus Ritt](https://www.inf.ufrgs.br/~mrpritt), [Manuel López-Ibáñez](http://lopez-ibanez.eu) and [Leslie Pérez Cáceres](https://sites.google.com/site/leslieperez).

**Contact:** marcelo.desouza@udesc.br

***

## Dependencies

The script requires [Python 3](https://www.python.org) and the following Python libraries:
+ [numpy](https://numpy.org)
+ [pandas](https://pandas.pydata.org)
+ [matplotlib](https://matplotlib.org)
+ [rpy2](https://rpy2.github.io)
+ [natsort](https://pypi.org/project/natsort)

Since irace exports the log file using the R syntax and format, you will need the [R software environment](https://www.r-project.org) installed (rpy2 module will communicate with R to get the necessary data).

***

## Usage

To use *acviz* you need to download the `acviz.py` script and run it according to the following instructions (make sure that the aforementioned dependencies were all satisfied). It is possible to control several elements of the visualization, including:
+ change the type of result to be presented;
+ change the imputation strategy;
+ change the scaling strategy;
+ disable the identification of instances and executions of elite configurations; 
+ show the configurations associated with the best executions;
+ plot executions over configuration time;
+ control the opacity of the points;
+ control the values and colors of the test plot;
+ export the produced plot;
+ monitor the irace log file during the execution of irace.

**Input:** an irace log file (typically called irace.Rdata) and optional parameters to control the plot details and the output format.

**Output:** a matplotlib plot.

```
usage: acviz.py [-h] [--iracelog <file>] [-v] [--typeresult <res>] [--bkv <file>]
                [--imputation <imp>] [--scale <s>] [--noelites] [--noinstances] [--pconfig <p>]
                [--overtime] [--alpha <alpha>] [--timelimit <tl>] [--testing] [--testcolors <col>]
                [--exportdata] [--exportplot] [--output <name>] [--monitor] [--reverse]

required arguments:
  --iracelog <file>   input of irace log file (.Rdata)

optional arguments:
  -v, --version       show description and exit
  --typeresult <res>  defines how the results should be presented in training or test plot [aval,
                      adev, rdev] (default: rdev)
  --bkv <file>        file containing best known values for the instances used (null by default)
  --imputation <imp>  imputation strategy for computing medians [elite, alive] (default: elite)
  --scale <s>         defines the strategy for the scale of y-axis of the training plot [log, lin]
                      (default: log)
  --noelites          disables identification of elite configurations (disabled by default)
  --noinstances       disables identification of instances (disabled by default)
  --pconfig <p>       show configurations of the p% best executions [0, 100] (default: 0)
  --overtime          plot the execution over the accumulated configuration time (disabled by
                      default)
  --alpha <alpha>     opacity of the points, the greater the more opaque [0, 1] (default: 1)
  --timelimit <tl>    when plotting running times (absolute values and linear scale), executions with
                      value greater than or equal to <tl> will be considered as not solved (NS) and
                      presented accordingly (default: 0 [disabled])
  --testing           plots the testing data instead of the configuration process (disabled by
                      default)
  --testcolors <col>  option for how apply the colormap in the test plot [overall, instance]
                      (default: instance)
  --exportdata        exports the used data to a csv format file (disabled by default)
  --exportplot        exports the resulting plot to png and pdf files (disabled by default)
  --output <name>     defines a name for the output files (default: export)
  --monitor           monitors the irace log file during irace execution; produces one plot for each
                      iteration (disabled by default)
  --reverse           reverses y-axis (disabled by default)
```

***

## Examples

The [examples](examples) directory contains some exemplary irace log files. To analyze the evolution of the configuration process, you can call *acviz* from the command line as follows:

```
python3 acviz.py --iracelog examples/acotsp-instances.Rdata --bkv examples/bkv.txt
```

In this case, *acviz* will present the plot with each execution performed in the configuration process and the corresponding relative deviations from the best known values (in log scale). The log file is provided using option `--iracelog` and the file containing the best known values of each instance is provided using option `--bkv`.

![](./examples/acotsp-instances.png)

In a second example, we visualize the quality of the best found configurations on the test instances (assuming that the testing options were enabled when running irace). We include the `--testing` option in the command as follows:

```
python3 acviz.py --iracelog examples/acotsp-overtuning.Rdata --bkv examples/bkv.txt --testing
```

In this case, *acviz* will present a visualization of the results of evaluating the best found configurations (best elite of each iteration and all elites of the last iteration) on all test instances.

![](./examples/acotsp-overtuning.png)
