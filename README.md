# cat: Configuration Analysis Tools (CAT) for irace

This Python script provides a set of functions to analyze the evolution of the algorithm configuration process with [irace](http://iridia.ulb.ac.be/irace).

**Maintainer:** [Marcelo de Souza](https://souzamarcelo.github.io).

**Contributors:** [Marcelo de Souza](https://souzamarcelo.github.io) and [Marcus Ritt](https://www.inf.ufrgs.br/~mrpritt).

If you have any difficult or want to collaborate with us, please write to me: marcelo.desouza@udesc.br.

***

## Dependencies

The script requires [Python 3.x](https://www.python.org) and the following Python modules (you can just install [anaconda](https://www.anaconda.com) to get Python with all modules included):

+ [numpy](https://numpy.org)
+ [pandas](https://pandas.pydata.org)
+ [matplotlib](https://matplotlib.org)
+ [rpy2](https://rpy2.github.io)

Since irace exports the log file using the R syntax and format, you will need the [R software environment](https://www.r-project.org) installed (rpy2 module will communicate with R to get the necessary data).

## Usage

To use **cat** you need to download the `cat.py` script and run it according to the following instructions (make sure that the aforementioned dependencies were all satisfied).

**Input:** an irace log file (typically called irace.Rdata) and optional parameters to control the plot details and the output format.

**Output:** a matplotlib graphic showing the candidate evaluations.

```
usage: cat.py [-h] --iracelog <file> [-v] [--elites] [--configurations]
              [--pconfig <p>] [--instances] [--exportdata] [--exportplot]
              [--output <name>]

required arguments:
  --iracelog <file>  input of irace log file (.Rdata)

optional arguments:
  -v, --version      show description and exit
  --elites           enables identification of elite configurations (disabled
                     by default)
  --configurations   enables identification of configurations (disabled by
                     default)
  --pconfig <p>      when --configurations, show configurations of the p% best
                     executions [0, 100] (default: 10)
  --instances        enables identification of instances (disabled by default)
  --exportdata       exports the used data to a csv format file (disabled by
                     default)
  --exportplot       exports the resulting plot to png and pdf files (disabled
                     by default)
  --output <name>    defines a name for the output files
```

### Optional arguments

+ `--elites`: enables the highlighting of the executions of elite configurations for each iteration, executions of final elite configurations, and executions of the best found configuration.
+ `--configurations`: enables the highlighting of configurations of the p% best performing executions of each iteration. Parameter p is given using the `--pconfig` argument (the defaulu value is 10).
+ `--instances`: enables the presentation of different instances used during the configuration process using colors.
+ `--exportdata`: exports the data used in csv format. The output will be saved in `./export/<name>.csv`, where `<name>` is defined using the `--output` argument.
+ `--exportplot`: exports the resulting plot in pdf and png formats. The output will be saved in `./export/<name>.pdf` and `./export/<name>.png`, where `<name>` is defined using the `--output` argument.
  + If no value for `--output` is provided, cat will use the name of the irace log file.

### Examples

The [examples](examples) directory has an irace log file example, which contains the log data of the ACOTSP algorithm configuration. To analyze it, you can call **cat** from the command line as follows:

```
python3 cat.py --iracelog examples/acotsp.Rdata --elites
```

In this case, **cat** will present the corresponding plot with each execution performed in the configuration process and the obtained relative deviation from the best found solution (logscale). Elite, final elite, and the best found configurations are presented using different markers and colors (since `--elites` is enabled), according to the provided legend. For each iteration, the plot presents the median performances (overall and of the elite candidates).

![](./examples/acotsp1.png)

We can enable the identification of instances by adding the `--instances` option. we can also enable the identification of configurations by adding the `--configurations` option. By setting `--pconfig 5` (see command below), it will produce a plot with the 5% best executions of each iteration identified with the corresponding configuration.

```
python3 cat.py --iracelog examples/acotsp.Rdata --elites --instances --configurations --pconfig 5
```

The output plot is:

![](./examples/acotsp2.png)
