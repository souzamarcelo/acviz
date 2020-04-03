# cat: Configuration Analysis Tools (CAT) for irace

This Python script provides a set of functions to analyze the evolution of the algorithm configuration process with [irace](http://iridia.ulb.ac.be/irace).

## Dependencies

The script requires [Python 3.x](https://www.python.org) and the following Python modules:

+ [numpy](https://numpy.org)
+ [pandas](https://pandas.pydata.org)
+ [matplotlib](https://matplotlib.org)
+ [rpy2](https://rpy2.github.io)

Since irace exports the logfile using the R syntax and format, you will need the [R software environment](https://www.r-project.org) installed (rpy2 module will communicate with R to get the necessary data for the analyses).

## Basic usage

To use **cat** you need to download the `cat.py` script and run it according to the following instructions (make sure that the aforementioned dependencies were all satisfied).

**Input:** an irace logfile (typically called irace.Rdata).

**Output:** a matplotlib graphic showing the candidate evaluations.

```
usage: cat.py [-h] [-v] [--iracelog <file>]

optional arguments:
  -h, --help         show this help message and exit
  -v, --version      show description and exit
  --iracelog <file>  input of irace log file (.Rdata)
```

### Example

The `example` directory has an irace logfile example, which contains the log data of the configuration of the ACOTSP algorithm. To analyze it, you can call **cat** from the command line as follows:

```
python3 cat.py --iracelog examples/acotsp.Rdata
```