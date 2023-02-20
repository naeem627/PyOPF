# PyOPF

A tool for solving AC Optimal Power Flow (OPF) in Python using Pyomo optimization modeling. 

---

## Description

ACOPF solver that uses a current-voltage formulation. This project is still in development. The only supported cases at 
the moment are the IEEE-14, IEEE-118, and Texas7k. No other cases have been tested. 

Uses [C2DataUtilities](https://github.com/jesseholzerpnnl/C2DataUtilities/) to parse grid data from a RAW file.

## Requirements

This package requires [IPOPT](https://github.com/coin-or/Ipopt). You must install this yourself. A future version of 
this project will add support to verify IPOPT is installed.

A conda environment.yml file is included for this project that you can use to install other project requirements by
running the following:

```
conda env create -f environment.yml
```

You can update the environment with:
```
conda env export | cut -f -2 -d "=" | grep -v "prefix" > environment.yml
```

## Tests
Use the following command to run tests before committing any new code. Always run tests.
```
python -m pytest
```

## Build Instructions

Local package installation support coming soon

##  Run

To run a case such as the IEEE-14 do:
```
python -m pyopf --case IEEE-14 --obj "min cost"
```