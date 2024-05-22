
# My Paper Title

This repository contains the official implementation of [Competitive Algorithms for Online Knapsack with Succinct Predictions].

> 📋 *Optional: Include a graphic explaining your approach/main result, BibTeX entry, link to demos, blog posts, and tutorials.*

## Requirements

- Any C++ compiler
- Any Python compiler
- An application to unzip the dataset

## Pre-Evaluation

You need to run `bit-2017-partition.py` and `bit-2018-partition.py` in the `Fig4` folder or have all `con_numbers_i.txt` files available.

## Evaluation

Each folder contains code that can be run to generate the figures described in the paper's main body or appendix. You can modify parameters in the `inith()` function, as shown below:

```train
def inith():
    alg_list1 = [
        Algorithms("PP-a", ppa_a, [], "simple"),
        Algorithms("CONV-PP-A", conv_ppa_a, [], "simple2"),
        Algorithms("SENTINEL", im, [], "input"),
        Algorithms("ZCL", noadvice, [], "simple"),
    ]
    info_list = [
        Setting(
            L=1,
            U=100,
            delta=0,
            n=10000,
            capacity=1,
            num_runs=10,
            name="δ=0",
            alg_list=copy.deepcopy(alg_list1),
        ),
        Setting(
            L=1,
            U=100,
            delta=0.5,
            n=10000,
            capacity=1,
            num_runs=10,
            name="δ=0.5",
            alg_list=copy.deepcopy(alg_list1),
        ),
        Setting(
            L=1,
            U=100,
            delta=1,
            n=10000,
            capacity=1,
            num_runs=10,
            name="δ=1",
            alg_list=copy.deepcopy(alg_list1),
        ),
        Setting(
            L=1,
            U=100,
            delta=1.5,
            n=10000,
            capacity=1,
            num_runs=10,
            name="δ=1.5",
            alg_list=copy.deepcopy(alg_list1),
        ),
        Setting(
            L=1,
            U=100,
            delta=2,
            n=10000,
            capacity=1,
            num_runs=10,
            name="δ=2",
            alg_list=copy.deepcopy(alg_list1),
        ),
    ]
    return info_list
```
This function specifies running four algorithms in five different settings. Each setting is indicated by the lowest value, highest value, delta factor, number of items, capacity, number of runs, and a name.

To change the input, you can use different initialization functions specified for each figure.

The latest version of the code is in `Fig4/code`.

### Running on Bitcoin Data

1. Run `bit-year-partition` on zipped files.
2. Run `ravi.cpp`, which implements the Sentinel algorithm.
3. Run the code.

### For Figure 3

1. Run `ravi.cpp`.
2. Run `code.py` to get results on Sentinel synthetic data.

## Results

Our algorithm achieves a high competitive ratio on different synthetic datasets as well as the Bitcoin dataset.
