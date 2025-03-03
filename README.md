
<p align="center">
  <img src="https://github.com/user-attachments/assets/1e156ecb-75d5-4004-b768-cbd8edab7940" width="300">
</p>

<!-- ![sb_logo](https://github.com/user-attachments/assets/6c5d9280-70b2-4f66-8bfb-c513317aea06) -->

# Spaceborne

# Installation

We recommend using Spaceborne in a dedicated Conda environment. This ensures all dependencies are properly managed.

## 1. Create the environment and install dependencies

In the root folder of the Spaceborne repository, run

```bash
$ conda env create -f environment.yaml
$ conda activate spaceborne
```

`Spaceborne` uses `CCL` as the backend library for many cosmological calculations. Some installation issues have been found with its Python wrapper `pyccl`; in case of problems with this package, please refer to the official [instructions](https://github.com/LSSTDESC/CCL). To facilitate the process, however, its main dependencies - `Swig` and `CMake` - have already been included in the environment, so *after* creating and activating it, a simple 

```bash
$ conda install -c conda-forge pyccl
```

or

```bash
$ pip install pyccl
```
should do the job.

If the problem persists, you can try with

```bash
$ sudo apt-get install gfortran cmake build-essential autoconf bison
```

on Linux and 

```bash
$ brew install gfortran cmake build-essential autoconf bison
```

on OSX. The installation on Windows machines is not supported.

---

## 2. Install Spaceborne

### Option A: Using `pip`

To install Spaceborne directly:

```bash
$ pip install .
```

### Option B: Using Poetry

[Poetry](https://python-poetry.org/) is an alternative package manager. To use it:

1. Install Poetry:
   ```bash
   $ curl -sSL https://install.python-poetry.org | python3 -
   ```
2. Install Spaceborne:
   ```bash
   $ poetry install
   ```

---

## 3. Install Julia for Computational Efficiency

Spaceborne leverages `julia` for computationally intensive tasks. We recommend installing `julia` via [`juliaup`](https://github.com/JuliaLang/juliaup):

```bash
$ curl -fsSL https://install.julialang.org | sh  # Install juliaup
$ juliaup default 1.10                           # Install Julia version 1.10
```

Then, install the required Julia packages:

```bash
$ julia -e 'using Pkg; Pkg.add("LoopVectorization"); Pkg.add("YAML"); Pkg.add("NPZ")'
```

---

# Running the Code

All the available options and configurations can be found, along with their explanation, in the `config.yaml` file. To run `Spaceborne` *with the configuration specified in the* `Spaceborne/config.yaml` *file*, simply execute the following command:

```bash
$ python main.py
```

If you want to use a configuration file with a different name and/or location, you can instead run with

```bash
$ python main.py --config=<path_to_config_file>
```

for example:

```bash
$ python main.py --config="path/to/my/config/config.yaml"
```

To display the plots generated by the code, add the `--show_plots` flag:

```bash
$ python main.py --config="path/to/my/config/config.yaml" --show_plots
```


