<p align="center">
  <a href="https://travis-ci.org/neurolib-dev/neurolib">
  	<img alt="Build" src="https://travis-ci.org/caglorithm/mopet.svg?branch=master"></a>
  
  <a href="https://www.python.org/downloads/release">
  	<img alt="Python 3.6 + 3.7" src="https://img.shields.io/badge/python-3.6 + 3.7-blue.svg"></a>
    
  <a href="https://github.com/caglorithm/mopet/releases">
  	<img alt="Release" src="https://img.shields.io/github/v/release/caglorithm/mopet"></a>
  
  <a href="https://pypi.org/project/mopet/">
  	<img alt="PyPI" src="https://img.shields.io/pypi/v/mopet"></a>
  
  <a href="https://codecov.io/gh/caglorithm/mopet">
  	<img alt="codecov" src="https://codecov.io/gh/caglorithm/mopet/branch/master/graph/badge.svg"></a>
  
  <a href="https://pepy.tech/project/mopet">
  	<img alt="Downloads" src="https://pepy.tech/badge/mopet"></a>
  
  <a href="https://github.com/psf/black">
  	<img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg"></a>
  
</p>


# mopet 🛵
*The mildly ominous parameter exploration toolkit*

Isn't it strange that, although parameter explorations are a crucial part of computational modeling, there are almost no Python tools available for making your life easier? 
`mopet` is here to help! You can run extensive grid searches in parallel (powered by `ray`) and store extremely huge amounts of data into a HDF file (powered by `pytables`) for later analysis - or whatever your excuse is for buying a yet another hard disk. 

# Installation 💻
The easiest way to get going is to install the pypi package using `pip`:

```
pip install mopet
```
Alternatively, you can also clone this repository and install all dependencies with

```
git clone https://github.com/caglorithm/mopet.git
cd mopet/
pip install -r requirements.txt
pip install .
```

# Example usage 🐝
Setting up an exploration is as easy as can be!

```python
# first we define an toy evaluation function
def distance_from_circle(params):
	# let's simply calculate the distance of 
	# the x-y parameters to the unit circle
    distance = abs((params["x"] ** 2 + params["y"] ** 2) - 1)
    
    # we package the result in a dictionary
    result = {"result" : distance}
    return result

``` 

Let's set up the exploration by defining the parameters to explore and passing the evaluation function from above:

```python
import numpy as np
import mopet

explore_params = {"x": np.linspace(-2, 2, 21), "y": np.linspace(-2, 2, 21)}
ex = mopet.Exploration(distance_from_circle, explore_params)
```

Running the exploration is in parallel and is handled by `ray`. You can also use a private cluster or cloud infrastructure, see [here](https://ray.readthedocs.io/en/latest/autoscaling.html) for more info.

```python
ex.run()
>> 100%|██████████| 441/441 [426.57it/s]
ex.load_results()
```

An overview of the runs and runs is given as a `pandas` DataFrame, available as `ex.df`. Here we load the result, which is simply a `float`, directly into the DataFrame. However, if the result was a timeseries (a `numpy.ndarray`), we could process it at this stage and extract some scalar value, for example the amplitude of the data or the dominant frequency. Using some fancy pivoting, we can create a 2D matrix with the results as entries


```python
ex.df["result"] = None
for r in ex.df.index:
    ex.df.loc[r, "result"] = ex.results[r]['result']
    
pivoted = ex.df.pivot_table(values='result', index = 'y', columns='x', aggfunc='first')
```
<p align="center">
  <img src="https://github.com/caglorithm/mopet/raw/master/resources/pandas_pivot_table.png", width="480">
</p>

Let's plot the results!

```python

import matplotlib.pyplot as plt
# a nice color map
plt.imshow(pivoted, \
           extent = [min(ex.df.x), max(ex.df.x),
                     min(ex.df.y), max(ex.df.y)], origin='lower')
plt.colorbar(label='Distance from unit circle')
plt.xlabel("x")
plt.ylabel("y")
```

<p align="center">
  <img src="https://github.com/caglorithm/mopet/raw/master/resources/unit_circle.png", width="350">
</p>

## More information 📓

### Inspired by 🤔

`mopet` is inspired by [`pypet`](https://github.com/SmokinCaterpillar/pypet), a wonderful python parameter exploration toolkit. I have been using `pypet` for a very long time and I'm greatful for its existence! Unfortunately, the project is not maintained anymore and has run into several compatibility issues, which was the primary reason why I built `mopet`. 

### Built With 💞

`mopet` is built on other amazing open source projects:

* [`ray`](https://github.com/ray-project/ray) - A fast and simple framework for building and running distributed applications.
* [`pytables`](https://github.com/PyTables/PyTables) - A Python package to manage extremely large amounts of data.
* [`tqdm`](https://github.com/tqdm/tqdm) - A Fast, Extensible Progress Bar for Python and CLI
* [`pandas`](https://github.com/pandas-dev/pandas) - Flexible and powerful data analysis / manipulation library for Python
* [`numpy`](https://github.com/numpy/numpy) - The fundamental package for scientific computing with Python
