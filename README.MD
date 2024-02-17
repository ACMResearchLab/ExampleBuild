
## Startup Guide for DDQN Stock Research Proposal

This is the `README.md` for the ACM DDQN RL Stock Trading Research Proposal. Currently the project is based off the Research Paper [Deep reinforcement learning stock market trading, utilizing a CNN with candlestick images](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0263181) by Andrew Brim and Nicholas S. Flann.

After researching we found a [codebase](https://github.com/AntoniSkup/Winegold-Technologies/tree/main) written by Antoni Skup, after contacting him we were able to find some updated code from him [here](https://github.com/AntoniSkup/Winegold-Technologies-v3)

### Requirements

If you haven't already please install [anaconda](https://www.anaconda.com/download) and open a conda prompt. Here we can make new venv 

```
conda create -n ddqn-env python=3.10
```

Now we can switch to this and start loading up our requirements for the project

```
conda activate ddqn-env
```

The main things we will need can be installed with pip with the exception of the package [TA-Lib](https://ta-lib.org) to do this we will take advantage of `conda`. As the `pip install ta-lib` doesn't seem to build

**This does not need to be built if we are not creating new data**

```
conda install -c conda-forge ta-lib
```

The rest of the requirements can simply be done with a pip install, the main ones needed for this project is 

```
pip install gym
pip install tensorflow
pip install polygon
pip install python-dotenv
pip install matplotlib
```

And for later when I upload the dataset to kaggle

```
pip install kaggle
```

Its also possible to take the requirements.txt included in the GitHub after cloning it and preform the following 

```
pip install -r requirements.txt
```
### Git Cloning and Code

Now that you have a virtual environment to work in with *hopefully* I listed all the packages needed, if not I'm sure a quick run of the code will reveal any missing packages. I don't believe you need to as installing TensorFlow and some of these other packages will automatically drag `numpy` and `pandas` but if you don't have those in your environment you can do a quick pip install

To clone this repository 

```
git clone git@github.com:ACMResearchLab/ExampleBuild.git
```

This will pull in both our main branch which is closely related to the first codebase, and the *buildv2* branch is closer to the second codebase given to us.

Note if you want to create candles yourself utilizing our *candles.py* our getBarData method is dependent on polygon and the api key in the `.env` file. If you don't pay for this API, don't worry a free account in [polygon.io](https://polygon.io) will still allow you to make enough api calls to get data and form your own candle graphs. 

From here you can run the main function in either of the branches, the agent should fly by the 64 days and then start chugging along
