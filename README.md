# Active Inference Interaction Modeling (work in progress)
Implementation of active inference for highway car following. 

## Setup
* Environment variables are in [environment.yml](environment.yml). You might run into an OMP error installing numpy, scipy along with pytorch in anaconda. You can fix this by first installing nomkl (see [here](https://stackoverflow.com/questions/53014306/error-15-initializing-libiomp5-dylib-but-found-libiomp5-dylib-already-initial)).
* Download the [INTERACTION dataset](https://interaction-dataset.com/) and perform the preprocessing steps described in [here](./doc/preprocess.md).

## Usage
To train the active inference and basedline agents, run:
```
python ./scripts/train_agent_recurrent.py
```
You can modify observation features, agent size, learning rate, training epochs by specifying additional arguments. You can use the corresponding ``.sh`` script to edit these arguments. Please see the scripts for detailed arguments.  To train agents in colab, clone the repo to google drive and run the corresponding ``.ipynb`` file. 

To test agents on static dataset, run:
```
python ./scripts/eval_offline.py
```

To test agents in simulator, run:
```
python ./scripts/eval_online.py
```
Description of the simulator can be found [here](./doc/simulation.md).