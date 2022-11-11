# Preprocessing instructions for the INTERACTION dataset merging scenarios
This repo contains the preprocessing, simulation, and visualization utilities for merging scenarios in the interaction dataset. The utilities are designed for car-following use cases. The four main data-related modules are listed below:
* data: geometry calculations and drive segment filtering.
* map_api: ``.osm`` map parser and conversion utilities between cartesian and frenet coordinates. Detailed descriptions of the map parsing mechanism and coordinate conversion can be found [here](./map.md). 
* simulation: sensors and single agent simulator. Details of the sensors and simulators can be found [here](./simulation.md).
* visualization: map plotting and trajectory animation.

## Dataset
Download the interaction dataset by filling out the request form on the [official website](https://interaction-dataset.com/). Organize the ```./interaction-dataset-master``` folder according to the structure below:
```
interaction-dataset-master
|---maps
|   |   location1.osm
|   |   location2.osm
|   |   ...
|
|---recorded_trackfiles
    |
    |---location1
    |   |   vehicle_tracks_000.csv
    |   |   vehicle_tracks_001.csv
    |   |   ...
    |
    |---location2
        |   vehicle_tracks_000.csv
        |   vehicle_tracks_001.csv
        |   ...
```
You may clone the official [interaction dataset repository](https://github.com/interaction-dataset/interaction-dataset) to visualize the trajectories. 

## Usage
To extract features, run:
```
python ./scripts/preprocess.py --scenario scneario_name --filename vehicle_tracks_xxx.csv --task features
```

To extract train and test labels for car-following, run:
```
python ./scripts/preprocess.py --scenario scneario_name --filename vehicle_tracks_xxx.csv --task train_labels
```

You can alternatively used the ``.sh`` script in the same folder. Make sure you check the script arguments before running. 