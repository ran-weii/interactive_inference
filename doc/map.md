# Maps
This page documents the parsing procedures of ``.osm`` maps in the interaction dataset and functions of the extracted map object. This module was written because of the difficulty of installing the [Lanelet2 library](https://github.com/fzi-forschungszentrum-informatik/Lanelet2) on Mac. 

## Lanelet2 format
Lanelet2 is a map representation framework proposed for autonomous vehicles. A lanelet is a small road segment, usually a polygon. A lanelet2 map, stored in ``.osm`` files, consists of a set of nodes defining the keys points on the lanes and other road objects, a set of ways connecting nodes into a line string, and a set of relations associating ways to a particular lanelet. The relations also specify whether a ways in the left bound or right bound of a lanelet. 


## Frenet coordinate
The frenet coordinate is a path-centric coordinate. In the 2D case, the x and y axes are defined as the normal and tangent line at each point along the reference path (usually the lane centerline in car following). Tangent and normal lines of the reference path can be computed by fitting a high order polynomial to the path, computing the polynomial derivatives and then applying the tangent formula. Curvature at a specific tangent point along the reference path can be computed similarly. Using the tangent and curvature of the path, we can transform a point in the frenet coordinate to the cartesian coordinate and back. In our implementation, the frenet coordinate is defined as tuple ``[s, ds, dds, d, dd, ddd]`` where ``s`` is the longitudinal component and the last ``d`` in each item denotes the lateral component. The ``d``s in front of a letter denotes order of derivative. The cartesian coordinate is defined as tuple ``[x, y, v, a, theta, kappa]``, where ``theta`` and ``kappa`` are the heading the curvature of the point's path; ``v`` and ``a`` are signed velocity and acceleration in the direction of ``theta``. We adapte the coordinate transformation from [Baidu Apollo's implementation](https://github.com/ApolloAuto/apollo).

## The ``MapReader`` object

### Parsing
The ``parse`` method extracts lanelets from an ``.osm`` file and store it in ``MapReader``'s properties. Parsing proceeds in the order of nodes -> ways -> relations to fillup ``MapReader``'s ``points``, ``linestrings``, and ``lanelets`` dictionaries. The ``Lanelet`` objects will compute the heading direction of the lanelet based on the left and right bound indicators stored in the relations and align the left and right bounds along the heading direction. At the same time, it will divide itself into a number of smaller polygons stored as ``Cell`` objects with lengths specified by the ``cell_len`` parameter. It then computes the centerline of the ``Cell`` objects and merge the cell centerlines to form its own centerline. 

Once all the lanelets are extracted, ``MapReader`` will automatically identify lanes as lanelets with connected boundaries and store grouped lanelets into a ``Lane`` object. The ``Lane`` object will then align the lanelets by forming a directed graph based on the lanelet headings and perform a topological sort. It will then extract its left and right bounds and centerline by connecting those of its lanelets. It will also store a ``FrenetPath`` object of the centerline line. 

### Matching
The ``match_lane`` method matches a rectangle with heading, length, and width to a lane in the map by finding a lane that has the highest amount of overlap with the rectangle's area. If there is no lane which has a positive overlap with the rectangle, the methods returns ``None``. 

### Visualization
The ``plot`` method plots the map with four options: ``ways, lanelets, cells, lanes``. The most used option is ``lanes``. You can find out about other options by playing with the ``./tests/test_map_api.py`` scripts. 