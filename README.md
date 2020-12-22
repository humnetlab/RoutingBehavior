# Understanding Routing Behavior with LBS Data
Source code for "Understanding vehicular routing behavior with location-based service data"

Developed by Yanyan Xu (yanyanxu@sjtu.edu.cn) and Marta Gonzalez

Human Mobility and NNetworks Lab, UC Berkeley


This data analysis framework process the raw LBS data, extract the vehicule trips, and detect routinng behavior. For each active LBS use, we select her top OP pair with the largest number of trips for further routing behavior analysis. The simple rule based route dectection is illustrated as follows,

![alt text](./images/routesdetection.png?raw=true)

To understand the individual routing behavior, we connect the user's number of routes with the travel distance, the number of trips, the departure time, the travel time index and the buffer index. Empirical results demonstrate that during the peak hours, travelers tend to reduce the impact of traffic congestion by taking alternative routes.

This work is implemented with Python3.8. Related packages include numpy, scipy, matplotlib, scikitlearn, geojson, fiona, rtee.

## Struture of source code:

#### step0_user_selection.py: 
(1) analysis of the raw LBS data

(2) select active users from the raw LBS data

#### step1_trip_processing.py
(1) trip segmentation

(2) vehicular trips detection

(3) find home location of LBS users

(4) origin and destination detection

(5) route detection

#### step2_routing_behavior.py
(1) Correlation between routing behavior and travel characteristics

(2) Calcualte travel time index (TTI) and buffer index (BI), and relate them to the routing behavior

#### step3_ODmatrix.py
(1) Expand the travel demand of LBS users to the population level

(2) Compare the OD matrix with the NCTCOG travel survey

