# Point of interest detection: Home / Work

## Installation

```
pip install -U pip
pip install -r requirements.txt
```

The csv file `technical_test_data.csv` should be in the folder `data/`

## Files

The file `exploration.ipynb` shows some statistics and how the solution was
elaborate. The file `solution.ipynb` uses the functions of `functions.py` to
display the scatter of each users with their home and work places.

## Solution description

The idea comes from a simple assumption: people go from home to  work during
week days and stay at home more often on week end days.

The first step is to extract the point of interest of the user using a
clustering algorithm `DBSCAN` and the haverstine distance metrics which fits
best for geospatial data.
We ask for groups of dots not too far from each other (around 10 meters if you
don't count the horizontal precision) and with a sufficient amount of data (at
least 100 event should have occured

## Future Work

Consecutive event during the clustering part.
Instead of counting the occurrence of each point of interest, analyze the order
of them: for instance, people are more likely doing the pattern HOME/WORK/HOME
during the week whereas during week ends the number of point of interest can
increase.


