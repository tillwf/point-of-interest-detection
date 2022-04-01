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

The idea comes from a simple assumption: people go from home to work during
week days and stay at home more often on week end days.

The first step is to extract the points of interest of the user using a
clustering algorithm `DBSCAN` and the haverstine distance metrics which fits
best for geospatial data.
We try to find groups of dots not too far from each other (around 10 meters if you
don't count the horizontal precision) and with a sufficient amount of data (at
least 100 event should have occured).

Then we take the top two most visited places and match them to the week days
and the week end days to separate home from work.

### Users predictions

#### 861100071
![861100071](https://user-images.githubusercontent.com/7115035/161342921-426bfcdb-af77-4571-87a1-04db19ce547b.png)

#### 1853210804
![1853210804](https://user-images.githubusercontent.com/7115035/161342924-a7b51653-c3d2-4ebd-ae37-ed2d5478a17d.png)

#### 3330315587
![3330315587](https://user-images.githubusercontent.com/7115035/161342926-c7697022-6389-48a0-8c3e-864f3adc40b3.png)

## Future Work

- Consecutive event during the clustering part.
- Instead of counting the occurrence of each point of interest, analyze the order
of them: for instance, people are more likely doing the pattern HOME/WORK/HOME
during the week whereas during week ends the number of point of interest can
increase.
- Use values of latitude and longitude to map point of interest to actual places (using an external API)

