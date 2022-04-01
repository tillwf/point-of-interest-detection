import pandas as pd
import logging

from collections import Counter
from datetime import datetime
from math import radians, cos, sin, asin, sqrt
from sklearn.cluster import DBSCAN

DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
WEEK_DAYS = [13, 14, 15, 16, 17, 20, 21, 22, 23, 24]
WEEK_END_DAYS = [18, 19, 25, 26]

logger = logging.getLogger(__name__)


def read_data(path="data/technical_test_data.csv"):
    """ Read csv as dataframe """
    print("Reading the data...")
    return pd.read_csv(
        path,
        names=[
            "user_id",
            "timestamp",
            "latitude",
            "longitude",
            "horizontal_precision",
            "speed",
            "crc32_hash"
        ]
    )

def enrich_df(df):
    """ Add multiple columns to the dataframe `df`
     - datetime: datetime conversion of `timestamp`
     - date: date conversion of `datetime`
     - weekday/dayofyear: extracted value from `datetime`
     - timestamp_norm: time in second elapsed since first day at midnight
     - speed_kmh: speed in km/s for better understanding
    """
    print("Enrich the data...")
    df["datetime"] = df.timestamp.apply(pd.Timestamp.fromtimestamp)
    df["date"] = df["datetime"].dt.date
    df["weekday"] = df["datetime"].dt.weekday
    df["dayofyear"] = df["datetime"].dt.dayofyear
    df["speed_kmh"] = df["speed"]*3600/1000

    min_date = df["date"].min()
    dt = datetime(
            year=min_date.year,
            month=min_date.month,
            day=min_date.day
         )
    min_timestamp = int(dt.timestamp())

    df["timestamp_norm"] = df["timestamp"] - min_timestamp

    return df


def filter_df(df, speed=5, horizontal_precision=66):
    """ Remove lines from `df`
    Here we remove speed value above 5 as people at work or at home don't run
    that much
    We also remove data point where the horizontal value is to high (above the mode)
    """
    print("Filter the data...")
    return df[
        (df["speed_kmh"] < speed) &
        (df["horizontal_precision"] < horizontal_precision)
    ]


def plot_scatter(df, user_id):
    """ Plot scatter with one color for each day """
    import matplotlib.pylab as plt
    sub_df = df.query("user_id == {}".format(user_id))

    x = sub_df.latitude.values
    y = sub_df.longitude.values
    c = sub_df.dayofyear.values

    fig, ax = plt.subplots()
    ax.scatter(
        x,
        y,
        c=c,
        alpha=0.1,
        label=c,
        cmap=plt.get_cmap("Dark2")
    )
    ax.legend()
    plt.show()


def haversine(X, Y):
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)
    """
    lat1, lon1 = X
    lat2, lon2 = Y

    # convert decimal degrees to radians
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

    # haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    # Radius of earth in kilometers. Use 3956 for miles
    r = 6371
    return c * r


def haversine_acc(X, Y):
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees) adding an accuracy value:
    We take the mean distance between the max possible distance if you add the
    horizontal accuracy to each point and the minimum distance by removing this
    accuracy (0 if the substraction is negative)
    """
    lat1, lon1, ac1 = X
    lat2, lon2, ac2 = Y

    ac1 = ac1 * 0.001
    ac2 = ac2 * 0.001

    distance = haversine((lat1, lon1), (lat2, lon2))

    min_distance = max(0, distance - ac2 - ac1)
    max_distance = distance + ac1 + ac2

    return (max_distance + min_distance) / 2


def do_clustering(df, eps=0.01, min_samples=100):
    """ Cluster spatial point using DBSCAN
    - eps: The minimal distance to consider point to be close (in km)
    - min_samples: The minimum number of dots to create a cluster
    """
    print("Clustering the data...")
    db = DBSCAN(
        eps=eps,
        min_samples=min_samples,
        algorithm='ball_tree',
        n_jobs=-1,
        metric=lambda X, Y: haversine(X, Y)
    )
    db.fit(
        df[['latitude', 'longitude']]
    )
    print("Clustering done.")
    return db.labels_


def plot_point_of_interest(df):
    """ Plot the point of interest based on the column `label` """
    import matplotlib.pylab as plt
    df['label'] = db.labels_
    df.query("label >= 0").plot.scatter(
        x="latitude",
        y="longitude",
    #   s="horizontal_precision",
        c="label",
        cmap=plt.get_cmap("Dark2"))


def plot_home_work(df):
    """ Plot the scatterplot and the home and work predicted """
    import matplotlib.pylab as plt
    point_of_interest = (df.query("label >= 0")
                 .groupby('dayofyear')
                 .label
                 .apply(Counter)
                 .unstack()
    )
    top_two_classes = (point_of_interest.sum(axis=0)
                                        .sort_values(ascending=False)
                                        .iloc[:2]
                                        .index
                                        .values)

    top_two_class_events = point_of_interest[top_two_classes]
    work_class = top_two_class_events.loc[WEEK_DAYS].sum(axis=0).idxmax()
    home_class = top_two_class_events.loc[WEEK_END_DAYS].sum(axis=0).idxmax()

    work_location = df.query("label == {}".format(work_class))[['latitude', 'longitude']].mean()
    home_location = df.query("label == {}".format(home_class))[['latitude', 'longitude']].mean()

    x = df.latitude.values
    y = df.longitude.values
    c = df.dayofyear.values
    l = df.label.values

    fig, ax = plt.subplots()
    ax.scatter(
        x,
        y,
        alpha=0.1,
    )
    ax.scatter(
        x=work_location['latitude'],
        y=work_location['longitude'],
        c="red",
        label="Work",
        alpha=1,
    )
    ax.scatter(
        x=home_location['latitude'],
        y=home_location['longitude'],
        c="orange",
        label="Home",
        alpha=1,
    )
    ax.legend()
    plt.show()
