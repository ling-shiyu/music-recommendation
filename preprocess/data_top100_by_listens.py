#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys

# And pyspark.sql to get the spark session
from pyspark.sql import SparkSession
from pyspark.sql.functions import count


def main(spark, file_path):
    df = spark.read.parquet(file_path)

    # group by recording_msid, count the number of distinct user_id, and sort in descending order
    popular_tracks = df.groupBy("recording_msid").agg(count("user_id").alias("num_users"))\
                       .sort("num_users", ascending=False)

    # select the top 100 most popular tracks
    top_100_tracks = popular_tracks.limit(100)

    # Write the train and val sets to separate Parquet files
    top_100_tracks.write.parquet(
        'hdfs:/user/yx1750_nyu_edu/final-project-group20/top100.parquet')


# Only enter this block if we're in main
if __name__ == "__main__":
    # Create the spark session object
    spark = SparkSession.builder.appName('top100').getOrCreate()

    # Get file_path for dataset to analyze
    file_path = sys.argv[1]

    main(spark, file_path)
