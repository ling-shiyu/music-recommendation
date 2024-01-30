#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys

# And pyspark.sql to get the spark session
from pyspark.sql import SparkSession
from pyspark.sql.functions import count, countDistinct


def main(spark, file_path):
    df = spark.read.parquet(file_path)

    # Group the interactions by track_id and calculate the total number of listens and unique users
    track_interactions = df.groupBy('recording_msid').agg(count('user_id').alias(
        'num_listens'), countDistinct('user_id').alias('num_unique_users'))

    # set hyperparameter
    beta = 10

    # Calculate the ratio of total_listens to unique_users for each track
    track_ratios = track_interactions.withColumn(
        'listen_user_ratio', track_interactions.num_listens / (track_interactions.num_unique_users + beta))

    # Show the top 100 tracks with the highest listen-to-user ratio
    output = track_ratios.orderBy('listen_user_ratio', ascending=False).select(
        'recording_msid').limit(100)
    output.show()

    # Write the train and val sets to separate Parquet files
    track_ratios.write.parquet(
        'hdfs:/user/yx1750_nyu_edu/final-project-group20/top100_by_ratio.parquet')


# Only enter this block if we're in main
if __name__ == "__main__":
    # Create the spark session object
    spark = SparkSession.builder.appName('train_baseline').getOrCreate()

    # Get file_path for dataset to analyze
    file_path = sys.argv[1]

    main(spark, file_path)
