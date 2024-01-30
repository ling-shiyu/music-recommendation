#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys

# And pyspark.sql to get the spark session
from pyspark.sql import SparkSession
from pyspark.mllib.evaluation import RankingMetrics
from pyspark.sql.functions import count, countDistinct, expr


def main(spark, train_file_path, test_file_path):
    # Load the train and validation datasets
    train_df = spark.read.parquet(train_file_path)
    test_df = spark.read.parquet(test_file_path)

    # Group the interactions by track_id and calculate the total number of listens and unique users
    track_interactions = train_df.groupBy('recording_msid').agg(count('user_id').alias(
        'num_listens'), countDistinct('user_id').alias('num_unique_users'))

    beta = -500

    # Calculate the ratio of total_listens to unique_users for each track
    track_ratios = track_interactions.withColumn(
        'listen_user_ratio', track_interactions.num_listens / (track_interactions.num_unique_users + beta))

    # The top 100 tracks with the highest listen-to-user ratio
    top_tracks = track_ratios.orderBy(
        'listen_user_ratio', ascending=False).select('recording_msid').limit(100)

    predicted_ranking = [row['recording_msid']
                         for row in top_tracks.collect()]

    # Get the actual listens for each user and sorted in descending order
    user_listen_tracks = test_df.groupBy('user_id') \
        .agg(expr('sort_array(collect_list(recording_msid), False) as listened_tracks'))

    predictionAndLabels = user_listen_tracks.rdd.map(
        lambda row: (
            predicted_ranking, row['listened_tracks'])
    )

    metrics = RankingMetrics(predictionAndLabels)
    map_score = metrics.meanAveragePrecision

    # Print the MAP value for this beta value
    print('Beta = {}: Test MAP = {}'.format(beta, map_score))


# Only enter this block if we're in main
if __name__ == "__main__":
    # Create the spark session object
    spark = SparkSession.builder.appName('test_baseline').getOrCreate()

    # Get file paths for train and validation datasets
    train_file_path = sys.argv[1]
    test_file_path = sys.argv[2]

    main(spark, train_file_path, test_file_path)
