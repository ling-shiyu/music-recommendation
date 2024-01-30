#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys

# And pyspark.sql to get the spark session
from pyspark.sql import SparkSession
from pyspark.mllib.evaluation import RankingMetrics
from pyspark.sql.functions import count, countDistinct, expr


def main(spark, train_file_path, val_file_path):
    # Load the train and validation datasets
    train_df = spark.read.parquet(train_file_path)
    val_df = spark.read.parquet(val_file_path)

    # Group the interactions by track_id and calculate the total number of listens and unique users
    track_interactions = train_df.groupBy('recording_msid').agg(count('user_id').alias(
        'num_listens'), countDistinct('user_id').alias('num_unique_users'))

    # Initialize the beta hyperparameter values to test
    beta_values = [-400, -600, -800]

    # Define the list to store the MAP values for each beta value
    map_values = []

    # Train and evaluate the model for each beta value
    for beta in beta_values:
        # Calculate the ratio of total_listens to unique_users for each track
        track_ratios = track_interactions.withColumn(
            'listen_user_ratio', track_interactions.num_listens / (track_interactions.num_unique_users + beta))

        # The top 100 tracks with the highest listen-to-user ratio
        top_tracks = track_ratios.orderBy(
            'listen_user_ratio', ascending=False).select('recording_msid').limit(100)

        predicted_ranking = [row['recording_msid']
                             for row in top_tracks.collect()]

        # Get the actual listens for each user and sorted in descending order
        user_listen_tracks = val_df.groupBy('user_id') \
            .agg(expr('sort_array(collect_list(recording_msid), False) as listened_tracks'))

        predictionAndLabels = user_listen_tracks.rdd.map(
            lambda row: (
                predicted_ranking, row['listened_tracks'])
        )

        metrics = RankingMetrics(predictionAndLabels)
        map_score = metrics.meanAveragePrecision

        # Store the MAP value for this beta value
        map_values.append(map_score)

        # Print the MAP value for this beta value
        print('Beta = {}: MAP = {}'.format(beta, map_score))

    # Find the beta value with the highest MAP score
    best_beta = beta_values[map_values.index(max(map_values))]

    # Print the best beta value
    print('Best Beta Value: {}'.format(best_beta))


# Only enter this block if we're in main
if __name__ == "__main__":
    # Create the spark session object
    spark = SparkSession.builder.appName('tune_baseline').getOrCreate()

    # Get file paths for train and validation datasets
    train_file_path = sys.argv[1]
    val_file_path = sys.argv[2]

    main(spark, train_file_path, val_file_path)
