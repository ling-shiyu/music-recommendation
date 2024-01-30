#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys

# And pyspark.sql to get the spark session
from pyspark.sql import SparkSession
from pyspark.ml.recommendation import ALS
from pyspark.mllib.evaluation import RankingMetrics
from pyspark.sql.functions import count, countDistinct, expr, col, monotonically_increasing_id


def main(spark, train_file_path, val_file_path):
    # Load the train and validation datasets
    train_df = spark.read.parquet(train_file_path)
    val_df = spark.read.parquet(val_file_path)
    # tracks_df = spark.read.parquet(tracks_file_path)

    # Group the interactions by track_id and calculate the total number of listens and unique users
    track_interactions = train_df.groupBy('recording_msid').agg(count('user_id').alias(
        'num_listens'), countDistinct('user_id').alias('num_unique_users'))

    beta = -500

    # Calculate the ratio of total_listens to unique_users for each track
    track_ratios = track_interactions.withColumn(
        'listen_user_ratio', track_interactions.num_listens / (track_interactions.num_unique_users + beta))

    # Join the track ratios dataframe with the train dataframe
    train_with_ratios = train_df.join(track_ratios, on='recording_msid')

    indexed_track_model = train_with_ratios.select(
        'recording_msid',
        'user_id',
        'listen_user_ratio',
    ).distinct().withColumn('track_index',  monotonically_increasing_id())

    # Cast relevant columns to float
    indexed_track_model = indexed_track_model.select(
        col('user_id').cast('integer'),
        col('track_index').cast('integer'),
        col('listen_user_ratio').cast('float')
    )

    # Define the rank (dimension) of the latent factors
    rank = 10
    # Define the implicit feedback parameter (alpha)
    alpha = 40.0
    # Define the regularization parameter
    regParam = 0.01
    # Train the ALS model on the interaction data
    als = ALS(rank=rank, maxIter=10, implicitPrefs=True, alpha=alpha, regParam=regParam,
              userCol='user_id', itemCol='track_index', ratingCol='listen_user_ratio')
    model = als.fit(indexed_track_model)

    # Evaluate the model on the validation data
    predictions = model.recommendForAllUsers(100)

    # Get the actual listens for each user and sorted in descending order
    user_listen_tracks = val_df.groupBy('user_id').agg(
        expr('sort_array(collect_list(recording_msid), False) as listened_tracks'))

    # Join with ground truth data to get predictionAndLabels
    user_listen_tracks_with_predictions = user_listen_tracks.join(
        predictions, 'user_id')
    predictionAndLabels = user_listen_tracks_with_predictions.rdd.map(
        lambda row: (
            [prediction.track_index for prediction in row['recommendations']],
            row['listened_tracks']
        )
    )

    metrics = RankingMetrics(predictionAndLabels)
    map_score = metrics.meanAveragePrecision

    # Store the MAP value for this beta value
    # map_values.append(map_score)

    # Print the MAP value for this beta value
    print('rank = {}, alpha = {}, regParam = {} : MAP = {}'.format(
        rank, alpha, regParam, map_score))


# Only enter this block if we're in main
if __name__ == "__main__":
    # Create the spark session object
    spark = SparkSession.builder.appName('LF').getOrCreate()

    # Get file paths for train and validation datasets
    train_file_path = sys.argv[1]
    val_file_path = sys.argv[2]
    # tracks_file_path = sys.argv[3]

    main(spark, train_file_path, val_file_path)
