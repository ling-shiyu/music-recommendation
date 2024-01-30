#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys

# And pyspark.sql to get the spark session
from pyspark.sql import SparkSession
from pyspark.mllib.evaluation import RankingMetrics
from pyspark.sql.functions import count, countDistinct, expr, sum


def main(spark, train_file_path, test_file_path):
    # Load the train and validation datasets
    train_df = spark.read.parquet(train_file_path)
    test_df = spark.read.parquet(test_file_path)

    # Group the interactions by track_id and calculate the total number of listens and unique users
    track_interactions = train_df.groupBy('reindex_int').agg(count('user_id').alias(
        'num_listens'))
    popularity = train_df.groupBy('reindex_int').agg(sum('scaled_rating').alias('sum_rating'))
    popularity_df = popularity.join(track_interactions, "reindex_int")
    popularity_df.show()

    beta = 60000

    # Calculate the ratio of total_listens to unique_users for each track
    track_ratios = popularity_df.withColumn(
        'popularity', popularity_df.sum_rating / (popularity_df.num_listens + beta))

    track_ratios.show()

    # The top 100 tracks with the highest listen-to-user ratio
    top_tracks = track_ratios.orderBy(
        'popularity', ascending=False).select('reindex_int').limit(100)
    
    top_tracks.show()

    predicted_ranking = [row['reindex_int']
                        for row in top_tracks.collect()]

    # Get the actual listens for each user and sorted in descending order
    user_listen_tracks = test_df.groupBy('user_id') \
        .agg(expr('sort_array(collect_list(reindex_int), False) as listened_tracks'))

    predictionAndLabels = user_listen_tracks.rdd.map(
        lambda row: (
            predicted_ranking, row['listened_tracks'])
    )

    metrics = RankingMetrics(predictionAndLabels)
    map_score = metrics.meanAveragePrecision

    # Print the MAP value for this beta value
    print('Beta = {}: test MAP = {}'.format(beta, map_score))


# Only enter this block if we're in main
if __name__ == "__main__":
    # Create the spark session object
    spark = SparkSession.builder.appName('part').getOrCreate()

    # Get file paths for train and validation datasets
    train_file_path = sys.argv[1]
    test_file_path = sys.argv[2]

    main(spark, train_file_path, test_file_path)
