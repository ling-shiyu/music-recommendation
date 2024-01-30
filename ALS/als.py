#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys

# And pyspark.sql to get the spark session
from pyspark.sql import SparkSession
from pyspark.mllib.evaluation import RankingMetrics
from pyspark.sql.functions import count, countDistinct, expr, col, monotonically_increasing_id
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
from pyspark.sql import Row
from pyspark.ml.feature import MinMaxScaler, VectorAssembler
from pyspark.ml.linalg import Vectors
import time



def main(spark, train_file_path, val_file_path):
    # Load the train and validation datasets
    train_df = spark.read.parquet(train_file_path)
    val_df = spark.read.parquet(val_file_path)
    train_df = train_df.filter(col("scaled_rating") != 0)
    

    
    # Define the rank (dimension) of the latent factors
    rank = 100
    # Define the implicit feedback parameter (alpha)
    alpha = 1.0
    # Define the regularization parameter
    regParam = 1
    train_df.show()


    start = time.process_time()
    als = ALS(rank = rank, alpha = alpha, maxIter=5, regParam=regParam, userCol="user_id", itemCol="reindex_int", ratingCol="scaled_rating",
          coldStartStrategy="drop")
    model = als.fit(train_df)
    end = time.process_time()
    elapsed_time = end - start
    print("Time to fit model: ", elapsed_time)




    # Evaluate the model on the validation data
    predictions = model.recommendForAllUsers(100)

    # Get the actual listens for each user and sorted in descending order
    user_listen_tracks = val_df.groupBy('user_id').agg(
        expr('sort_array(collect_list(reindex_int), False) as listened_tracks'))

    # Join with ground truth data to get predictionAndLabels
    user_listen_tracks_with_predictions = user_listen_tracks.join(
        predictions, 'user_id')
    predictionAndLabels = user_listen_tracks_with_predictions.rdd.map(
        lambda row: (
            [prediction.reindex_int for prediction in row['recommendations']],
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
    spark = SparkSession.builder.appName('als').getOrCreate()

    # Get file paths for train and validation datasets
    train_file_path = sys.argv[1]
    val_file_path = sys.argv[2]
    # tracks_file_path = sys.argv[3]

    main(spark, train_file_path, val_file_path)
