#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys

# And pyspark.sql to get the spark session
from pyspark.sql import SparkSession
from pyspark.ml.recommendation import ALS
#from pyspark.mllib.evaluation import RankingMetrics
#from pyspark.sql.functions import count, countDistinct, expr, col, monotonically_increasing_id
from pyspark.sql.functions import coalesce, dense_rank, rank, count, col
#import pyspark.sql.functions as F
from pyspark.sql.window import Window



def main(spark, interaction_file_path, tracks_file_path):

    df_interaction = spark.read.parquet(interaction_file_path)
    df_tracks = spark.read.parquet(tracks_file_path)
    
    merged_df = df_interaction.join(df_tracks, "recording_msid")

    # # Drop rows where the 'user_id' column is null or NaN
    # merged_df = merged_df.filter("user_id is not null")
    # # Group by user_id and recording_msid, and count the number of rows
    # count_df = merged_df.groupBy("user_id", "reindex_int").agg(count("*").alias("num_listens"))
    # #count_df.show()

    # # Group by user_id and calculate the total number of recordings the user has listened to
    # total_listens_df = count_df.groupBy("user_id").agg(count("*").alias("total_listens"))
    # #total_listens_df.show()

    # # Join the two dataframes to get the number of listens per recording per user
    # #rating_df
    # rating_df = count_df.join(total_listens_df, "user_id").withColumn("rating", col("num_listens") / col("total_listens"))

    # # Show the resulting dataframe
    # rating_df.show()

    # num_rows = rating_df.count()
    # print("The rating dataset has", num_rows, "rows.")
    merged_df.write.parquet(
        "hdfs:/user/sl9344_nyu_edu/interactions_test_reindex.parquet")



# Only enter this block if we're in main
if __name__ == "__main__":
    # Create the spark session object
    spark = SparkSession.builder.appName('test').getOrCreate()

    # Get file paths for train and validation datasets
    interaction_file_path = sys.argv[1]
    tracks_file_path = sys.argv[2]
    

    main(spark, interaction_file_path, tracks_file_path)
