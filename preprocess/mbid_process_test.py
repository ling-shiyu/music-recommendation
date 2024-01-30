#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys

# And pyspark.sql to get the spark session
from pyspark.sql import SparkSession
from pyspark.ml.recommendation import ALS
from pyspark.sql.functions import coalesce, dense_rank, rank
#from pyspark.sql.functions import when, col
#import pyspark.sql.functions as F
from pyspark.sql.window import Window



def main(spark, trainmbid_file_path, tracks_test_file_path):

    df_train = spark.read.parquet(trainmbid_file_path)
    df_test = spark.read.parquet(tracks_test_file_path)

    test_combined = df_test.join(df_train, "recording_msid")
    test_combined = test_combined.drop("artist_name", "recording_mbid", "track_name")
    #checked that there is no new mbid/msid from train data
    #df_test = df_test.withColumn("track_test_id", coalesce(df_test.recording_mbid, df_test.recording_msid))
    #test_combined = df_test.join(df_train, "recording_msid")
    #test_combined = test_combined.filter("track_test_id is not null")

    #test_combined = test_combined.withColumn("reindex_int_test", when(col("tracks_id") == col("tracks_test_id"), col("reindex_int")).otherwise(col("tracks_test_id")))

    test_combined.show()

    test_combined.write.parquet(
        "hdfs:/user/sl9344_nyu_edu/tracks_test_reindex.parquet")



# Only enter this block if we're in main
if __name__ == "__main__":
    # Create the spark session object
    spark = SparkSession.builder.appName('data').getOrCreate()

    # Get file paths for train and validation datasets
    trainmbid_file_path = sys.argv[1]
    tracks_test_file_path = sys.argv[2]
    

    main(spark, trainmbid_file_path, tracks_test_file_path)
