#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys

# And pyspark.sql to get the spark session
from pyspark.sql import SparkSession
from pyspark.ml.recommendation import ALS
#from pyspark.mllib.evaluation import RankingMetrics
#from pyspark.sql.functions import count, countDistinct, expr, col, monotonically_increasing_id
from pyspark.sql.functions import coalesce, dense_rank, rank
#import pyspark.sql.functions as F
from pyspark.sql.window import Window


def main(spark, file_path):

    df = spark.read.parquet(file_path)
    df = df.withColumn("track_id", coalesce(
        df.recording_mbid, df.recording_msid))

    # df = df.groupBy("track_id").agg().alias("reindex_int"))

    #window_spec = Window.partitionBy("track_id").orderBy("track_id")
    window_spec = Window.orderBy("track_id")
    df = df.withColumn("reindex_int", rank().over(window_spec))

    #df = df.withColumn("track_int_id", df.group_rank.cast("integer"))

    df.show()
    df.write.parquet(
        "hdfs:/user/yx1750_nyu_edu/final-project-group20/tracks_small_grouped_reindex.parquet")


# Only enter this block if we're in main
if __name__ == "__main__":
    # Create the spark session object
    spark = SparkSession.builder.appName('data').getOrCreate()

    # Get file paths for train and validation datasets
    file_path = sys.argv[1]

    main(spark, file_path)
