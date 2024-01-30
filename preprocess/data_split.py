#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys

# And pyspark.sql to get the spark session
from pyspark.sql import SparkSession
from pyspark.sql.functions import rand


def main(spark, file_path):
    # Load the Parquet file into a DataFrame
    interactions_df = spark.read.parquet(file_path)

    interaction_data = interactions_df.orderBy(rand())

    # Split the interaction data into training and validation sets
    train_data, val_data = interaction_data.randomSplit([0.8, 0.2])

    train_data.write.parquet(
        "hdfs:/user/yx1750_nyu_edu/final-project-group20/interactions_split_train.parquet")
    val_data.write.parquet(
        "hdfs:/user/yx1750_nyu_edu/final-project-group20/interactions_split_val.parquet")


# Only enter this block if we're in main
if __name__ == "__main__":

    # Create the spark session object
    spark = SparkSession.builder.appName('split').getOrCreate()

    # Get file_path for dataset to analyze
    file_path = sys.argv[1]

    main(spark, file_path)
