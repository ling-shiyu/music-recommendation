#!/usr/bin/env python
# -*- coding: utf-8 -*-

from random import random
import sys
# And pyspark.sql to get the spark session
from pyspark.sql import SparkSession


def sample_udf(df):
    return df.sample(frac=0.8, random_state=random.randint(0, 1000))


def main(spark, file_path):
    # Load the Parquet file into a DataFrame
    df = spark.read.parquet(file_path)

    training = df.groupby('user_id').apply(sample_udf).toPandas()
    validation = df.loc[set(df.index) - set(training.index)]

    training = training.reset_index(drop=True)
    validation = validation.reset_index(drop=True)

    # Save the train and validation DataFrames to Parquet files
    training_spark = spark.createDataFrame(training)
    validation_spark = spark.createDataFrame(validation)

    training_spark.write.parquet(
        "hdfs:/user/yx1750_nyu_edu/final-project-group20/interactions_split_train.parquet")
    validation_spark.write.parquet(
        "hdfs:/user/yx1750_nyu_edu/final-project-group20/interactions_split_val.parquet")


# Only enter this block if we're in main
if __name__ == "__main__":

    # Create the spark session object
    spark = SparkSession.builder.appName('split').getOrCreate()

    # Get file_path for dataset to analyze
    file_path = sys.argv[1]

    main(spark, file_path)
