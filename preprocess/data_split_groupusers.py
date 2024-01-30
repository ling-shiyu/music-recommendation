#!/usr/bin/env python
# -*- coding: utf-8 -*-

from random import random
import sys
# And pyspark.sql to get the spark session
from pyspark.sql import SparkSession
from pyspark.sql.functions import rand
from pyspark.sql.functions import count, col, floor, row_number, max, round
from pyspark.sql.window import Window


def main(spark, file_path):
    # Load the Parquet file into a DataFrame
    df = spark.read.parquet(file_path)
    df = df.withColumn("random", rand())

    window_spec = Window.partitionBy("user_id").orderBy("random")
    numbered_df = df.withColumn("row_number", row_number().over(window_spec))

    df_max_rownum = numbered_df.groupBy("user_id").agg(
        max("row_number").alias("max_row_number"))

    # Join with original DataFrame to add max_row_number column
    df_final = numbered_df.join(df_max_rownum, on="user_id")
    df_final = df_final.withColumn(
        "edge_row_number", round(col("max_row_number") * 0.8))

    training = df_final.filter(col('row_number') <= col("edge_row_number")).drop(
        "max_row_number", "row_number", "edge_row_number", "random")
    # Count the number of rows in the DataFrame
    num_rows = training.count()
    print("The training dataset has", num_rows, "rows.")

    validation = df_final.filter(col('row_number') > col("edge_row_number")).drop(
        "max_row_number", "row_number", "edge_row_number", "random")
    num_rows_val = validation.count()
    print("The validation dataset has", num_rows_val, "rows.")

    training.write.parquet(
        "hdfs:/user/yx1750_nyu_edu/final-project-group20/interactions_train_small_rating.parquet")
    validation.write.parquet(
        "hdfs:/user/yx1750_nyu_edu/final-project-group20/interactions_val_small_rating.parquet")


# Only enter this block if we're in main
if __name__ == "__main__":

    # Create the spark session object
    spark = SparkSession.builder.appName('split').getOrCreate()

    # Get file_path for dataset to analyze
    file_path = sys.argv[1]

    main(spark, file_path)
