#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys

# And pyspark.sql to get the spark session
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count


def main(spark, file_path):
    # Load the Parquet file into a DataFrame
    df = spark.read.parquet(file_path)
    df = df.filter(col("scaled_rating") != 0)
    df.show()

    # Count the number of rows in the DataFrame
    num_rows = df.count()

    # Print the result
    print("The dataset has", num_rows, "rows.")
    df.write.parquet(
        "hdfs:/user/sl9344_nyu_edu/als_small_train_drop0.parquet")





# Only enter this block if we're in main
if __name__ == "__main__":
    # Create the spark session object
    spark = SparkSession.builder.appName('proprocess').getOrCreate()

    # Get file_path for dataset to analyze
    file_path = sys.argv[1]

    main(spark, file_path)
