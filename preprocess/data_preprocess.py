#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys

# And pyspark.sql to get the spark session
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count


def main(spark, file_path):
    # Load the Parquet file into a DataFrame
    interactions_df = spark.read.parquet(file_path)

    # Count the number of rows in the DataFrame
    num_rows = interactions_df.count()

    # Print the result
    print("The dataset has", num_rows, "rows.")
    # The dataset has 179466123 rows.

    # Calculate the number of user interactions for each recording_msid
    num_interactions_df = interactions_df.groupBy("recording_msid") \
        .agg(count("user_id").alias("num_interactions"))

    # Calculate the mean, min, and max number of user interactions
    # stats_df = num_interactions_df.agg(mean("num_interactions").alias("mean_num_interactions"),
    #                                    min("num_interactions").alias(
    #                                        "min_num_interactions"),
    #                                    max("num_interactions").alias("max_num_interactions"))

    # Show the results
    # num_interactions_df.show()
    # stats_df.show()

    # Filter out recording_msids with 3 or fewer interactions
    filtered_df = num_interactions_df.filter("num_interactions > 3")

    # Join the filtered DataFrame with the original interactions DataFrame to get the complete records
    joined_df = interactions_df.join(filtered_df, "recording_msid", "inner")
    # The dataset has 148787192 rows now.

    # Save the output as a new Parquet file
    joined_df.write.parquet(
        "hdfs:/user/yx1750_nyu_edu/final-project-group20/interactions_train_process_small.parquet")


# Only enter this block if we're in main
if __name__ == "__main__":
    # Create the spark session object
    spark = SparkSession.builder.appName('preprocess').getOrCreate()

    # Get file_path for dataset to analyze
    file_path = sys.argv[1]

    main(spark, file_path)
