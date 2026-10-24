"""
Module: spark_session.py
Description: Initializes and configures the PySpark SparkSession.
Includes automatic Windows Hadoop binary (winutils) configuration.
"""

import os
import sys
from pyspark.sql import SparkSession


def get_spark_session(app_name: str = "JobRecommendationSystem", memory: str = "4g") -> SparkSession:
    """
    Creates or retrieves an active PySpark SparkSession with custom memory and Hadoop configuration.
    
    Parameters:
        app_name (str): Name of the PySpark application.
        memory (str): Driver memory allocation (default '4g').
        
    Returns:
        SparkSession: The active PySpark session.
    """
    # Configure HADOOP_HOME on Windows to support local Parquet file operations
    hadoop_dir = os.path.abspath("hadoop")
    if os.path.exists(hadoop_dir):
        os.environ["HADOOP_HOME"] = hadoop_dir
        bin_dir = os.path.join(hadoop_dir, "bin")
        if bin_dir not in os.environ.get("PATH", ""):
            os.environ["PATH"] = bin_dir + os.path.pathsep + os.environ.get("PATH", "")

    builder = (
        SparkSession.builder
        .appName(app_name)
        .master("local[*]")
        .config("spark.driver.memory", memory)
        .config("spark.executor.memory", memory)
        .config("spark.sql.shuffle.partitions", "8")
        .config("spark.driver.host", "127.0.0.1")
        .config("spark.driver.bindAddress", "127.0.0.1")
    )
    
    spark = builder.getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    
    return spark


if __name__ == "__main__":
    print("Testing PySpark SparkSession initialization...")
    spark = get_spark_session()
    print(f"SparkSession created successfully! Version: {spark.version}")
    spark.stop()
