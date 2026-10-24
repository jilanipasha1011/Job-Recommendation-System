"""
Module: data_ingestion.py
Description: Data ingestion and inspection functions using PySpark.

Why PySpark DataFrames?
- PySpark DataFrames provide distributed, memory-optimized table processing.
- They allow schema inference and precise handling of CSV quoting/escaping for fields containing complex text.
- Null checks and schema metadata extraction are performed natively in PySpark without converting to Pandas.
"""

from typing import Dict, Any, List
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F


def load_job_data(spark: SparkSession, file_path: str) -> DataFrame:
    """
    Loads the job dataset CSV file into a PySpark DataFrame.
    
    Parameters:
        spark (SparkSession): Active SparkSession instance.
        file_path (str): Path to the CSV file.
        
    Returns:
        DataFrame: PySpark DataFrame containing the job dataset.
    """
    print(f"Loading dataset from: {file_path} using PySpark...")
    df = (
        spark.read
        .option("header", "true")
        .option("inferSchema", "true")
        .option("quote", '"')
        .option("escape", '"')
        .option("multiLine", "true")  # Handle multi-line text fields in Job Description/Responsibilities safely
        .csv(file_path)
    )
    return df


def get_dataset_summary(df: DataFrame) -> Dict[str, Any]:
    """
    Extracts structural summary of the PySpark DataFrame including row count, column count, and schema.
    
    Parameters:
        df (DataFrame): PySpark DataFrame.
        
    Returns:
        dict: Summary metadata dictionary.
    """
    total_rows = df.count()
    columns = df.columns
    schema_info = [(field.name, str(field.dataType)) for field in df.schema.fields]
    
    return {
        "total_rows": total_rows,
        "total_columns": len(columns),
        "columns": columns,
        "schema": schema_info
    }


def analyze_missing_values(df: DataFrame) -> List[Dict[str, Any]]:
    """
    Calculates missing/null count and missing percentage for every column using PySpark SQL functions.
    
    Parameters:
        df (DataFrame): PySpark DataFrame.
        
    Returns:
        List[Dict[str, Any]]: Summary list with column name, null count, and percentage missing.
    """
    total_rows = df.count()
    if total_rows == 0:
        return []

    # Build PySpark aggregations for every column
    # Check for null, NaN (if numeric), or empty string
    agg_exprs = []
    for col_name, data_type in df.dtypes:
        is_string = data_type.startswith("string")
        is_float_or_double = data_type in ["double", "float"]

        cond = F.col(f"`{col_name}`").isNull()
        if is_string:
            cond = cond | (F.trim(F.col(f"`{col_name}`")) == "")
        elif is_float_or_double:
            cond = cond | F.isnan(F.col(f"`{col_name}`"))

        agg_exprs.append(F.count(F.when(cond, 1)).alias(col_name))

    # Execute aggregation in a single PySpark pass
    null_counts_row = df.select(agg_exprs).collect()[0]

    missing_summary = []
    for col_name in df.columns:
        null_count = null_counts_row[col_name]
        pct = (null_count / total_rows) * 100
        missing_summary.append({
            "column": col_name,
            "null_count": null_count,
            "missing_pct": round(pct, 2)
        })

    return missing_summary


if __name__ == "__main__":
    from src.spark_session import get_spark_session

    print("Testing data_ingestion module independently...")
    spark = get_spark_session()
    df = load_job_data(spark, "job_descriptions.csv")
    summary = get_dataset_summary(df)
    print(f"Total Rows: {summary['total_rows']}")
    print(f"Total Columns: {summary['total_columns']}")
    print("Detected Columns:")
    for col, dtype in summary['schema']:
        print(f" - {col} ({dtype})")
    
    missing_info = analyze_missing_values(df)
    print("\nMissing Value Summary:")
    for item in missing_info:
        print(f" - {item['column']}: {item['null_count']} missing ({item['missing_pct']}%)")
    
    spark.stop()
