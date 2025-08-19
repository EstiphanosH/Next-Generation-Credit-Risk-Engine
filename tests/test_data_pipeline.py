"""Test data pipeline transformations."""
import pytest
from scripts.data_pipeline import process_raw_data

def test_pipeline_output(spark_session):
    """Verify Delta table creation."""
    process_raw_data()
    assert DeltaTable.isDeltaTable(spark_session, "data/processed/transactions")
