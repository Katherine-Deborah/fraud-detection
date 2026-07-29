"""Great Expectations checks for a batch of ingested transactions.

Runs against the deduplicated batch produced by the Airflow DAG's
ingest_batch task (dags/fraud_pipeline_dag.py), before any engineered
features are computed -- this validates the *raw* fields exactly as they
landed in the Kafka consumer's data lake (see docs/kafka.md's schema).

Uses Great Expectations' Ephemeral Data Context (in-memory, no on-disk GX
project) since there's nothing here that needs to persist between runs --
the DAG owns persistence of the batch parquet and the pass/fail outcome is
what the Airflow task result already captures.
"""

from __future__ import annotations

import great_expectations as gx
import pandas as pd
from great_expectations import expectations as gxe
from great_expectations.core.expectation_suite import ExpectationSuite

REQUIRED_FIELDS = [
    "transaction_id", "account_id", "timestamp", "amount",
    "merchant_category", "location", "device_id", "is_fraud",
]

# Generous plausibility bound, not a tight statistical one -- this exists to
# catch corruption (a units bug, a stray negative, a parsing error turning
# "12.50" into "1250000"), not to flag genuinely large legitimate purchases.
MAX_PLAUSIBLE_AMOUNT = 50_000.0


class BatchValidationError(Exception):
    """Raised when the batch fails one or more Great Expectations checks."""


def build_suite() -> ExpectationSuite:
    suite = ExpectationSuite(name="transaction_batch_suite")

    for field in REQUIRED_FIELDS:
        suite.add_expectation(gxe.ExpectColumnValuesToNotBeNull(column=field))

    suite.add_expectation(
        gxe.ExpectColumnValuesToBeUnique(column="transaction_id")
    )
    suite.add_expectation(
        gxe.ExpectColumnValuesToBeBetween(
            column="amount", min_value=0.01, max_value=MAX_PLAUSIBLE_AMOUNT
        )
    )
    suite.add_expectation(
        gxe.ExpectColumnValuesToBeInSet(column="is_fraud", value_set=[True, False])
    )
    suite.add_expectation(
        gxe.ExpectColumnValuesToMatchRegex(
            column="account_id", regex=r"^acct_\d{6}$"
        )
    )
    suite.add_expectation(
        gxe.ExpectColumnValuesToNotMatchRegex(
            column="transaction_id", regex=r"^\s*$"
        )
    )
    return suite


def validate_batch(df: pd.DataFrame) -> dict:
    """Runs the suite against df. Returns the GX result as a dict on
    success; raises BatchValidationError with the failing expectations
    listed on failure -- this is what makes the Airflow task fail visibly
    rather than silently pass a bad batch through to feature engineering.
    """
    if len(df) == 0:
        raise BatchValidationError("Batch is empty -- nothing to validate.")

    context = gx.get_context(mode="ephemeral")
    data_source = context.data_sources.add_pandas("batch_pandas_source")
    data_asset = data_source.add_dataframe_asset(name="batch_asset")
    batch_definition = data_asset.add_batch_definition_whole_dataframe("batch_def")
    batch = batch_definition.get_batch(batch_parameters={"dataframe": df})

    suite = context.suites.add(build_suite())
    result = batch.validate(suite)

    if not result.success:
        failures = [
            f"{r.expectation_config.type}({r.expectation_config.kwargs.get('column')}): "
            f"{r.result.get('unexpected_count', '?')} unexpected of "
            f"{r.result.get('element_count', '?')}"
            for r in result.results
            if not r.success
        ]
        raise BatchValidationError(
            f"Batch validation failed ({len(failures)} check(s)): " + "; ".join(failures)
        )

    return result.to_json_dict()
