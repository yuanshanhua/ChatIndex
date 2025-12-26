from .schema import Column, DatabaseSchema, TableSchema


financial_schema = DatabaseSchema(
    "financial",
    table_count=8,
    tables=[
        TableSchema(
            "account",
            [
                Column(name="account_id", typ="integer"),
                Column(name="date", typ="character varying"),
                Column(name="district_id", typ="integer"),
                Column(name="frequency", typ="character varying"),
            ],
            primary_key="account_id",
        ),
        TableSchema(
            "card",
            [
                Column(name="card_id", typ="integer"),
                Column(name="disp_id", typ="integer"),
                Column(name="issued", typ="character varying"),
                Column(name="type", typ="character varying"),
            ],
            primary_key="card_id",
        ),
        TableSchema(
            "client",
            [
                Column(name="birth_date", typ="character varying"),
                Column(name="client_id", typ="integer"),
                Column(name="district_id", typ="integer"),
                Column(name="gender", typ="character varying"),
            ],
            primary_key="client_id",
        ),
        TableSchema(
            "disp",
            [
                Column(name="account_id", typ="integer"),
                Column(name="client_id", typ="integer"),
                Column(name="disp_id", typ="integer"),
                Column(name="type", typ="character varying"),
            ],
            primary_key="disp_id",
        ),
        TableSchema(
            "district",
            [
                Column(name="a10", typ="numeric"),
                Column(name="a11", typ="integer"),
                Column(name="a12", typ="numeric"),
                Column(name="a13", typ="numeric"),
                Column(name="a14", typ="integer"),
                Column(name="a15", typ="integer"),
                Column(name="a16", typ="integer"),
                Column(name="a2", typ="character varying"),
                Column(name="a3", typ="character varying"),
                Column(name="a4", typ="integer"),
                Column(name="a5", typ="integer"),
                Column(name="a6", typ="integer"),
                Column(name="a7", typ="integer"),
                Column(name="a8", typ="integer"),
                Column(name="a9", typ="integer"),
                Column(name="district_id", typ="integer"),
            ],
            primary_key="district_id",
        ),
        TableSchema(
            "loan",
            [
                Column(name="account_id", typ="integer"),
                Column(name="amount", typ="integer"),
                Column(name="date", typ="character varying"),
                Column(name="duration", typ="integer"),
                Column(name="loan_id", typ="integer"),
                Column(name="payments", typ="numeric"),
                Column(name="status", typ="character varying"),
            ],
            primary_key="loan_id",
        ),
        TableSchema(
            "order1",
            [
                Column(name="account_id", typ="integer"),
                Column(name="account_to", typ="integer"),
                Column(name="amount", typ="numeric"),
                Column(name="bank_to", typ="character varying"),
                Column(name="k_symbol", typ="character varying"),
                Column(name="order_id", typ="integer"),
            ],
            primary_key="order_id",
        ),
        TableSchema(
            "trans",
            [
                Column(name="account", typ="integer"),
                Column(name="account_id", typ="integer"),
                Column(name="amount", typ="integer"),
                Column(name="balance", typ="integer"),
                Column(name="bank", typ="character varying"),
                Column(name="date", typ="character varying"),
                Column(name="k_symbol", typ="character varying"),
                Column(name="operation", typ="character varying"),
                Column(name="trans_id", typ="integer"),
                Column(name="type", typ="character varying"),
            ],
            primary_key="trans_id",
        ),
    ],
    index_count=8,
    indexes=[],
)
