from .schema import Column, DatabaseSchema, TableSchema, add_table_stats


tpch_schema = DatabaseSchema(
    "tpch",
    table_count=8,
    tables=[
        TableSchema(
            "nation",
            [
                Column(name="n_comment", typ="character varying"),
                Column(name="n_name", typ="character"),
                Column(name="n_nationkey", typ="integer"),
                Column(name="n_regionkey", typ="integer"),
            ],
            primary_key="n_nationkey",
            large_for_index=["n_comment"],
        ),
        TableSchema(
            "region",
            [
                Column(name="r_comment", typ="character varying"),
                Column(name="r_name", typ="character"),
                Column(name="r_regionkey", typ="integer"),
            ],
            primary_key="r_regionkey",
            large_for_index=["r_comment"],
        ),
        TableSchema(
            "part",
            [
                Column(name="p_brand", typ="character"),
                Column(name="p_comment", typ="character varying"),
                Column(name="p_container", typ="character"),
                Column(name="p_mfgr", typ="character"),
                Column(name="p_name", typ="character varying"),
                Column(name="p_partkey", typ="integer"),
                Column(name="p_retailprice", typ="numeric"),
                Column(name="p_size", typ="integer"),
                Column(name="p_type", typ="character varying"),
            ],
            primary_key="p_partkey",
            large_for_index=["p_comment"],
        ),
        TableSchema(
            "supplier",
            [
                Column(name="s_acctbal", typ="numeric"),
                Column(name="s_address", typ="character varying"),
                Column(name="s_comment", typ="character varying"),
                Column(name="s_name", typ="character"),
                Column(name="s_nationkey", typ="integer"),
                Column(name="s_phone", typ="character"),
                Column(name="s_suppkey", typ="integer"),
            ],
            primary_key="s_suppkey",
            large_for_index=["s_address", "s_comment"],
        ),
        TableSchema(
            "partsupp",
            [
                Column(name="ps_availqty", typ="integer"),
                Column(name="ps_comment", typ="character varying"),
                Column(name="ps_partkey", typ="integer"),
                Column(name="ps_suppkey", typ="integer"),
                Column(name="ps_supplycost", typ="numeric"),
            ],
            primary_key=["ps_partkey", "ps_suppkey"],
            large_for_index=["ps_comment"],
        ),
        TableSchema(
            "customer",
            [
                Column(name="c_acctbal", typ="numeric"),
                Column(name="c_address", typ="character varying"),
                Column(name="c_comment", typ="character varying"),
                Column(name="c_custkey", typ="integer"),
                Column(name="c_mktsegment", typ="character"),
                Column(name="c_name", typ="character varying"),
                Column(name="c_nationkey", typ="integer"),
                Column(name="c_phone", typ="character"),
            ],
            primary_key="c_custkey",
            large_for_index=["c_address", "c_comment"],
        ),
        TableSchema(
            "orders",
            [
                Column(name="o_clerk", typ="character"),
                Column(name="o_comment", typ="character varying"),
                Column(name="o_custkey", typ="integer"),
                Column(name="o_orderdate", typ="date"),
                Column(name="o_orderkey", typ="integer"),
                Column(name="o_orderpriority", typ="character"),
                Column(name="o_orderstatus", typ="character"),
                Column(name="o_shippriority", typ="integer"),
                Column(name="o_totalprice", typ="numeric"),
            ],
            primary_key="o_orderkey",
            large_for_index=["o_comment"],
        ),
        TableSchema(
            "lineitem",
            [
                Column(name="l_comment", typ="character varying"),
                Column(name="l_commitdate", typ="date"),
                Column(name="l_discount", typ="numeric"),
                Column(name="l_extendedprice", typ="numeric"),
                Column(name="l_linenumber", typ="integer"),
                Column(name="l_linestatus", typ="character"),
                Column(name="l_orderkey", typ="integer"),
                Column(name="l_partkey", typ="integer"),
                Column(name="l_quantity", typ="numeric"),
                Column(name="l_receiptdate", typ="date"),
                Column(name="l_returnflag", typ="character"),
                Column(name="l_shipdate", typ="date"),
                Column(name="l_shipinstruct", typ="character"),
                Column(name="l_shipmode", typ="character"),
                Column(name="l_suppkey", typ="integer"),
                Column(name="l_tax", typ="numeric"),
            ],
            primary_key=["l_orderkey", "l_linenumber"],
            large_for_index=["l_comment"],
        ),
    ],
    index_count=8,
    indexes=[],
)


# 预计算表统计信息
add_table_stats(tpch_schema, "nation", 25, {"n_name": 25, "n_nationkey": 25, "n_comment": 25, "n_regionkey": 5})
add_table_stats(tpch_schema, "region", 5, {"r_regionkey": 5, "r_comment": 5, "r_name": 5})
add_table_stats(
    tpch_schema,
    "supplier",
    100000,
    {
        "s_phone": 100000,
        "s_nationkey": 25,
        "s_address": 100000,
        "s_acctbal": 95588,
        "s_name": 100000,
        "s_suppkey": 100000,
        "s_comment": 99983,
    },
)
add_table_stats(
    tpch_schema,
    "part",
    2000000,
    {
        "p_container": 40,
        "p_brand": 25,
        "p_size": 50,
        "p_mfgr": 5,
        "p_retailprice": 31681,
        "p_type": 150,
        "p_partkey": 2000000,
        "p_name": 1999828,
        "p_comment": 806046,
    },
)
add_table_stats(
    tpch_schema,
    "customer",
    1500000,
    {
        "c_custkey": 1500000,
        "c_acctbal": 818834,
        "c_address": 1500000,
        "c_nationkey": 25,
        "c_name": 1500000,
        "c_comment": 1496636,
        "c_phone": 1499963,
        "c_mktsegment": 5,
    },
)
add_table_stats(
    tpch_schema,
    "partsupp",
    8000000,
    {
        "ps_supplycost": 99901,
        "ps_partkey": 2000000,
        "ps_availqty": 9999,
        "ps_suppkey": 100000,
        "ps_comment": 7914164,
    },
)
add_table_stats(
    tpch_schema,
    "orders",
    15000000,
    {
        "o_custkey": 999982,
        "o_orderstatus": 3,
        "o_comment": 14097230,
        "o_totalprice": 11944103,
        "o_orderpriority": 5,
        "o_shippriority": 1,
        "o_clerk": 10000,
        "o_orderkey": 15000000,
        "o_orderdate": 2406,
    },
)
add_table_stats(
    tpch_schema,
    "lineitem",
    59986052,
    {
        "l_suppkey": 100000,
        "l_orderkey": 15000000,
        "l_returnflag": 3,
        "l_comment": 34378943,
        "l_shipinstruct": 4,
        "l_partkey": 2000000,
        "l_receiptdate": 2555,
        "l_discount": 11,
        "l_quantity": 50,
        "l_shipmode": 7,
        "l_extendedprice": 1351462,
        "l_tax": 9,
        "l_commitdate": 2466,
        "l_linestatus": 2,
        "l_shipdate": 2526,
        "l_linenumber": 7,
    },
)
