from .accidents import accidents_schema
from .airline import airline_schema
from .basketball import basketball_schema
from .chembl import chembl_schema
from .credit import credit_schema
from .dsb import dsb_schema
from .ergastf1 import ergastf1_schema
from .financial import financial_schema
from .fnhk import fnhk_schema
from .gifshow import gifshow_schema
from .grants import grants_schema
from .hockey import hockey_schema
from .imdb import imdb_schema
from .legalacts import legalacts_schema
from .sap import sap_schema
from .schema import ColumnCoder, ColumnMetric, DatabaseSchema, Index, parse_indexes_str
from .ssb import ssb_schema
from .stats import stats_schema
from .talkingdata import talkingdata_schema
from .tpcds import tpcds_schema
from .tpch import tpch_schema


schemas = {
    "imdb": imdb_schema,
    "tpch": tpch_schema,
    "tpch1g": tpch_schema,
    "tpcds": tpcds_schema,
    "dsb": dsb_schema,
    "gifshow": gifshow_schema,
    # 以下为训练集
    "accidents": accidents_schema,
    "airline": airline_schema,
    "basketball": basketball_schema,
    "chembl": chembl_schema,
    "credit": credit_schema,
    "ergastf1": ergastf1_schema,
    "financial": financial_schema,
    "fnhk": fnhk_schema,
    "grants": grants_schema,
    "hockey": hockey_schema,
    "legalacts": legalacts_schema,
    "sap": sap_schema,
    "ssb": ssb_schema,
    "stats": stats_schema,
    "talkingdata": talkingdata_schema,
}

# 定义别名. 注意: key 被视为主要名称, 所有别名将连接到主要名称数据库.
aliases = {
    "tpch10g": ["tpch", "tpch1g"],
    "dsb10g": ["dsb"],
    "tpcds10g": ["tpcds"],
}
for name, anames in aliases.items():
    for db in (name, *anames):
        if db in schemas:
            schemas[name] = schemas[db]
            break
    for aname in anames:
        schemas[aname] = schemas[name]

__all__ = ["DatabaseSchema", "Index", "schemas", "parse_indexes_str", "ColumnCoder", "ColumnMetric"]
