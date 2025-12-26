import ast
import re
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import psycopg
import sqlparse

from .db import Connection, init_worker_connections, worker_connection
from .ia_logging import logger
from .schemas import DatabaseSchema
from .workload import Workload


log = logger.getChild("sql_features")


def _safe_rollback(connection: Connection) -> None:
    try:
        connection.rollback()
    except Exception as exc:  # pragma: no cover - defensive
        log.debug("rollback failed: %s", exc)


SQL_KEYWORDS = [
    "add",
    "all",
    "alter",
    "and",
    "any",
    "as",
    "asc",
    "autoincrement",
    "between",
    "boolean",
    "by",
    "call",
    "case",
    "cast",
    "char",
    "column",
    "commit",
    "constraint",
    "create",
    "cross",
    "current_date",
    "current_time",
    "current_timestamp",
    "database",
    "date",
    "default",
    "delete",
    "desc",
    "distinct",
    "drop",
    "else",
    "end",
    "exists",
    "extract",
    "false",
    "foreign",
    "from",
    "full",
    "function",
    "grant",
    "group",
    "having",
    "if",
    "in",
    "inner",
    "insert",
    "int",
    "integer",
    "intersect",
    "into",
    "is",
    "join",
    "key",
    "left",
    "like",
    "limit",
    "not",
    "null",
    "on",
    "or",
    "order",
    "outer",
    "primary",
    "procedure",
    "rename",
    "right",
    "rollback",
    "row",
    "select",
    "set",
    "show",
    "table",
    "then",
    "to",
    "truncate",
    "union",
    "update",
    "values",
    "view",
    "where",
    "with",
    "true",
    "unique",
    "alter",
    "table",
    "index",
    "view",
    "user",
    "load",
    "replace",
    "insert",
    "returning",
    "group_concat",
    "extract",
    "recursive",
    "isnull",
]
SQL_FUNCTIONS = [
    "sum",
    "count",
    "avg",
    "max",
    "min",
    "extract",
    "group_concat",
    "string_agg",
    "variance",
    "stddev",
    "median",
    "percentile_cont",
    "percentile_disc",
    "abs",
    "ceiling",
    "ceil",
    "floor",
    "round",
    "power",
    "exp",
    "extract",
    "log",
    "sqrt",
    "sin",
    "cos",
    "tan",
    "concat",
    "substring",
    "substr",
    "upper",
    "lower",
    "trim",
    "length",
    "replace",
    "lpad",
    "rpad",
    "now",
    "date_add",
    "date_sub",
    "date_part",
    "to_date",
    "to_char",
    "coalesce",
    "nullif",
    "case",
    "greatest",
    "least",
    "json_extract",
    "json_agg",
    "xmlagg",
    "st_distance",
    "st_intersects",
    "st_union",
    "row_number",
    "rank",
    "dense_rank",
    "lead",
    "lag",
    "ntile",
    "cast",
]
IDENTIFIER_PATTERN = r"\b[a-zA-Z_][a-zA-Z0-9_]*\b"
TABLE_DOT_COLUMN = r"\b[a-zA-Z_][a-zA-Z0-9_]*\.[a-zA-Z_][a-zA-Z0-9_]*\b"
TABLE_COLUMN_REGEX = re.compile(TABLE_DOT_COLUMN)

SimpleAlias = dict[str, tuple[str, ...]]
ComplexAlias = dict[str, str]


@dataclass(frozen=True)
class QueryPredicateStats:
    where_selectivities: dict[str, float] = field(default_factory=dict)
    join_predicates: dict[str, int] = field(default_factory=dict)
    group_order_columns: dict[str, int] = field(default_factory=dict)


@dataclass(frozen=True)
class WorkloadPredicateStats:
    workload_id: int
    db: str
    aggregate: QueryPredicateStats
    per_query: dict[str, QueryPredicateStats] = field(default_factory=dict)
    column_selectivities: dict[str, list[float]] = field(default_factory=dict)


@dataclass(frozen=True)
class ParsedSQLData:
    where_predicates: list[str]
    other_predicates: list[str]
    multi_where_predicates: list[str]
    group_order_columns: list[str]


@dataclass
class ParserContext:
    db_name: str
    schema_name: str
    schema: DatabaseSchema
    connection: Connection
    table_columns: dict[str, list[str]] = field(init=False)
    table_names: list[str] = field(init=False)
    column_names: list[str] = field(init=False)
    table_rows: dict[str, int] = field(init=False)
    view_to_columns: dict[str, list[str]] = field(init=False)
    view_columns: list[str] = field(init=False)

    def __post_init__(self) -> None:
        self.table_columns = {name: sorted(table.column_names) for name, table in self.schema.tables.items()}
        self.table_names = sorted(self.table_columns)
        self.column_names = sorted({col for cols in self.table_columns.values() for col in cols})
        self.table_rows = self._load_table_row_counts()
        self.view_to_columns, self.view_columns = self._load_view_info()

    def _load_table_row_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for table_name, table in self.schema.tables.items():
            try:
                metric = self.schema.column_statistic(self.connection, table_name, table.any_col, "row_count")
                counts[table_name] = int(metric) if metric is not None else 0
            except Exception as exc:  # pragma: no cover - defensive path
                log.debug("row count lookup failed for %s: %s", table_name, exc)
                _safe_rollback(self.connection)
                counts[table_name] = 0
        return counts

    def _load_view_info(self) -> tuple[dict[str, list[str]], list[str]]:
        query = (
            "SELECT table_name, column_name "
            "FROM information_schema.columns "
            "WHERE table_schema = %s "
            "AND table_name IN ("
            "    SELECT table_name FROM information_schema.views WHERE table_schema = %s"
            ") "
            "ORDER BY table_name, ordinal_position"
        )
        cursor = self.connection.cursor()
        view_to_columns: dict[str, list[str]] = {}
        try:
            cursor.execute(query, (self.schema_name, self.schema_name))
            for table_name, column_name in cursor.fetchall():
                view_to_columns.setdefault(table_name, []).append(column_name)
        except Exception as exc:  # pragma: no cover - metadata best-effort
            log.debug("failed to load view metadata: %s", exc)
            _safe_rollback(self.connection)
            return {}, []
        finally:
            cursor.close()
        columns = sorted({col for cols in view_to_columns.values() for col in cols})
        return view_to_columns, columns


def normalize_tokenlist(tokenlist: sqlparse.sql.TokenList) -> str:
    normalized_tl = ""
    for token in tokenlist:
        if token.ttype is not None:
            if "-" in token.value:
                if token.ttype == sqlparse.tokens.Token.Operator:
                    tmp_values = token.value.split("-")
                    normalized_tl += " - ".join(tmp for tmp in tmp_values)
                elif token.ttype == sqlparse.tokens.Token.Literal.String.Single:
                    normalized_tl += token.value + " "
            else:
                normalized_tl += token.value + " "
        else:
            if any(t is not None for t in token):
                normalized_tl += normalize_tokenlist(token) + " "
            else:
                normalized_tl += token.value + " "
    return normalized_tl


def normalize_sql(sql: str) -> str:
    parsed_sql = sqlparse.parse(sql)
    normalized_sql = ""
    for psql in parsed_sql:
        for token in psql:
            if token.ttype is not None:
                normalized_sql += token.value + " "
            else:
                normalized_sql += normalize_tokenlist(token) + " "
    normalized_sql = re.sub(r"\s+", " ", normalized_sql).strip().replace(" . ", ".")
    return normalized_sql


def is_subquery(token_list: sqlparse.sql.TokenList) -> bool:
    if isinstance(token_list, sqlparse.sql.TokenList):
        for token in token_list.tokens:
            if token.ttype is sqlparse.tokens.DML and token.value.upper() == "SELECT":
                return True
    return False


def extract_subqueries(token_list: sqlparse.sql.TokenList) -> list[sqlparse.sql.TokenList]:
    subqueries: list[sqlparse.sql.TokenList] = []
    if not hasattr(token_list, "tokens"):
        return subqueries

    for token in token_list.tokens:
        if isinstance(token, sqlparse.sql.TokenList):
            if is_subquery(token):
                subqueries.append(token)
            subqueries.extend(extract_subqueries(token))

    return subqueries


def is_literal_list(string: str) -> bool:
    try:
        result = ast.literal_eval(string)
        return isinstance(result, list)
    except (ValueError, SyntaxError):
        if (
            "," in string
            and all(keyword not in string for keyword in SQL_KEYWORDS)
            and all(func not in string for func in SQL_FUNCTIONS)
        ):
            return True
        return False


def extract_alias(sql: str, ctx: ParserContext) -> tuple[SimpleAlias, ComplexAlias]:
    alias: dict[str, Any] = {}
    simple_alias: SimpleAlias = {}
    complex_alias: ComplexAlias = {}
    table_names = list(ctx.table_columns.keys())
    column_names: list[str] = []
    for columns in ctx.table_columns.values():
        column_names.extend(columns)
    parsed_sql = sqlparse.parse(normalize_sql(sql))
    if len(parsed_sql) != 1:
        return simple_alias, complex_alias
    parsed = parsed_sql[0]
    aggregation = ""
    with_subquery = False
    keep_identifier = False
    rl = ""
    for token in parsed.tokens:
        if isinstance(token, sqlparse.sql.IdentifierList):
            if with_subquery:
                for identifier in token.get_identifiers():
                    if " as " in identifier.value.lower():
                        identifier_value = identifier.value
                        if " as " in identifier_value:
                            al = identifier_value.split(" as ")[0]
                            rl = identifier_value.replace(al + " as ", "").strip()
                        else:
                            al = identifier_value.split(" AS ")[0]
                            rl = identifier_value.replace(al + " AS ", "").strip()
                        pair = (rl, al)
                        subsql = rl[1:-1].strip()
                        s_sa, c_sa = extract_alias(subsql, ctx)
                        simple_alias.update(s_sa)
                        complex_alias.update(c_sa)
                        if pair and pair[1].lower() not in SQL_KEYWORDS:
                            alias[pair[1]] = pair[0]
                with_subquery = False
            else:
                for identifier in token.get_identifiers():
                    rl = ""
                    al = ""
                    if isinstance(identifier, sqlparse.sql.Identifier):
                        if identifier.get_alias():
                            al = identifier.value.split()[-1]
                            rl = " ".join(identifier.value.split()[:-1])
                            if rl[-2:].lower() == "as":
                                rl = rl[:-2].strip()
                        elif identifier.value in table_names or identifier.value in column_names:
                            keep_identifier = True
                            rl = identifier.value
                            continue
                        else:
                            continue
                        pair = (rl, al)
                        if rl.startswith("(") and rl.endswith(")") and "select" in rl.lower().split():
                            subsql = rl[1:-1].strip()
                            s_sa, c_sa = extract_alias(subsql, ctx)
                            simple_alias.update(s_sa)
                            complex_alias.update(c_sa)
                        if aggregation:
                            pair = (aggregation + pair[0], pair[1])
                            aggregation = ""
                        if pair and pair[1].lower() not in SQL_KEYWORDS:
                            alias[pair[1]] = pair[0]
                    elif isinstance(identifier, sqlparse.sql.Token):
                        if keep_identifier and identifier.value.strip() != "":
                            next_index = parsed.tokens.index(token)
                            if next_index + 1 > len(parsed.tokens) or (
                                next_index + 2 < len(parsed.tokens)
                                and parsed.tokens[next_index + 1].value != "("
                                and parsed.tokens[next_index + 2].value.lower() in SQL_KEYWORDS
                            ):
                                al = identifier.value
                                pair = (rl, al)
                                if aggregation:
                                    pair = (aggregation + pair[0], pair[1])
                                    aggregation = ""
                                if pair and pair[1].lower() not in SQL_KEYWORDS:
                                    alias[pair[1]] = pair[0]
                            elif identifier.value.lower() in SQL_FUNCTIONS:
                                aggregation = identifier.value
                            keep_identifier = False
                        elif identifier.value.lower() in SQL_FUNCTIONS:
                            aggregation = identifier.value
        elif isinstance(token, sqlparse.sql.Identifier):
            rl = ""
            al = ""
            if with_subquery:
                if " as " in token.value.lower():
                    if " as " in token.value:
                        al = token.value.split(" as ")[0]
                        rl = token.value.replace(al + " as ", "").strip()
                    else:
                        al = token.value.split(" AS ")[0]
                        rl = token.value.replace(al + " AS ", "").strip()
                with_subquery = False
            elif token.get_alias():
                al = token.value.split()[-1]
                rl = " ".join(token.value.split()[:-1])
                if rl[-2:].lower() == "as":
                    rl = rl[:-2].strip()
            elif token.value in table_names or token.value in column_names:
                keep_identifier = True
                rl = token.value
                continue
            else:
                continue
            pair = (rl, al)
            if rl.startswith("(") and rl.endswith(")") and "select" in rl.lower().split():
                subsql = rl[1:-1].strip()
                s_sa, c_sa = extract_alias(subsql, ctx)
                simple_alias.update(s_sa)
                complex_alias.update(c_sa)
            if aggregation:
                pair = (aggregation + pair[0], pair[1])
                aggregation = ""
            if pair and pair[1].lower() not in SQL_KEYWORDS:
                alias[pair[1]] = pair[0]
        elif isinstance(token, sqlparse.sql.Token):
            if token.value.lower() == "with":
                with_subquery = True
            elif token.value.lower() in SQL_FUNCTIONS:
                aggregation = token.value
            elif aggregation and token.value.lower() in SQL_KEYWORDS and token.value.lower() not in SQL_FUNCTIONS:
                aggregation = ""
            elif isinstance(token, sqlparse.sql.Where):
                if "(" in token.value and ")" in token.value:
                    for where_token in token:
                        if (
                            where_token.value.startswith("(")
                            and where_token.value.endswith(")")
                            and "select" in where_token.value.lower().split()
                        ):
                            subsql = where_token.value[1:-1].strip()
                            s_sa, c_sa = extract_alias(subsql, ctx)
                            simple_alias.update(s_sa)
                            complex_alias.update(c_sa)
            else:
                if keep_identifier and token.value.strip() != "":
                    idx = parsed.tokens.index(token)
                    if idx + 1 > len(parsed.tokens) or (
                        idx + 2 < len(parsed.tokens) and parsed.tokens[idx + 2].ttype is sqlparse.tokens.Keyword
                    ):
                        al = token.value
                        pair = (rl, al)
                        if aggregation:
                            pair = (aggregation + pair[0], pair[1])
                            aggregation = ""
                        if pair and pair[1].lower() not in SQL_KEYWORDS:
                            alias[pair[1]] = pair[0]
                    keep_identifier = False
    single_alias = {}
    for key, value in alias.items():
        if re.fullmatch(IDENTIFIER_PATTERN, value):
            single_alias[key] = value
    for key, value in list(alias.items()):
        if key not in single_alias:
            if re.fullmatch(TABLE_DOT_COLUMN, value):
                table, column = value.split(".")
                if table in single_alias:
                    table = single_alias[table]
                alias[key] = (table, column)
            else:
                tokens = value.split()
                expression = ""
                for token in tokens:
                    current = ""
                    if "." in token:
                        table_name = token.split(".")[0]
                        if table_name in single_alias:
                            table_name = single_alias[table_name]
                        col_name = token.split(".")[1]
                        if col_name in single_alias:
                            col_name = single_alias[col_name]
                        current = f"{table_name}.{col_name}"
                    elif token in single_alias:
                        current = single_alias[token]
                    else:
                        current = token
                    expression += current + " "
                alias[key] = expression
        else:
            if alias[key] in table_names:
                alias[key] = (alias[key],)
            else:
                resolved = False
                for table_name in table_names:
                    cols = ctx.table_columns[table_name]
                    if alias[key] in cols:
                        alias[key] = (table_name, alias[key])
                        resolved = True
                        break
                if not resolved:
                    log.debug("alias %s not resolved for value %s", key, alias[key])
                    alias[key] = ("", alias[key])
        if isinstance(alias[key], tuple):
            simple_alias[key] = alias[key]
        else:
            complex_alias[key] = alias[key]
    return simple_alias, complex_alias


def judge_identifier(tokens: sqlparse.sql.Token, ctx: ParserContext, alias: Mapping[str, Any]) -> bool:
    table_names = set(ctx.table_columns.keys())
    column_names = set(ctx.column_names)
    view_names = set(ctx.view_to_columns.keys())
    view_columns = set(ctx.view_columns)
    all_identifiers = column_names.union(table_names).union(alias.keys()).union(view_columns).union(view_names)
    for token in tokens.value.split():
        if re.fullmatch(IDENTIFIER_PATTERN, token):
            if token in all_identifiers:
                return True
        elif re.fullmatch(TABLE_DOT_COLUMN, token):
            return True
    if "( * )" in tokens.value:
        return True
    return False


def get_preidcate(
    original_predicate: str, simple_alias: SimpleAlias, used_tables: list[str], ctx: ParserContext
) -> str:
    rewrite_predicate = ""
    variables = original_predicate.split()
    for variable in variables:
        tmp = variable
        if "." in variable:
            tab, col = variable.split(".", 1)
            if col in simple_alias:
                alias_value = simple_alias[col]
                if len(alias_value) == 2:
                    tmp = f"{alias_value[0]}.{alias_value[1]}"
                elif tab in simple_alias:
                    tmp = f"{simple_alias[tab][0]}.{col}"
            elif tab in simple_alias:
                tmp = f"{simple_alias[tab][0]}.{col}"
        else:
            if variable in simple_alias:
                alias_value = simple_alias[variable]
                if len(alias_value) == 1:
                    tmp = alias_value[0]
                else:
                    tmp = f"{alias_value[0]}.{alias_value[1]}"
            else:
                found = False
                for table in used_tables:
                    if variable in ctx.table_columns.get(table, []):
                        tmp = f"{table}.{variable}"
                        found = True
                        break
                if not found:
                    for table, columns in ctx.table_columns.items():
                        if variable in columns:
                            tmp = f"{table}.{variable}"
                            break
        rewrite_predicate += tmp + " "
    return rewrite_predicate


def rewrite_expression(
    tokenlist: sqlparse.sql.Token,
    simple_alias: SimpleAlias,
    used_tables: list[str],
    ctx: ParserContext,
) -> str:
    original_predicate = tokenlist.value
    return get_preidcate(original_predicate, simple_alias, used_tables, ctx)


def get_predicate_str(
    tokens: sqlparse.sql.Comparison,
    aggregation_func: str,
    simple_alias: SimpleAlias,
    ctx: ParserContext,
    used_tables: list[str],
    alias: Mapping[str, Any],
) -> tuple[str, bool]:
    operator_token: sqlparse.sql.Token | None = None
    predicate = ""
    where_condition = True
    cnt = 0
    for token_part in tokens:
        if token_part.ttype == sqlparse.tokens.Token.Operator.Comparison or (
            token_part.ttype == sqlparse.tokens.Token.Keyword
            and token_part.value.lower() in ["like", "in", "exists", "between"]
        ):
            operator_token = token_part
            break
    if "select" in tokens.left.value.lower() or "select" in tokens.right.value.lower():
        return "", False
    if judge_identifier(tokens.left, ctx, alias):
        cnt += 1
    if judge_identifier(tokens.right, ctx, alias):
        cnt += 1
    if cnt == 2:
        where_condition = False
    elif cnt != 1:
        return tokens.value, False
    if operator_token is None:
        return tokens.value, False
    left = rewrite_expression(tokens.left, simple_alias, used_tables, ctx)
    right = rewrite_expression(tokens.right, simple_alias, used_tables, ctx)
    predicate = f"{aggregation_func}{left} {operator_token.value} {right}"
    return predicate, where_condition


def find_table_info(predicate: str, ctx: ParserContext) -> list[str]:
    tables: list[str] = []
    for item in predicate.split(" "):
        token = item.strip()
        if not token:
            continue
        if re.fullmatch(TABLE_DOT_COLUMN, token):
            table = token.split(".", 1)[0]
            if table in ctx.table_columns and table not in tables:
                tables.append(table)
        elif re.fullmatch(IDENTIFIER_PATTERN, token):
            found = False
            for table, columns in ctx.table_columns.items():
                if token in columns:
                    if table not in tables:
                        tables.append(table)
                    found = True
                    break
            if not found:
                for view_name, columns in ctx.view_to_columns.items():
                    if token in columns and view_name not in tables:
                        tables.append(view_name)
                        break
    return tables


def _update_column_selectivities(target: dict[str, list[float]], where_selectivities: Mapping[str, float]) -> None:
    for predicate, selectivity in where_selectivities.items():
        matches = TABLE_COLUMN_REGEX.findall(predicate)
        if not matches:
            continue
        for match in matches:
            target.setdefault(match, []).append(selectivity)


def parse_sql(sql: str, ctx: ParserContext) -> tuple[ParsedSQLData, SimpleAlias, ComplexAlias]:
    parsed_sql = sqlparse.parse(normalize_sql(sql))
    where_predicates: list[str] = []
    multi_where_predicates: list[str] = []
    other_predicates: list[str] = []
    group_order_columns: list[str] = []
    aggregation_func = ""
    aggregation_bool = False
    multi_where_predicate = ""
    is_and = False
    is_or = False
    is_from = False
    is_between = False
    group_order = False
    column_names: list[str] = []
    for columns in ctx.table_columns.values():
        column_names.extend(columns)
    used_tables: list[str] = []
    simple_alias, complex_alias = extract_alias(sql, ctx)
    alias: dict[str, Any] = {**simple_alias, **complex_alias}
    if len(parsed_sql) != 1:
        return ParsedSQLData([], [], [], []), simple_alias, complex_alias
    parsed = parsed_sql[0]
    if "intersect" in parsed.value.lower().split():
        if "intersect" in parsed.value:
            sqls = parsed.value.split(" intersect ")
        else:
            sqls = parsed.value.split(" INTERSECT ")
        for part in sqls:
            info, sa_sub, ca_sub = parse_sql(part, ctx)
            where_predicates.extend(info.where_predicates)
            other_predicates.extend(info.other_predicates)
            multi_where_predicates.extend(info.multi_where_predicates)
            group_order_columns.extend(info.group_order_columns)
            simple_alias.update(sa_sub)
            complex_alias.update(ca_sub)
        return (
            ParsedSQLData(
                where_predicates,
                other_predicates,
                multi_where_predicates,
                group_order_columns,
            ),
            simple_alias,
            complex_alias,
        )
    for token in parsed.tokens:
        predicate = ""
        subquery = extract_subqueries(token)
        if subquery:
            for sub in subquery:
                sub_sql = sub.value.strip()
                if sub_sql.startswith("(") and sub_sql.endswith(")") and len(sub_sql) >= 2:
                    sub_sql = sub_sql[1:-1].strip()
                info, sa_sub, ca_sub = parse_sql(sub_sql, ctx)
                where_predicates.extend(info.where_predicates)
                other_predicates.extend(info.other_predicates)
                multi_where_predicates.extend(info.multi_where_predicates)
                group_order_columns.extend(info.group_order_columns)
                simple_alias.update(sa_sub)
                complex_alias.update(ca_sub)
                alias = {**simple_alias, **complex_alias}
        if token.ttype is sqlparse.tokens.Keyword:
            if token.value.lower() in SQL_FUNCTIONS:
                aggregation_func = token.value
                aggregation_bool = True
            elif token.value.lower() in ["group by", "order by"]:
                group_order = True
            elif token.value.lower() == "from" or "join" in token.value.lower():
                is_from = True
            continue
        elif isinstance(token, sqlparse.sql.Comparison):
            predicate, where = get_predicate_str(token, aggregation_func, simple_alias, ctx, used_tables, alias)
            if aggregation_func != "":
                aggregation_func = ""
            if predicate:
                if where:
                    where_predicates.append(predicate)
                else:
                    other_predicates.append(predicate)
        elif isinstance(token, sqlparse.sql.Where):
            aggregation_func = ""
            where_predicate_str = ""
            where_condition = False
            for where_token in token:
                if where_token.ttype is sqlparse.tokens.Keyword and where_token.value.lower() in SQL_FUNCTIONS:
                    aggregation_func = where_token.value
                    continue
                elif isinstance(where_token, sqlparse.sql.Comparison):
                    predicate, where_condition = get_predicate_str(
                        where_token,
                        aggregation_func,
                        simple_alias,
                        ctx,
                        used_tables,
                        alias,
                    )
                    if aggregation_func != "":
                        aggregation_func = ""
                elif isinstance(where_token, sqlparse.sql.Token):
                    if where_token.value.lower() == "where":
                        continue
                    elif where_token.value.lower() in ["and", "or"] or where_token == token[-1]:
                        if where_token.value.lower() in ["and", "or"] and (
                            "between" not in where_predicate_str.lower() or is_between
                        ):
                            if where_token.value.lower() == "and":
                                is_and = True
                            elif where_token.value.lower() == "or":
                                is_or = True
                        else:
                            if "between" in where_predicate_str.lower():
                                is_between = True
                            if not (where_token.value.lower() == "and" and "and" in where_predicate_str.lower()):
                                where_predicate_str += where_token.value
                            if where_token != token[-1]:
                                continue
                            if where_token == token[-1] and where_token.value == ";":
                                where_predicate_str = where_predicate_str
                        if where_predicate_str.strip() != "":
                            if where_predicate_str.lower().strip().startswith(("and", "or")):
                                where_predicate_str = where_predicate_str[3:]
                            where_predicate_str = get_preidcate(
                                where_predicate_str,
                                simple_alias,
                                used_tables,
                                ctx,
                            )
                            if where_predicate_str:
                                where_predicates.append(where_predicate_str)
                                if multi_where_predicate == "":
                                    multi_where_predicate = where_predicate_str
                                elif is_and and where_predicate_str not in multi_where_predicate:
                                    multi_where_predicate += " and " + where_predicate_str
                                elif is_or and where_predicate_str not in multi_where_predicate:
                                    multi_where_predicate += " or " + where_predicate_str
                                is_or = False
                                is_and = False
                                if "between" in where_predicate_str.lower():
                                    is_between = False
                                where_predicate_str = ""
                            continue
                    elif isinstance(where_token, sqlparse.sql.Parenthesis):
                        if (
                            where_token.value.strip().startswith("(")
                            and where_token.value.strip().endswith(")")
                            and "select" not in where_token.value.lower()
                        ):
                            if not aggregation_func:
                                predicates_ = where_token.value[1:-1].strip()
                            else:
                                predicates_ = aggregation_func + where_token.value
                            if is_literal_list("[" + predicates_ + "]") or any(
                                key not in predicates_.lower() for key in ["and", "or", "select"]
                            ):
                                if is_literal_list("[" + predicates_ + "]"):
                                    where_predicate_str += where_token.value
                                else:
                                    where_predicate_str += predicates_
                            else:
                                fake_sql = f"SELECT * FROM table WHERE {predicates_}"
                                sub_info, _, _ = parse_sql(fake_sql, ctx)
                                where_predicates = list(set(where_predicates).union(set(sub_info.where_predicates)))
                                other_predicates = list(set(other_predicates).union(set(sub_info.other_predicates)))
                                multi_where_predicates = list(
                                    set(multi_where_predicates).union(set(sub_info.multi_where_predicates))
                                )
                                group_order_columns = list(
                                    set(group_order_columns).union(set(sub_info.group_order_columns))
                                )
                            aggregation_func = ""
                        elif (
                            where_token.value.strip().startswith("(")
                            and where_token.value.strip().endswith(")")
                            and "select" in where_token.value.lower()
                        ):
                            where_predicate_str = ""
                            continue
                        else:
                            where_predicate_str += where_token.value
                    else:
                        if where_predicate_str.strip() == "" and where_token.value.lower() in ["and", "or"]:
                            continue
                        elif isinstance(where_token, sqlparse.sql.Operation):
                            if aggregation_func != "":
                                where_predicate_str += aggregation_func
                                aggregation_func = ""
                            where_predicate_str += where_token.value
                            continue
                        elif isinstance(where_token, sqlparse.sql.Identifier) and ")" in where_token.value:
                            where_predicate_str += where_token.value.split(")")[0]
                            continue
                        where_predicate_str += where_token.value
                if predicate:
                    if where_condition:
                        where_predicates.append(predicate)
                        if multi_where_predicate == "":
                            multi_where_predicate = predicate
                        elif is_and and predicate not in multi_where_predicate:
                            multi_where_predicate += " and " + predicate
                        elif is_or and predicate not in multi_where_predicate:
                            multi_where_predicate += " or " + predicate
                        is_or = False
                        is_and = False
                    else:
                        other_predicates.append(predicate)
                    predicate = ""
        elif isinstance(token, sqlparse.sql.Identifier):
            if is_from:
                for table in ctx.table_columns.keys():
                    if table in token.value and table not in used_tables:
                        used_tables.append(table)
                is_from = False
            elif group_order:
                tv = token.value
                if "desc" in token.value.lower():
                    tv = tv.replace("desc", " ").replace("DESC", " ").strip()
                elif "asc" in token.value.lower():
                    tv = tv.replace("asc", " ").replace("ASC", " ").strip()
                original_value = ""
                if tv in column_names:
                    original_value = tv
                elif len(token.value.strip().split()) == 1 and "." in token.value:
                    original_value = token.value.split(".")[-1]
                else:
                    if tv in alias:
                        temp = alias[tv]
                        if isinstance(temp, tuple) and len(temp) == 2:
                            original_value = temp[1]
                        elif isinstance(temp, str):
                            for cn in column_names:
                                if cn in temp:
                                    original_value = cn
                        else:
                            log.debug("unexpected alias form for %s: %s", tv, temp)
                    aggregation_func = ""
                if original_value:
                    for table_name, cols in ctx.table_columns.items():
                        if original_value in cols:
                            group_order_columns.append(f"{table_name}.{original_value}")
                group_order = False
            elif token.value.lower().startswith("case when "):
                tmp_predicate = token.value.lower().split("case when ")[-1].split("then")[0].strip()
                fake_sql = f"SELECT * FROM table WHERE {tmp_predicate} ;"
                sub_info, _, _ = parse_sql(fake_sql, ctx)
                where_predicates = list(set(where_predicates).union(set(sub_info.where_predicates)))
                other_predicates = list(set(other_predicates).union(set(sub_info.other_predicates)))
                multi_where_predicates = list(set(multi_where_predicates).union(set(sub_info.multi_where_predicates)))
                group_order_columns = list(set(group_order_columns).union(set(sub_info.group_order_columns)))
        elif isinstance(token, sqlparse.sql.IdentifierList):
            if is_from:
                for table in ctx.table_columns.keys():
                    if table in token.value and table not in used_tables:
                        used_tables.append(table)
                is_from = False
            if group_order:
                for identifier in token:
                    if re.fullmatch(IDENTIFIER_PATTERN, identifier.value) or re.fullmatch(
                        TABLE_DOT_COLUMN, identifier.value
                    ):
                        original_value = ""
                        if identifier.value.lower() in SQL_FUNCTIONS:
                            aggregation_func = identifier.value
                            continue
                        elif identifier.value in column_names:
                            original_value = identifier.value
                        elif len(identifier.value.strip().split()) == 1 and "." in identifier.value:
                            original_value = identifier.value.split(".")[-1]
                        else:
                            tv = (aggregation_func + " " + identifier.value).strip()
                            if tv in alias:
                                temp = alias[tv]
                                if isinstance(temp, tuple) and len(temp) == 2:
                                    original_value = temp[1]
                                elif isinstance(temp, str):
                                    for cn in column_names:
                                        if cn in temp:
                                            original_value = cn
                                else:
                                    log.debug("unexpected alias for %s: %s", tv, temp)
                            else:
                                log.debug("alias %s not found", tv)
                            aggregation_func = ""
                        if original_value:
                            for table_name, cols in ctx.table_columns.items():
                                if original_value in cols:
                                    group_order_columns.append(f"{table_name}.{original_value}")
                group_order = False
        if aggregation_bool and aggregation_func:
            aggregation_bool = False
            aggregation_func = ""
    if multi_where_predicate:
        if multi_where_predicate.startswith(" and "):
            multi_where_predicate = multi_where_predicate[4:].strip()
        elif multi_where_predicate.startswith(" or "):
            multi_where_predicate = multi_where_predicate[3:].strip()
        multi_where_predicates.append(multi_where_predicate)
    return (
        ParsedSQLData(where_predicates, other_predicates, multi_where_predicates, group_order_columns),
        simple_alias,
        complex_alias,
    )


class SQLPredicateAnalyzer:
    def __init__(self, ctx: ParserContext) -> None:
        self.ctx = ctx

    def analyze_query(self, sql: str) -> QueryPredicateStats:
        parsed, simple_alias, _ = parse_sql(sql, self.ctx)
        unique_where = list(set(parsed.where_predicates))
        where_selectivities = self._estimate_where_selectivities(unique_where)
        join_predicates = self._count_join_predicates(parsed.other_predicates, simple_alias)
        group_order_columns = self._count_group_order_columns(parsed.group_order_columns)
        return QueryPredicateStats(where_selectivities, join_predicates, group_order_columns)

    def _qualify_table(self, table: str) -> str:
        if "." in table:
            return table
        return f"{self.ctx.schema_name}.{table}"

    def _build_cardinality_sql(self, tables: Sequence[str], predicate: str) -> str:
        tbls = ", ".join(self._qualify_table(table) for table in tables)
        return f"EXPLAIN (FORMAT JSON) SELECT * FROM {tbls} WHERE {predicate};"

    def _build_extrema_cardinality_sql(self, tables: Sequence[str], predicate: str) -> str | None:
        lower = predicate.lower()
        if "max" not in lower and "min" not in lower:
            return None
        agg = "max" if "max" in lower else "min"
        agg_idx = lower.find(agg)
        tmp_predicate = predicate[agg_idx + len(agg) :]
        if not tmp_predicate.strip().startswith("("):
            return None
        cnt = 0
        collected = agg
        for token in tmp_predicate:
            if token == "(":
                cnt += 1
            elif token == ")":
                cnt -= 1
            collected += token
            if cnt == 0:
                break
        column_info = collected.split("(")[-1].split(")")[0]
        tbls = ", ".join(self._qualify_table(table) for table in tables)
        sub_tbls = tbls
        return (
            f"EXPLAIN (FORMAT JSON) SELECT * FROM {tbls} WHERE {column_info} = (SELECT {collected} FROM {sub_tbls});"
        )

    def _execute_explain(self, sql: str) -> list[Any] | None:
        if not isinstance(self.ctx.connection, psycopg.Connection):
            log.warning("selectivity estimation currently supports PostgreSQL connections only")
            return None
        cursor = self.ctx.connection.cursor()
        try:
            cursor.execute(sql)  # type: ignore[arg-type]
            return cursor.fetchall()
        except Exception as exc:
            log.debug("EXPLAIN failed for %s: %s", sql, exc)
            _safe_rollback(self.ctx.connection)
            return None
        finally:
            cursor.close()

    def _interpret_explain_result(self, result: list[Any] | None, tables: Sequence[str]) -> float | None:
        if not result:
            return None
        try:
            plan = result[0][0][0]["Plan"]
            card = plan["Plan Rows"]
        except (KeyError, IndexError, TypeError) as exc:
            log.debug("unexpected EXPLAIN output: %s", exc)
            return None
        total_rows = 0
        for table in tables:
            canonical = table.split(".")[-1]
            total_rows += self.ctx.table_rows.get(canonical, 0)
        if total_rows == 0:
            return None
        selectivity = card / total_rows
        if selectivity >= 1:
            return None
        return selectivity

    def _estimate_where_selectivities(self, predicates: Sequence[str]) -> dict[str, float]:
        selectivities: dict[str, float] = {}
        for predicate in predicates:
            tables = find_table_info(predicate, self.ctx)
            if not tables:
                continue
            sql = self._build_cardinality_sql(tables, predicate)
            result = self._execute_explain(sql)
            selectivity = self._interpret_explain_result(result, tables)
            if selectivity is not None:
                selectivities[predicate] = selectivity
                continue
            alt_sql = self._build_extrema_cardinality_sql(tables, predicate)
            if alt_sql:
                alt_result = self._execute_explain(alt_sql)
                alt_selectivity = self._interpret_explain_result(alt_result, tables)
                if alt_selectivity is not None:
                    selectivities[predicate] = alt_selectivity
        return selectivities

    def _resolve_alias_reference(self, token: str, simple_alias: SimpleAlias) -> tuple[str, str] | None:
        table, column = token.split(".", 1)
        if table in simple_alias:
            alias_value = simple_alias[table]
            if len(alias_value) == 2:
                table, column = alias_value
            elif len(alias_value) == 1:
                table = alias_value[0]
        if column in simple_alias:
            alias_value = simple_alias[column]
            if len(alias_value) == 2:
                table, column = alias_value
            elif len(alias_value) == 1:
                table = alias_value[0]
        if column in self.ctx.column_names and table not in self.ctx.table_columns:
            for real_table, columns in self.ctx.table_columns.items():
                if column in columns:
                    table = real_table
                    break
        if column not in self.ctx.column_names or table not in self.ctx.table_columns:
            return None
        return table, column

    def _count_join_predicates(self, predicates: Sequence[str], simple_alias: SimpleAlias) -> dict[str, int]:
        counts: dict[str, int] = {}
        for predicate in predicates:
            for token in predicate.split():
                token = token.strip()
                if re.fullmatch(TABLE_DOT_COLUMN, token):
                    resolved = self._resolve_alias_reference(token, simple_alias)
                    if resolved is None:
                        continue
                    table, column = resolved
                    key = f"{table}.{column}"
                    counts[key] = counts.get(key, 0) + 1
        return counts

    def _count_group_order_columns(self, columns: Sequence[str]) -> dict[str, int]:
        counts: dict[str, int] = {}
        for column in columns:
            column = column.strip()
            if not column:
                continue
            counts[column] = counts.get(column, 0) + 1
        return counts


def _resolve_schema_name(schema_names: Mapping[str, str] | str, db: str) -> str:
    if isinstance(schema_names, str):
        return schema_names
    return schema_names.get(db, "public")


_worker_db_schemas: Mapping[str, DatabaseSchema] | None = None
_worker_schema_names: Mapping[str, str] | str | None = None
_worker_analyzers: dict[tuple[str, str], SQLPredicateAnalyzer] = {}


def _get_or_create_worker_analyzer(db_name: str) -> SQLPredicateAnalyzer:
    """Lazily create analyzers in worker processes for multiprocessing runs."""

    if _worker_db_schemas is None or _worker_schema_names is None:
        raise RuntimeError("predicate stats worker state not initialized")
    if db_name not in _worker_db_schemas:
        raise ValueError(f"Missing schema definition for database {db_name}")

    schema_name = _resolve_schema_name(_worker_schema_names, db_name)
    analyzer_key = (db_name, schema_name)
    analyzer = _worker_analyzers.get(analyzer_key)
    if analyzer is None:
        ctx = ParserContext(db_name, schema_name, _worker_db_schemas[db_name], worker_connection(db_name))
        analyzer = SQLPredicateAnalyzer(ctx)
        _worker_analyzers[analyzer_key] = analyzer
    return analyzer


def init_worker_state(
    opt_dict: Mapping[str, Any],
    db_schemas: Mapping[str, DatabaseSchema],
    schema_names: Mapping[str, str] | str,
) -> None:
    global _worker_db_schemas, _worker_schema_names, _worker_analyzers
    init_worker_connections(opt_dict)
    _worker_db_schemas = db_schemas
    _worker_schema_names = schema_names
    _worker_analyzers = {}


def compute_sql_predicate_task(task: tuple[str, str]) -> tuple[str, str, QueryPredicateStats]:
    """Worker task to compute predicate stats for a single SQL statement."""

    if _worker_db_schemas is None or _worker_schema_names is None:
        raise RuntimeError("predicate stats worker state not initialized")
    db_name, sql = task
    try:
        analyzer = _get_or_create_worker_analyzer(db_name)
        return db_name, sql, analyzer.analyze_query(sql)
    except Exception as exc:
        raise RuntimeError(f"SQL 解析失败: db={db_name}, sql={sql}") from exc


def aggregate_workload_stats(
    workload: Workload,
    per_query_stats: Mapping[str, QueryPredicateStats],
) -> WorkloadPredicateStats:
    """Aggregate predicate stats for a workload using precomputed query-level stats."""

    per_query: dict[str, QueryPredicateStats] = {}
    aggregate_where: dict[str, float] = {}
    aggregate_join: dict[str, int] = {}
    aggregate_group: dict[str, int] = {}
    column_selectivities: dict[str, list[float]] = {}

    for query in sorted(workload.queries):
        stats = per_query_stats.get(query.sql)
        if stats is None:
            raise ValueError(f"Missing predicate stats for SQL: {query.sql}")
        per_query[query.sql] = stats
        for predicate, value in stats.where_selectivities.items():
            if predicate not in aggregate_where:
                aggregate_where[predicate] = value
        for key, value in stats.join_predicates.items():
            aggregate_join[key] = aggregate_join.get(key, 0) + value
        for key, value in stats.group_order_columns.items():
            aggregate_group[key] = aggregate_group.get(key, 0) + value
        _update_column_selectivities(column_selectivities, stats.where_selectivities)

    aggregate = QueryPredicateStats(aggregate_where, aggregate_join, aggregate_group)
    return WorkloadPredicateStats(workload.id, workload.db, aggregate, per_query, column_selectivities)
