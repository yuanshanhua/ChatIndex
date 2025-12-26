-- 生成删除所有非主键索引的SQL语句
SELECT
    format('DROP INDEX %I.%I;', schemaname, indexname)
FROM pg_indexes
WHERE schemaname NOT IN ('pg_catalog', 'information_schema')
  AND indexname NOT IN (
      SELECT conindid::regclass::text
      FROM pg_constraint
      WHERE contype = 'p'
  );

-- 查找所有没有主键的表
SELECT
    n.nspname  AS schema_name,
    c.relname  AS table_name
FROM pg_class c
JOIN pg_namespace n ON n.oid = c.relnamespace
LEFT JOIN pg_constraint con
       ON con.conrelid = c.oid
      AND con.contype = 'p'
WHERE c.relkind = 'r'          -- 普通表
  AND n.nspname NOT IN ('pg_catalog', 'information_schema')
  AND con.oid IS NULL
ORDER BY schema_name, table_name;
