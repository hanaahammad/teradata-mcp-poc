# MCP Tools Reference
## Teradata Data Engineering MCP Server

All 15 tools exposed by `mcp/server.py` — usable via Claude Desktop natural language.

---

### Querying

#### `run_sql`
Execute any Teradata SQL query.
```
"Run this SQL: SELECT TOP 10 * FROM demo_user.transactions"
```
| Parameter | Type | Required | Description |
|---|---|---|---|
| query | string | ✅ | SQL to execute |
| database | string | | Target database |
| max_rows | integer | | Cap on rows returned (default 500) |

---

### Schema Discovery

#### `list_databases`
List all databases the user has access to with size info.
```
"What databases do I have access to?"
```

#### `list_tables`
List all tables and views in a database.
```
"Show me all tables in demo_user"
```

#### `get_schema`
Get column definitions from DBC.ColumnsV.
```
"Show me the schema of the transactions table"
```

#### `get_table_ddl`
Get the full CREATE TABLE DDL statement.
```
"Show me the DDL for customer_demographics"
```

---

### Data Profiling

#### `profile_table`
Row count, null %, distinct values, min/max/avg per column, table size.
```
"Profile the transactions table"
"Are there any data quality issues in credit_risk?"
```

#### `check_duplicates`
Find duplicate rows across specified key columns.
```
"Check for duplicate customer_ids in customer_demographics"
```

#### `get_table_stats`
Show COLLECT STATISTICS metadata from DBC.StatsV.
```
"Have stats been collected on the segmentation_dataset table?"
```

---

### Performance & Skew

#### `check_data_skew`
Analyze data distribution across AMPs. Skew > 10% = performance risk.
```
"Is the transactions table skewed?"
```
Returns: AMP distribution, skew factor, severity (LOW / MEDIUM / HIGH)

#### `get_pi_info`
Show the Primary Index definition from DBC.IndicesV.
```
"What is the Primary Index on customer_features_rfm?"
"Is the PI a good choice for this table?"
```

---

### Lineage & Dependencies

#### `find_table_references`
Find all views and macros that reference a given table.
```
"What objects depend on the transactions table?"
```

---

### Export

#### `export_to_csv`
Run a SQL query and save results to a local CSV file.
```
"Export all customers with credit_score < 500 to /tmp/high_risk.csv"
```

---

### Space Management

#### `get_space_usage`
Current and max perm space usage per database.
```
"How much space is demo_user using?"
"Show me space usage across all databases"
```

---

### In-Database ML (KMeans)

#### `run_kmeans_experiment`
Loop k=2..N, run `val.tda_kmeans` for each k, collect Within-SS (elbow data).
```
"Run a KMeans elbow experiment on segmentation_rfm_scaled 
 using monetary_scaled, frequency_scaled, recency_scaled for k=2 to 8"
```
| Parameter | Type | Required | Description |
|---|---|---|---|
| database | string | ✅ | Teradata database |
| table_name | string | ✅ | Scaled input table |
| id_column | string | ✅ | ID column (e.g. customer_id) |
| feature_columns | array | ✅ | Scaled feature column names |
| k_min | integer | | Min k (default 2) |
| k_max | integer | | Max k (default 10) |
| max_iterations | integer | | KMeans iterations (default 100) |

Returns: per-k Within-SS, Between-SS, cluster sizes, convergence info, delta (elbow indicator)

#### `run_kmeans_final`
Run final KMeans with chosen k and save model centroids to a Teradata table.
```
"Run final KMeans with k=4 on segmentation_rfm_scaled 
 and save the model to kmeans_model_rfm_final"
```

---

### Built-in Prompts

#### `td-dq-check`
Full data quality audit: schema → profile → duplicates → skew → stats → quality score.
```
"Run a full data quality check on demo_user.transactions"
```

#### `td-pi-review`
Primary Index analysis and recommendation with DDL.
```
"Review the Primary Index on demo_user.customer_features_rfm"
```

---

### Resources

The server also exposes two reference resources Claude can read:
- `teradata://dbc-reference` — DBC system views cheat sheet
- `teradata://best-practices` — PI selection, stats, skew, performance tips
