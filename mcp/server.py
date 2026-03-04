# server.py — Teradata Data Engineering MCP Server
import json
import csv
import io
import os
from pathlib import Path
from typing import Optional

import yaml
import teradatasql
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp import types

app = Server("teradata-data-engineering-mcp")

# ─── LOAD CONFIG ─────────────────────────────────────────────
def load_config():
    """Load config.yaml from project root (two levels up from mcp/server.py)."""
    config_path = Path(__file__).parent.parent / "config.yaml"
    if config_path.exists():
        with open(config_path) as f:
            return yaml.safe_load(f)
    # Fallback to environment variables
    return {
        "teradata": {
            "host":     os.environ.get("TD_HOST", ""),
            "user":     os.environ.get("TD_USER", ""),
            "password": os.environ.get("TD_PASSWORD", ""),
            "database": os.environ.get("TD_DATABASE", ""),
        },
        "mcp": {"max_rows": 500, "max_iterations": 100, "k_min": 2, "k_max": 10}
    }

CONFIG   = load_config()
TD_CONF  = CONFIG.get("teradata", {})
MCP_CONF = CONFIG.get("mcp", {})

TD_CONFIG = {k: v for k, v in {
    "host":     TD_CONF.get("host", ""),
    "user":     TD_CONF.get("user", ""),
    "password": TD_CONF.get("password", ""),
    "database": TD_CONF.get("database", ""),
}.items() if v}  # strip empty values

def get_connection(database: Optional[str] = None):
    """Create a Teradata connection."""
    config = {k: v for k, v in TD_CONFIG.items() if v}  # remove empty values
    if database:
        config["database"] = database
    return teradatasql.connect(**config)


# ─── TOOLS ────────────────────────────────────────────────────

@app.list_tools()
async def list_tools():
    return [

        types.Tool(
            name="run_sql",
            description="Execute a Teradata SQL query and return results. Results capped at 500 rows.",
            inputSchema={
                "type": "object",
                "properties": {
                    "query":    {"type": "string", "description": "SQL query to execute"},
                    "database": {"type": "string", "description": "Target Teradata database (optional)"},
                    "max_rows": {"type": "integer", "description": "Max rows to return (default 500)"}
                },
                "required": ["query"]
            }
        ),

        types.Tool(
            name="list_databases",
            description="List all Teradata databases the user has access to, with size info.",
            inputSchema={"type": "object", "properties": {}}
        ),

        types.Tool(
            name="list_tables",
            description="List all tables and views in a Teradata database with sizes.",
            inputSchema={
                "type": "object",
                "properties": {
                    "database": {"type": "string", "description": "Teradata database name"}
                },
                "required": ["database"]
            }
        ),

        types.Tool(
            name="get_table_ddl",
            description="Get the full DDL (CREATE TABLE statement) for a Teradata table.",
            inputSchema={
                "type": "object",
                "properties": {
                    "database":   {"type": "string"},
                    "table_name": {"type": "string"}
                },
                "required": ["database", "table_name"]
            }
        ),

        types.Tool(
            name="get_schema",
            description="Get column definitions for a Teradata table from DBC.Columns.",
            inputSchema={
                "type": "object",
                "properties": {
                    "database":   {"type": "string"},
                    "table_name": {"type": "string"}
                },
                "required": ["database", "table_name"]
            }
        ),

        types.Tool(
            name="profile_table",
            description="Profile a Teradata table: row count, nulls, distinct values, min/max/avg for numerics, size in MB.",
            inputSchema={
                "type": "object",
                "properties": {
                    "database":   {"type": "string"},
                    "table_name": {"type": "string"}
                },
                "required": ["database", "table_name"]
            }
        ),

        types.Tool(
            name="check_duplicates",
            description="Check for duplicate rows in a Teradata table across specified key columns.",
            inputSchema={
                "type": "object",
                "properties": {
                    "database":    {"type": "string"},
                    "table_name":  {"type": "string"},
                    "key_columns": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Columns to check uniqueness on"
                    }
                },
                "required": ["database", "table_name", "key_columns"]
            }
        ),

        types.Tool(
            name="get_table_stats",
            description="Get Teradata optimizer statistics for a table.",
            inputSchema={
                "type": "object",
                "properties": {
                    "database":   {"type": "string"},
                    "table_name": {"type": "string"}
                },
                "required": ["database", "table_name"]
            }
        ),

        types.Tool(
            name="check_data_skew",
            description="Analyze data distribution skew across AMPs for a Teradata table. Skew > 10% hurts performance.",
            inputSchema={
                "type": "object",
                "properties": {
                    "database":   {"type": "string"},
                    "table_name": {"type": "string"}
                },
                "required": ["database", "table_name"]
            }
        ),

        types.Tool(
            name="get_pi_info",
            description="Get the Primary Index (PI) definition for a table — critical for Teradata performance.",
            inputSchema={
                "type": "object",
                "properties": {
                    "database":   {"type": "string"},
                    "table_name": {"type": "string"}
                },
                "required": ["database", "table_name"]
            }
        ),

        types.Tool(
            name="find_table_references",
            description="Search for all views and macros that reference a given table.",
            inputSchema={
                "type": "object",
                "properties": {
                    "database":   {"type": "string"},
                    "table_name": {"type": "string"}
                },
                "required": ["database", "table_name"]
            }
        ),

        types.Tool(
            name="export_to_csv",
            description="Run a SQL query and save results to a local CSV file.",
            inputSchema={
                "type": "object",
                "properties": {
                    "query":       {"type": "string"},
                    "output_path": {"type": "string"},
                    "database":    {"type": "string"}
                },
                "required": ["query", "output_path"]
            }
        ),

        types.Tool(
            name="get_space_usage",
            description="Get current and max perm space usage for Teradata databases.",
            inputSchema={
                "type": "object",
                "properties": {
                    "database": {"type": "string", "description": "Filter to a specific DB (optional)"}
                }
            }
        ),

        types.Tool(
            name="run_kmeans_experiment",
            description="""Run KMeans clustering experiment for k=2..N using VAL tda_kmeans.
For each k, captures: Within-SS (elbow), cluster sizes, convergence info.
Returns a table of results to identify the optimal number of clusters.
Use this to find the elbow point before committing to a final k.""",
            inputSchema={
                "type": "object",
                "properties": {
                    "database":      {"type": "string", "description": "Teradata database"},
                    "table_name":    {"type": "string", "description": "Scaled input table for clustering"},
                    "id_column":     {"type": "string", "description": "ID column name (e.g. customer_id)"},
                    "feature_columns": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of scaled feature column names to cluster on"
                    },
                    "k_min":         {"type": "integer", "description": "Minimum k to try (default 2)"},
                    "k_max":         {"type": "integer", "description": "Maximum k to try (default 10)"},
                    "max_iterations":{"type": "integer", "description": "Max KMeans iterations (default 100)"}
                },
                "required": ["database", "table_name", "id_column", "feature_columns"]
            }
        ),

        types.Tool(
            name="run_kmeans_final",
            description="""Run final KMeans with a chosen k and save model + assignments to tables.
Use after run_kmeans_experiment to lock in the optimal k.""",
            inputSchema={
                "type": "object",
                "properties": {
                    "database":        {"type": "string"},
                    "table_name":      {"type": "string", "description": "Scaled input table"},
                    "id_column":       {"type": "string"},
                    "feature_columns": {"type": "array", "items": {"type": "string"}},
                    "n_clusters":      {"type": "integer", "description": "Final number of clusters"},
                    "model_table":     {"type": "string", "description": "Table to save model centroids"},
                    "max_iterations":  {"type": "integer", "description": "Max iterations (default 100)"}
                },
                "required": ["database", "table_name", "id_column", "feature_columns", "n_clusters", "model_table"]
            }
        ),
    ]


# ─── TOOL HANDLERS ────────────────────────────────────────────

@app.call_tool()
async def call_tool(name: str, arguments: dict):

    # ── run_sql ───────────────────────────────────────────────
    if name == "run_sql":
        try:
            max_rows = arguments.get("max_rows", 500)
            with get_connection(arguments.get("database")) as conn:
                with conn.cursor() as cur:
                    cur.execute(arguments["query"])
                    if cur.description:
                        cols = [d[0] for d in cur.description]
                        rows = cur.fetchmany(max_rows)
                        result = {
                            "columns": cols,
                            "rows": [list(r) for r in rows],
                            "row_count": len(rows),
                            "truncated": len(rows) == max_rows
                        }
                    else:
                        result = {"message": "Query executed successfully (no rows returned)"}
            return [types.TextContent(type="text", text=json.dumps(result, indent=2, default=str))]
        except Exception as e:
            return [types.TextContent(type="text", text=f"Error: {str(e)}")]

    # ── list_databases ────────────────────────────────────────
    elif name == "list_databases":
        query = """
            SELECT DatabaseName, DBKind,
                   CAST(SUM(CurrentPerm)/1e9 AS DECIMAL(10,2)) AS CurrentPermGB,
                   CAST(SUM(MaxPerm)/1e9 AS DECIMAL(10,2)) AS MaxPermGB
            FROM DBC.DiskSpaceV
            GROUP BY 1,2
            ORDER BY CurrentPermGB DESC
        """
        try:
            with get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(query)
                    cols = [d[0] for d in cur.description]
                    rows = [dict(zip(cols, r)) for r in cur.fetchall()]
            return [types.TextContent(type="text", text=json.dumps(rows, indent=2, default=str))]
        except Exception as e:
            return [types.TextContent(type="text", text=f"Error: {str(e)}")]

    # ── list_tables ───────────────────────────────────────────
    elif name == "list_tables":
        query = f"""
            SELECT T.TableName, T.TableKind, T.CreateTimeStamp,
                   CAST(SUM(S.CurrentPerm)/1e6 AS DECIMAL(10,2)) AS SizeMB
            FROM DBC.TablesV T
            LEFT JOIN DBC.TableSizeV S
                ON S.DatabaseName = T.DatabaseName
               AND S.TableName    = T.TableName
            WHERE T.DatabaseName = '{arguments["database"]}'
            GROUP BY 1,2,3
            ORDER BY SizeMB DESC NULLS LAST
        """
        try:
            with get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(query)
                    cols = [d[0] for d in cur.description]
                    rows = [dict(zip(cols, r)) for r in cur.fetchall()]
            return [types.TextContent(type="text", text=json.dumps(rows, indent=2, default=str))]
        except Exception as e:
            return [types.TextContent(type="text", text=f"Error: {str(e)}")]

    # ── get_table_ddl ─────────────────────────────────────────
    elif name == "get_table_ddl":
        query = f"SHOW TABLE {arguments['database']}.{arguments['table_name']}"
        try:
            with get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(query)
                    ddl = "\n".join([row[0] for row in cur.fetchall()])
            return [types.TextContent(type="text", text=ddl)]
        except Exception as e:
            return [types.TextContent(type="text", text=f"Error: {str(e)}")]

    # ── get_schema ────────────────────────────────────────────
    elif name == "get_schema":
        query = f"""
            SELECT ColumnName, ColumnType, ColumnLength,
                   Nullable, DefaultValue, ColumnFormat,
                   ColumnTitle, CommentString
            FROM DBC.ColumnsV
            WHERE DatabaseName = '{arguments["database"]}'
              AND TableName    = '{arguments["table_name"]}'
            ORDER BY ColumnId
        """
        try:
            with get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(query)
                    cols = [d[0] for d in cur.description]
                    rows = [dict(zip(cols, r)) for r in cur.fetchall()]
            return [types.TextContent(type="text", text=json.dumps(rows, indent=2, default=str))]
        except Exception as e:
            return [types.TextContent(type="text", text=f"Error: {str(e)}")]

    # ── profile_table ─────────────────────────────────────────
    elif name == "profile_table":
        db    = arguments["database"]
        table = arguments["table_name"]
        try:
            with get_connection(db) as conn:
                with conn.cursor() as cur:
                    cur.execute(f"SELECT COUNT(*) FROM {db}.{table}")
                    row_count = cur.fetchone()[0]

                    cur.execute(f"""
                        SELECT CAST(SUM(CurrentPerm)/1e6 AS DECIMAL(10,2))
                        FROM DBC.TableSizeV
                        WHERE DatabaseName='{db}' AND TableName='{table}'
                    """)
                    size_mb = cur.fetchone()[0]

                    cur.execute(f"""
                        SELECT ColumnName, ColumnType
                        FROM DBC.ColumnsV
                        WHERE DatabaseName='{db}' AND TableName='{table}'
                        ORDER BY ColumnId
                    """)
                    columns = cur.fetchall()

                    profile = {
                        "table": f"{db}.{table}",
                        "row_count": row_count,
                        "size_mb": float(size_mb or 0),
                        "columns": {}
                    }

                    for col_name, col_type in columns:
                        col_stats = {"type": col_type}
                        cur.execute(f"SELECT COUNT(*) FROM {db}.{table} WHERE {col_name} IS NULL")
                        nulls = cur.fetchone()[0]
                        col_stats["null_count"] = nulls
                        col_stats["null_pct"]   = round(nulls / row_count * 100, 2) if row_count else 0
                        cur.execute(f"SELECT COUNT(DISTINCT {col_name}) FROM {db}.{table}")
                        col_stats["distinct_count"] = cur.fetchone()[0]
                        if col_type in ('I', 'I2', 'I8', 'F', 'D', 'N'):
                            cur.execute(f"SELECT MIN({col_name}), MAX({col_name}), AVG({col_name}) FROM {db}.{table}")
                            mn, mx, av = cur.fetchone()
                            col_stats.update({"min": mn, "max": mx, "avg": round(float(av), 4) if av else None})
                        profile["columns"][col_name] = col_stats

            return [types.TextContent(type="text", text=json.dumps(profile, indent=2, default=str))]
        except Exception as e:
            return [types.TextContent(type="text", text=f"Error: {str(e)}")]

    # ── check_duplicates ──────────────────────────────────────
    elif name == "check_duplicates":
        db       = arguments["database"]
        table    = arguments["table_name"]
        key_cols = ", ".join(arguments["key_columns"])
        query = f"""
            SELECT {key_cols}, COUNT(*) AS duplicate_count
            FROM {db}.{table}
            GROUP BY {key_cols}
            HAVING COUNT(*) > 1
            ORDER BY duplicate_count DESC
        """
        try:
            with get_connection(db) as conn:
                with conn.cursor() as cur:
                    cur.execute(query)
                    cols = [d[0] for d in cur.description]
                    rows = [dict(zip(cols, r)) for r in cur.fetchmany(100)]
            result = {
                "duplicate_groups_found": len(rows),
                "sample_duplicates": rows
            }
            return [types.TextContent(type="text", text=json.dumps(result, indent=2, default=str))]
        except Exception as e:
            return [types.TextContent(type="text", text=f"Error: {str(e)}")]

    # ── get_table_stats ───────────────────────────────────────
    elif name == "get_table_stats":
        query = f"""
            SELECT ColumnName, StatsType, SampleSize,
                   RowCount, UniqueValueCount, LastCollectTimeStamp
            FROM DBC.StatsV
            WHERE DatabaseName = '{arguments["database"]}'
              AND TableName    = '{arguments["table_name"]}'
        """
        try:
            with get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(query)
                    cols = [d[0] for d in cur.description]
                    rows = [dict(zip(cols, r)) for r in cur.fetchall()]
            if not rows:
                return [types.TextContent(type="text", text="No stats collected yet. Consider running COLLECT STATISTICS.")]
            return [types.TextContent(type="text", text=json.dumps(rows, indent=2, default=str))]
        except Exception as e:
            return [types.TextContent(type="text", text=f"Error: {str(e)}")]

    # ── check_data_skew ───────────────────────────────────────
    elif name == "check_data_skew":
        db    = arguments["database"]
        table = arguments["table_name"]
        query = f"""
            SELECT Vproc AS AMP,
                   CurrentPerm AS BytesOnAMP,
                   CAST(100.0 * CurrentPerm /
                        NULLIFZERO(SUM(CurrentPerm) OVER()) AS DECIMAL(5,2)) AS PctOfTotal
            FROM DBC.TableSizeV
            WHERE DatabaseName = '{db}'
              AND TableName    = '{table}'
            ORDER BY CurrentPerm DESC
        """
        try:
            with get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(query)
                    cols = [d[0] for d in cur.description]
                    rows = [dict(zip(cols, r)) for r in cur.fetchall()]

            if rows:
                pcts  = [float(r["PctOfTotal"] or 0) for r in rows]
                max_p = max(pcts)
                avg_p = sum(pcts) / len(pcts)
                skew  = round(max_p - avg_p, 2)
                result = {
                    "amp_count":        len(rows),
                    "max_amp_pct":      max_p,
                    "avg_amp_pct":      round(avg_p, 2),
                    "skew_factor":      skew,
                    "skew_severity":    "HIGH" if skew > 10 else "MEDIUM" if skew > 5 else "LOW",
                    "amp_distribution": rows[:20]
                }
            else:
                result = {"message": "No size data found — table may be empty"}

            return [types.TextContent(type="text", text=json.dumps(result, indent=2, default=str))]
        except Exception as e:
            return [types.TextContent(type="text", text=f"Error: {str(e)}")]

    # ── get_pi_info ───────────────────────────────────────────
    elif name == "get_pi_info":
        query = f"""
            SELECT IndexType, IndexName, ColumnName, ColumnPosition, UniqueFlag
            FROM DBC.IndicesV
            WHERE DatabaseName = '{arguments["database"]}'
              AND TableName    = '{arguments["table_name"]}'
            ORDER BY IndexType, ColumnPosition
        """
        try:
            with get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(query)
                    cols = [d[0] for d in cur.description]
                    rows = [dict(zip(cols, r)) for r in cur.fetchall()]

            pi_map = {
                "P": "Primary Index (PI)",
                "Q": "Primary AMP Index",
                "S": "Secondary Index (SI)",
                "J": "Join Index",
                "K": "Primary Key"
            }
            for r in rows:
                r["IndexTypeDescription"] = pi_map.get(r.get("IndexType", ""), "Other")

            return [types.TextContent(type="text", text=json.dumps(rows, indent=2, default=str))]
        except Exception as e:
            return [types.TextContent(type="text", text=f"Error: {str(e)}")]

    # ── find_table_references ─────────────────────────────────
    elif name == "find_table_references":
        query = f"""
            SELECT ObjectDatabaseName AS ReferencingDB,
                   ObjectTableName    AS ReferencingObject,
                   ObjectType
            FROM DBC.ObjectUsageV
            WHERE DatabaseName    = '{arguments["database"]}'
              AND ObjectTableName LIKE '%{arguments["table_name"]}%'
            ORDER BY 1,2
        """
        try:
            with get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(query)
                    cols = [d[0] for d in cur.description]
                    rows = [dict(zip(cols, r)) for r in cur.fetchall()]
            return [types.TextContent(type="text", text=json.dumps(rows, indent=2, default=str))]
        except Exception as e:
            return [types.TextContent(type="text", text=f"Error: {str(e)}")]

    # ── export_to_csv ─────────────────────────────────────────
    elif name == "export_to_csv":
        try:
            with get_connection(arguments.get("database")) as conn:
                with conn.cursor() as cur:
                    cur.execute(arguments["query"])
                    cols = [d[0] for d in cur.description]
                    rows = cur.fetchall()
            output = io.StringIO()
            writer = csv.writer(output)
            writer.writerow(cols)
            writer.writerows(rows)
            Path(arguments["output_path"]).write_text(output.getvalue())
            return [types.TextContent(type="text", text=f"Exported {len(rows)} rows → {arguments['output_path']}")]
        except Exception as e:
            return [types.TextContent(type="text", text=f"Error: {str(e)}")]

    # ── get_space_usage ───────────────────────────────────────
    elif name == "get_space_usage":
        db_filter = f"WHERE DatabaseName = '{arguments['database']}'" if arguments.get("database") else ""
        query = f"""
            SELECT DatabaseName,
                   CAST(SUM(CurrentPerm)/1e9 AS DECIMAL(10,2)) AS CurrentPermGB,
                   CAST(SUM(MaxPerm)/1e9     AS DECIMAL(10,2)) AS MaxPermGB,
                   CAST(100.0 * SUM(CurrentPerm) /
                        NULLIFZERO(SUM(MaxPerm)) AS DECIMAL(5,2)) AS UsedPct
            FROM DBC.DiskSpaceV
            {db_filter}
            GROUP BY 1
            ORDER BY CurrentPermGB DESC
        """
        try:
            with get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(query)
                    cols = [d[0] for d in cur.description]
                    rows = [dict(zip(cols, r)) for r in cur.fetchall()]
            return [types.TextContent(type="text", text=json.dumps(rows, indent=2, default=str))]
        except Exception as e:
            return [types.TextContent(type="text", text=f"Error: {str(e)}")]

    # ── run_kmeans_experiment ─────────────────────────────────
    elif name == "run_kmeans_experiment":
        db        = arguments["database"]
        table     = arguments["table_name"]
        id_col    = arguments["id_column"]
        features  = arguments["feature_columns"]
        k_min     = arguments.get("k_min", 2)
        k_max     = arguments.get("k_max", 10)
        max_iter  = arguments.get("max_iterations", 100)
        feat_cols = ", ".join([f"'{f}'" for f in features])

        results = []
        for k in range(k_min, k_max + 1):
            seed_sql = " UNION ALL ".join([
                f"SELECT {i} AS td_clusterid_kmeans, {', '.join(features)} FROM {db}.{table} SAMPLE 1"
                for i in range(k)
            ])
            query = f"""
                SELECT * FROM val.tda_kmeans (
                    ON (SELECT * FROM {db}.{table}) AS InputTable PARTITION BY ANY
                    ON ({seed_sql}) AS InitialSeeds DIMENSION
                    USING
                        TargetColumns ({feat_cols})
                        IDColumn ('{id_col}')
                        MaxIterations ('{max_iter}')
                ) AS dt
            """
            try:
                with get_connection(db) as conn:
                    with conn.cursor() as cur:
                        cur.execute(query)
                        cols_desc = [d[0] for d in cur.description]
                        rows = cur.fetchall()

                within_ss_total = None
                between_ss      = None
                n_iterations    = None
                converged       = None
                cluster_sizes   = []

                for row in rows:
                    r = dict(zip(cols_desc, row))
                    info = r.get("td_modelinfo_kmeans")
                    if info:
                        info_str = str(info)
                        if "Total_WithinSS" in info_str:
                            try: within_ss_total = float(info_str.split(":")[-1].strip())
                            except: pass
                        elif "Between_SS" in info_str:
                            try: between_ss = float(info_str.split(":")[-1].strip())
                            except: pass
                        elif "Number of Iterations" in info_str:
                            try: n_iterations = int(info_str.split(":")[-1].strip())
                            except: pass
                        elif "Converged" in info_str:
                            converged = "True" in info_str
                    elif r.get("td_clusterid_kmeans") is not None:
                        cluster_sizes.append({
                            "cluster_id": r["td_clusterid_kmeans"],
                            "size":       r.get("td_size_kmeans"),
                            "within_ss":  r.get("td_withinss_kmeans")
                        })

                results.append({
                    "k":             k,
                    "within_ss":     within_ss_total,
                    "between_ss":    between_ss,
                    "n_iterations":  n_iterations,
                    "converged":     converged,
                    "cluster_sizes": cluster_sizes
                })

            except Exception as e:
                results.append({"k": k, "error": str(e)})

        # Compute elbow deltas to help identify the elbow point
        valid = [r for r in results if r.get("within_ss") is not None]
        for i, r in enumerate(valid):
            r["within_ss_delta"] = None if i == 0 else round(valid[i-1]["within_ss"] - r["within_ss"], 6)

        return [types.TextContent(type="text", text=json.dumps(results, indent=2, default=str))]

    # ── run_kmeans_final ──────────────────────────────────────
    elif name == "run_kmeans_final":
        db          = arguments["database"]
        table       = arguments["table_name"]
        id_col      = arguments["id_column"]
        features    = arguments["feature_columns"]
        k           = arguments["n_clusters"]
        model_table = arguments["model_table"]
        max_iter    = arguments.get("max_iterations", 100)
        feat_cols   = ", ".join([f"'{f}'" for f in features])
        seed_sql    = " UNION ALL ".join([
            f"SELECT {i} AS td_clusterid_kmeans, {', '.join(features)} FROM {db}.{table} SAMPLE 1"
            for i in range(k)
        ])
        try:
            with get_connection(db) as conn:
                with conn.cursor() as cur:
                    try:
                        cur.execute(f"DROP TABLE {db}.{model_table}")
                    except:
                        pass
                    cur.execute(f"""
                        CREATE MULTISET TABLE {db}.{model_table} AS (
                            SELECT * FROM val.tda_kmeans (
                                ON (SELECT * FROM {db}.{table}) AS InputTable PARTITION BY ANY
                                ON ({seed_sql}) AS InitialSeeds DIMENSION
                                USING
                                    TargetColumns ({feat_cols})
                                    IDColumn ('{id_col}')
                                    MaxIterations ('{max_iter}')
                            ) AS dt
                        ) WITH DATA
                    """)
            return [types.TextContent(type="text", text=json.dumps({
                "status":      "success",
                "model_table": f"{db}.{model_table}",
                "k":           k,
                "features":    features,
                "message":     f"KMeans model (k={k}) saved to {db}.{model_table}"
            }, indent=2))]
        except Exception as e:
            return [types.TextContent(type="text", text=f"Error: {str(e)}")]

    return [types.TextContent(type="text", text=f"Unknown tool: {name}")]


# ─── RESOURCES ────────────────────────────────────────────────

@app.list_resources()
async def list_resources():
    return [
        types.Resource(
            uri="teradata://dbc-reference",
            name="Teradata DBC System Tables Reference",
            description="Quick reference for DBC.* system views used in this server",
            mimeType="text/plain"
        ),
        types.Resource(
            uri="teradata://best-practices",
            name="Teradata Data Engineering Best Practices",
            description="PI selection, skew avoidance, stats collection guidance",
            mimeType="text/plain"
        )
    ]

@app.read_resource()
async def read_resource(uri: str):
    if str(uri) == "teradata://dbc-reference":
        return types.ReadResourceResult(contents=[types.TextContent(type="text", text="""
TERADATA DBC SYSTEM VIEWS REFERENCE
=====================================
DBC.TablesV       → All tables/views with metadata
DBC.ColumnsV      → Column definitions per table
DBC.IndicesV      → Index definitions (PI, SI, JI)
DBC.DiskSpaceV    → Perm/Spool space per database per AMP
DBC.TableSizeV    → Current perm size per table per AMP
DBC.StatsV        → Statistics collection metadata
DBC.ObjectUsageV  → Cross-object dependencies/references
DBC.SessionInfoV  → Active sessions
DBC.LogOnOffV     → Logon/logoff audit log
""")])

    if str(uri) == "teradata://best-practices":
        return types.ReadResourceResult(contents=[types.TextContent(type="text", text="""
TERADATA BEST PRACTICES FOR DATA ENGINEERING
=============================================
PRIMARY INDEX (PI):
  - Choose a column with HIGH cardinality to avoid skew
  - Use the most frequent JOIN column as PI
  - Avoid NULLable columns as PI
  - NUPI (Non-Unique PI) is default; use UPI when cardinality allows

STATISTICS:
  - COLLECT STATISTICS on all PI columns
  - COLLECT STATISTICS on JOIN columns (non-PI)
  - COLLECT STATISTICS on columns used in WHERE filters
  - Re-collect stats after >10% data change

SKEW:
  - Skew factor > 10% = investigate PI choice
  - Use HASHAMP(HASHBUCKET(HASHROW(col))) to test distribution

SPACE:
  - Monitor DBC.DiskSpaceV regularly
  - Spool space errors = query needs optimization or more spool
  - Use COMPRESS on low-cardinality columns to save perm space

PERFORMANCE:
  - Use EXPLAIN before running large queries
  - Avoid full table scans — use PARTITION BY RANGE where applicable
  - Prefer VOLATILE tables over global temp for session-scoped work
""")])


# ─── PROMPTS ──────────────────────────────────────────────────

@app.list_prompts()
async def list_prompts():
    return [
        types.Prompt(
            name="td-dq-check",
            description="Full Teradata data quality audit for a table",
            arguments=[
                types.PromptArgument(name="database",   description="Teradata database", required=True),
                types.PromptArgument(name="table_name", description="Table to audit",    required=True)
            ]
        ),
        types.Prompt(
            name="td-pi-review",
            description="Analyze and recommend a better Primary Index for a Teradata table",
            arguments=[
                types.PromptArgument(name="database",   description="Teradata database", required=True),
                types.PromptArgument(name="table_name", description="Table to review",   required=True)
            ]
        )
    ]

@app.get_prompt()
async def get_prompt(name: str, arguments: dict):
    db    = arguments.get("database", "")
    table = arguments.get("table_name", "")

    if name == "td-dq-check":
        return types.GetPromptResult(
            description=f"Data quality audit for {db}.{table}",
            messages=[types.PromptMessage(role="user", content=types.TextContent(type="text", text=f"""
Please perform a comprehensive Teradata data quality audit on {db}.{table}:

1. Get the schema (DBC.ColumnsV) and describe the table structure
2. Profile the table: row count, size, null %, distinct counts
3. Check for duplicate rows on the Primary Index columns
4. Review data skew across AMPs — flag if skew > 10%
5. Check if statistics have been collected and are recent
6. Identify columns with >20% nulls as data quality risks
7. Recommend COLLECT STATISTICS commands if stats are missing/stale
8. Provide a data quality score (0-100) with justification
"""))]
        )

    if name == "td-pi-review":
        return types.GetPromptResult(
            description=f"Primary Index review for {db}.{table}",
            messages=[types.PromptMessage(role="user", content=types.TextContent(type="text", text=f"""
Review the Primary Index design for Teradata table {db}.{table}:

1. Get the current PI definition using get_pi_info
2. Profile the PI column(s) — check cardinality and null %
3. Check data skew using check_data_skew
4. Review how the table is typically joined (look at DBC.ObjectUsageV)
5. Recommend whether to keep, change, or add a Partitioned Primary Index (PPI)
6. Provide the exact ALTER TABLE or CREATE TABLE DDL for any recommended changes
"""))]
        )


# ─── ENTRYPOINT ───────────────────────────────────────────────

async def main():
    async with stdio_server() as (read_stream, write_stream):
        await app.run(read_stream, write_stream, app.create_initialization_options())

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
