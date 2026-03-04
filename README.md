# 🏦 AI-Powered Customer Intelligence PoC
### Claude + Teradata ClearScape Analytics via Model Context Protocol (MCP)

---

## What Is This?

This project demonstrates how **Claude AI** can be connected directly to **Teradata** to enable:

- 🔍 **Natural language data exploration** — Ask Claude about your data in plain English
- 🤖 **In-database ML** — Run KMeans clustering inside Teradata via Claude, no data movement
- 📊 **Progressive segmentation** — Show how richer features reveal richer customer segments
- 🔗 **MCP as a data engineering framework** — A reusable pattern for AI + enterprise data

---

## The Business Story

> *"We started with 50,000 customers and 2 years of transactions.*
> *By asking Claude natural language questions, we profiled our data, identified quality issues,*
> *and ran customer segmentation — all without writing a single line of SQL manually.*
> *Everything ran inside Teradata. No data left the platform."*

### Three Phases of Segmentation

| Phase | Features Used | What It Reveals |
|---|---|---|
| **1. Behavioral** | RFM (Recency, Frequency, Monetary) | How customers transact |
| **2. + Risk** | RFM + Credit Score | Who is financially at-risk |
| **3. + Demographics** | RFM + Risk + Age + Income | Full customer identity |

Each phase finds a **different optimal number of clusters** — proving that richer data = richer insight.

---

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                  Claude Desktop                      │
│          (Natural Language Interface)                │
└────────────────────┬────────────────────────────────┘
                     │ MCP Protocol
┌────────────────────▼────────────────────────────────┐
│              MCP Server (server.py)                  │
│   13 Tools: query, profile, skew, PI, KMeans...     │
└────────────────────┬────────────────────────────────┘
                     │ teradatasql
┌────────────────────▼────────────────────────────────┐
│         Teradata ClearScape Analytics                │
│   transactions │ demographics │ credit_risk          │
│   segmentation_*_scaled │ val.tda_kmeans             │
└─────────────────────────────────────────────────────┘
```

---

## Project Structure

```
teradata-mcp-poc/
│
├── config.example.yaml        ← Copy this → config.yaml and add credentials
├── requirements.txt           ← Python dependencies
│
├── mcp/
│   └── server.py              ← MCP server (15 tools for Teradata)
│
├── notebooks/
│   └── 01_poc_walkthrough.ipynb  ← Full PoC narrative (run this first)
│
├── scripts/
│   └── kmeans_experiment.py   ← Standalone Python experiment runner
│
└── docs/
    └── mcp_tools_reference.md ← All 15 MCP tools explained
```

---

## Quick Start

### Prerequisites
- Python 3.11+
- Claude Desktop ([download here](https://claude.ai/download))
- Teradata ClearScape account ([free trial](https://www.teradata.com/getting-started/demos/clearscape-analytics))

### 1. Clone the repo
```bash
git clone https://github.com/YOUR_USERNAME/teradata-mcp-poc.git
cd teradata-mcp-poc
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Configure credentials
```bash
cp config.example.yaml config.yaml
# Edit config.yaml with your Teradata host, username, and password
```

### 4. Connect Claude Desktop to Teradata
Add this to your Claude Desktop config file:
- **Mac**: `~/Library/Application Support/Claude/claude_desktop_config.json`
- **Windows**: `%APPDATA%\Claude\claude_desktop_config.json`

```json
{
  "mcpServers": {
    "teradata": {
      "command": "python",
      "args": ["C:\\path\\to\\teradata-mcp-poc\\mcp\\server.py"]
    }
  }
}
```

Restart Claude Desktop — you'll see **teradata** appear in Settings → Connectors.

### 5. Talk to your data
Open Claude Desktop and ask:
```
List all my Teradata databases
Show me the tables in demo_user
Profile the transactions table
Check for data skew on the customer_demographics table
Run a data quality check on demo_user.credit_risk
```

### 6. Run the notebook
```bash
jupyter notebook notebooks/01_poc_walkthrough.ipynb
```

---

## MCP Tools Reference

| Tool | What it does |
|---|---|
| `run_sql` | Execute any Teradata SQL |
| `list_databases` | Show all accessible databases |
| `list_tables` | Show tables in a database |
| `get_schema` | Column definitions via DBC.ColumnsV |
| `get_table_ddl` | Full CREATE TABLE statement |
| `profile_table` | Row count, nulls, distinct values, size |
| `check_duplicates` | Find duplicate rows on key columns |
| `get_table_stats` | Optimizer statistics (COLLECT STATS) |
| `check_data_skew` | AMP distribution analysis |
| `get_pi_info` | Primary Index definition |
| `find_table_references` | Views/macros that use a table |
| `export_to_csv` | Save query results to CSV |
| `get_space_usage` | Perm space usage per database |
| `run_kmeans_experiment` | Elbow analysis for k=2..N |
| `run_kmeans_final` | Save final KMeans model to table |

---

## Example Conversations with Claude

**Data exploration:**
> *"Profile the transactions table and tell me if there are any data quality issues"*

**Segmentation:**
> *"Run a KMeans experiment on segmentation_rfm_scaled using monetary_scaled, frequency_scaled, recency_scaled for k=2 to 8"*

**Performance:**
> *"Check if the customer_demographics table has any skew issues and review its Primary Index"*

**Lineage:**
> *"Find everything that references the transactions table"*

---

## Dataset

The PoC uses a synthetic customer dataset with:
- **799,477 transactions** across 50,000 customers (Jan 2023 – Dec 2024)
- **Customer demographics**: age, income, employment years
- **Credit risk**: credit score (400–849), loan exposure, late payments
- **Pre-built features**: RFM aggregations, multiple scaled tables for clustering

---

## Tech Stack

| Component | Technology |
|---|---|
| AI Interface | Claude Desktop |
| AI ↔ Data Protocol | MCP (Model Context Protocol) |
| Database | Teradata ClearScape Analytics |
| In-Database ML | Teradata VAL (`val.tda_kmeans`) |
| Python ML | `teradataml`, `scikit-learn` |
| Visualization | `matplotlib`, `seaborn` |

---

## License

MIT — free to use and adapt for your own PoC.

---

*Built with Claude Desktop + Teradata ClearScape Analytics*
