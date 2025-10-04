import duckdb
import logging
from fastapi import FastAPI
from pydantic import BaseModel

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

app = FastAPI()

# Paths to your databases
DB_PATHS = {
    "log": "/root/sport_agents/app/log_data/188bet_log.duckdb",
    "predict": "/root/sport_agents/app/log_data/188bet_predict.duckdb"
}

class QueryRequest(BaseModel):
    sql: str

def run_duckdb_query(db_path: str, sql: str):
    logger.info(f"Executing query on {db_path}: {sql}")
    try:
        con = duckdb.connect(db_path, read_only=True)
        result = con.execute(sql).fetchall()
        columns = [desc[0] for desc in con.description]
        con.close()
        logger.info(f"Query executed successfully, returned {len(result)} rows")
        return {"columns": columns, "rows": result}
    except Exception as e:
        logger.error(f"Query failed on {db_path}: {e}")
        return {"error": str(e)}

@app.post("/query/log")
def query_log(request: QueryRequest):
    return run_duckdb_query(DB_PATHS["log"], request.sql)

@app.post("/query/predict")
def query_predict(request: QueryRequest):
    return run_duckdb_query(DB_PATHS["predict"], request.sql)

@app.get("/")
def root():
    logger.info("Health check hit at /")
    return {
        "message": "DuckDB query API is running",
        "databases": list(DB_PATHS.keys())
    }