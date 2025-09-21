import requests
import json
import pandas as pd
import re
import sys
import os
import numpy as np

from dotenv import load_dotenv
load_dotenv()  # This loads variables from .env into environment

sys_path = os.getenv("sys_path")
print(sys_path)
os.chdir(sys_path)
sys.path.append(sys_path)


sql = """

   WITH ranked AS (
    SELECT *
    FROM "hose_snapshot"
    WHERE "run_time"::TIMESTAMP = (SELECT max("run_time") FROM "hose_snapshot")
    )
    SELECT *
    FROM ranked
    WHERE rn = 1
    LIMIT 100
    ;

"""  # your query
resp = requests.post(
    "http://165.232.188.235:8002/query/hose",
    json={"sql": f"{sql}"}
)

print(resp.status_code)
print(resp.json())

data = resp.json()
df = pd.DataFrame(data["rows"], columns=data["columns"])
