import requests
import json
import pandas as pd

sql =     """
    SELECT * FROM "188bet_log" 
    WHERE "run_time"::TIMESTAMP >= (NOW()::timestamp) - INTERVAL '1.5 hours'
    AND "run_time"::TIMESTAMP <= (NOW()::timestamp + INTERVAL '7 hours')
    """

resp = requests.post("http://165.232.188.235:8000/query/log",
                     json={"sql": f"{sql}"})
print(resp.json())

data = resp.json()

df = pd.DataFrame(data["rows"], columns=data["columns"])