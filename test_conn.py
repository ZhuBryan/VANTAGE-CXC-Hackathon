import os
from dotenv import load_dotenv
import snowflake.connector

load_dotenv()

try:
    conn = snowflake.connector.connect(
        user=os.getenv("SNOW_USER"),
        password=os.getenv("SNOW_PASS"),
        account=os.getenv("SNOW_ACC"),
        warehouse=os.getenv("SNOW_WH"),
        database=os.getenv("SNOW_DB"),
        schema=os.getenv("SNOW_SCHEMA")
    )
    print("✅ Connection Successful! Member 2 infrastructure is LIVE.")
    conn.close()
except Exception as e:
    print(f"❌ Connection Failed: {e}")
