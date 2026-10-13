import duckdb

con = duckdb.connect("hp.duckdb")
for name in ["chapters", "segments", "sentences"]:
    con.execute(f"CREATE OR REPLACE VIEW {name} AS "
                f"SELECT * FROM 'data/processed/{name}.parquet'")
con.execute("INSTALL ui")
con.execute("LOAD ui")
con.execute("CALL start_ui_server()")
print("DuckDB UI running at http://localhost:4213")
input("Press Enter here to stop the UI... ")
