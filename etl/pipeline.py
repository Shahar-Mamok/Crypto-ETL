
from extract import extract
from transform import transform_markets, transform_history
from load import load_to_sqlite

def run_pipeline():
	# Extract
	data = extract()
	# Transform
	df_markets = transform_markets(data["markets"])
	df_history = transform_history(data["history"])
	# Load
	load_to_sqlite(df_markets, "markets", if_exists="replace")
	load_to_sqlite(df_history, "history", if_exists="replace")
	print("ETL pipeline completed. Markets and history tables updated.")

if __name__ == "__main__":
	run_pipeline()