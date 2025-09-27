from etl.extract import extract
from etl.transform import transform_markets, transform_history
from etl.dq_checks import validate_markets, validate_history
from etl.load import load_to_sqlite

print("Running extract()...")
data = extract()
print("markets shape:", data["markets"].shape)
print("history shape:", data["history"].shape)

print("Transform...")
m = transform_markets(data["markets"])
h = transform_history(data["history"])
print("markets after transform:", m.shape)
print("history after transform:", h.shape)

print("Validate...")
validate_markets(m); validate_history(h)

print("Load to SQLite...")
load_to_sqlite(m, "markets")
load_to_sqlite(h, "history")
print("DONE")
