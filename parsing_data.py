import pandas as pd

# Load the CSV
df = pd.read_csv("belgium_prices_feb2022_raw.csv")

# Rename columns for clarity
df.columns = ["start_time", "end_time", "price_eur_per_MWh"]

# Combine start and end columns into one timestamp string
df["timestamp_str"] = df["start_time"].str.strip() + " " + df["end_time"].str.strip()

# Parse the combined string into a datetime object
# Example format: "Feb 1 2022 12:00 AM"
df["timestamp"] = pd.to_datetime(df["timestamp_str"], format="%b %d %Y %I:%M %p", errors='coerce')

# Check for parsing errors
if df["timestamp"].isnull().any():
    print("Warning: Some timestamps could not be parsed:")
    print(df[df["timestamp"].isnull()][["timestamp_str", "start_time", "end_time"]].head())

# Convert €/MWh to €/kWh
df["price_eur_per_kWh"] = df["price_eur_per_MWh"] / 1000

# Keep only what we need
df_clean = df[["timestamp", "price_eur_per_kWh"]].dropna()

# Save cleaned data
df_clean.to_csv("belgium_prices_feb2022.csv", index=False)

print("✅ Preprocessed data saved to 'belgium_prices_feb2022.csv'")
