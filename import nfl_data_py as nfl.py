import pandas as pd
import nfl_data_py as nfl

# Load play-by-play data for 2022
stats = nfl.import_pbp_data([2022], downcast=True, cache=False)

# Show all columns
pd.set_option('display.max_columns', None)

# Now print the first few rows
print(stats.head())

# Optional: also print column names
print(stats.columns)
