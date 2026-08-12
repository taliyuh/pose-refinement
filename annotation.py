import sys
import pandas as pd

# Force terminal output to accept UTF-8 blocks cleanly
sys.stdout.reconfigure(encoding='utf-8')

url = "https://docs.google.com/document/d/e/2PACX-1vSZ9d7OCd4QMsjJi2VFQmPYLebG2sGqI879_bSPugwOo_fgRcZLAFyfajPWU91UDiLg-RxRD41lVYRA/pub"

# 1. Directly target the correct table and treat the first row as headers automatically
tables = pd.read_html(url, match="Character", header=0, encoding='utf-8')
df = tables[0]

# Clean column spaces instantly
df.columns = df.columns.str.strip()

# 2. Map coordinates (converted to int) to their respective characters
grid_dict = {
    (int(row['x-coordinate']), int(row['y-coordinate'])): row['Character'] 
    for _, row in df.iterrows()
}

# 3. Establish grid boundaries
max_x = max(x for x, y in grid_dict.keys()) + 1
max_y = max(y for x, y in grid_dict.keys()) + 1

# 4. Render the grid (reversing the Y loop so 0 sits at the bottom)
for y in range(max_y - 1, -1, -1):
    for x in range(max_x):
        print(grid_dict.get((x, y), " "), end="")
    print()