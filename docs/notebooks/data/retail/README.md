
# Online Retail II dataset

This dataset contains all the transactions occurring for a UK-based and registered, non-store online retail between 01/12/2009 and 09/12/2011.
The company mainly sells unique all-occasion gift-ware.
Many customers of the company are wholesalers.

The original dataset is available on the [UCI machine learning repository](https://archive.ics.uci.edu/ml/datasets/Online+Retail+II).
For convenience, the CSV variant from [Kaggle](https://www.kaggle.com/mashlyn/online-retail-ii-uci) has been used, which was released under CC-0.
To reduce size, the following extract has been done:

```python
import pandas as pd

# Load CSV dump
df = pd.read_csv("online_retail_II.csv")

# Clean a bit
df = df.dropna(subset=["Description"])
df["Description"] = df["Description"].str.strip()

# Index product names
names = df["Description"].value_counts().index
df["Index"] = names.get_indexer(df["Description"]).astype(str)

# Group by transaction
transaction_df = df.groupby("Invoice")["Index"].apply(";".join)
transaction_df = transaction_df.reset_index(name="Products")

# Export as CSV
name_df = names.to_frame(index=False, name="name")
name_df.to_csv("name.csv", index=False)
transaction_df.to_csv("transaction.csv", index=False)
```
