import pandas as pd

# read in the fitted values from the regression tree based aid model
# and turn it into a dictionary
aid_df = pd.read_csv("aid_model.csv", na_filter=False)
aid_context_model = {}
for (index, row) in aid_df.iterrows():
    aid_context_model[row["motif"]] = row["pred"]
