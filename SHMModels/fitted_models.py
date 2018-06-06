import pandas as pd
import pkgutil
import io

# read in the fitted values from the regression tree based aid model
# and turn it into a dictionary
aid_df = pd.read_csv(io.BytesIO(pkgutil.get_data("SHMModels", "data/aid_model.csv")),
                     encoding='utf8',
                     na_filter=False)
aid_context_model = {}
for (index, row) in aid_df.iterrows():
    aid_context_model[row["motif"]] = row["pred"]
