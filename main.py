import pandas as pd

import arabic
import prepare_df

pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)
pd.set_option('display.max_rows', None)

df = prepare_df.prepare_df()

print(df.head(50))
