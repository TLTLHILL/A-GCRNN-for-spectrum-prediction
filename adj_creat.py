import pandas as pd

# Calculate adjacency matrix through correlation coefficient
df = pd.read_csv('my_data/data.csv',header=None)

corr_matrix = df.corr(method='spearman',min_periods=1)

adj_matrix = (corr_matrix > 0.9).astype(int)

adj_matrix.to_csv('my_data/adj_matrix_0.9_spearman.csv', index=False)

