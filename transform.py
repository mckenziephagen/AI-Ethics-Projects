import pandas as pd
import pickle as pkl
from sklearn.decomposition import PCA

df = pd.read_pickle("/data/cong/connectome.pkl")
df.rename(columns={ df.columns[0]: "Subject" }, inplace = True)

pca = PCA(n_components=300)

X = pca.fit_transform(df.iloc[:, 1:].values)

with open('data/pca.pkl', 'a') as f:
    pickle.dump(X, f)