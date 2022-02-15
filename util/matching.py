from sklearn.neighbors import NearestNeighbors
import pandas as pd


####################NN###########################
def NN_matching(df):
    df_cases = df[df.disease==1]
    PS_cases = df_cases.PS.to_numpy()
    df_controls = df[df.disease==0]
    PS_controls = df_controls.PS.to_numpy()
    neigh = NearestNeighbors(n_neighbors=10)
    neigh.fit(PS_controls.reshape(-1,1))
    distances, indices = neigh.kneighbors(PS_cases.reshape(-1,1))
    indices = indices.flatten()
    df_matched_controls = df_controls.iloc[indices, :]
    df = pd.concat([df_cases, df_matched_controls], ignore_index=True)
    return df