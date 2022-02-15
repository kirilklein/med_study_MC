from sklearn.neighbors import NearestNeighbors


def NN_matching(df):
    PS_cases = df[df.disease==1].PS.toarray()
    PS_controls = df[df.disease==0].PS.toarray()
    neigh = NearestNeighbors(n_neighbors=10)
    neigh.fit(PS_controls)
    distances, indices = neigh.kneighbors(PS_cases)
    return distances, indices

