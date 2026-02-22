import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from scipy.stats import qmc

def lhs_nearest_sampling(df, features, target, n_samples=5000):
    # 1. Normalizzazione
    scaler = StandardScaler()
    
    X_all = scaler.fit_transform(df[features])
    
    # 2. Latin Hypercube Sampling (LHS) per coprire lo spazio
    sampler = qmc.LatinHypercube(d=len(features), seed=42)
    sample_points = sampler.random(n=n_samples)
    
    # Scaliamo i punti LHS nello spazio dei dati reali
    l_bounds = X_all.min(axis=0)
    u_bounds = X_all.max(axis=0)
    sample_points_scaled = qmc.scale(sample_points, l_bounds, u_bounds)
    
    # 3. Trova i punti reali più vicini (Space-filling) 
    nn = NearestNeighbors(n_neighbors=1)
    nn.fit(X_all)
    distances, indices = nn.kneighbors(sample_points_scaled)
    
    unique_indices = np.unique(indices)
    
    X_train = X_all[unique_indices]
    y_train = df[target].values[unique_indices]
    
    return X_train, y_train, scaler
