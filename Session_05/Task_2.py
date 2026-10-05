"""
Session 05 - Task 2
Agglomerative Hierarchical Clustering on 8 Songs (Tempo & Energy).
Plots dendrogram using scipy.cluster.hierarchy.dendrogram.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.preprocessing import StandardScaler

# Dataset of 8 Songs with Tempo (BPM) and Energy (0-1)
songs_data = [
    {"song_title": "Acoustic Sunset", "tempo_bpm": 85, "energy": 0.25},
    {"song_title": "Lo-Fi Midnight", "tempo_bpm": 90, "energy": 0.30},
    {"song_title": "Ambient Meditation", "tempo_bpm": 70, "energy": 0.15},
    {"song_title": "Pop Dance Anthem", "tempo_bpm": 124, "energy": 0.85},
    {"song_title": "EDM Festival Banger", "tempo_bpm": 128, "energy": 0.95},
    {"song_title": "Synthwave Drive", "tempo_bpm": 120, "energy": 0.80},
    {"song_title": "Heavy Metal Thunder", "tempo_bpm": 155, "energy": 0.90},
    {"song_title": "Punk Rock Rush", "tempo_bpm": 160, "energy": 0.92}
]

def run_song_clustering():
    df = pd.DataFrame(songs_data)
    song_labels = df['song_title'].tolist()
    X = df[['tempo_bpm', 'energy']].values
    
    # Feature Scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Compute Linkage Matrix using Ward's method
    Z = linkage(X_scaled, method='ward')
    
    print("=" * 75)
    print("SESSION 05 - TASK 2: Agglomerative Hierarchical Clustering (8 Songs)")
    print("=" * 75)
    print("Song Dataset Summary:")
    print(df.to_string(index=False))
    print("-" * 75)
    print("\nScipy Ward Linkage Matrix Z (First 4 merges):")
    print(Z)

    # Plot Dendrogram
    plt.figure(figsize=(9, 6))
    dendrogram(Z, labels=song_labels, leaf_rotation=45, leaf_font_size=10, color_threshold=1.5)
    
    plt.title("Hierarchical Clustering Dendrogram of 8 Songs (Tempo & Energy)", fontsize=12, fontweight='bold')
    plt.xlabel("Song Title", fontsize=11)
    plt.ylabel("Euclidean Distance (Ward Linkage Height)", fontsize=11)
    plt.grid(True, axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig("Session_05/task_2_song_dendrogram.png", dpi=300)
    print("\nSaved plot visualization to Session_05/task_2_song_dendrogram.png")
    plt.close()

if __name__ == "__main__":
    run_song_clustering()
