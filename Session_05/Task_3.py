"""
Session 05 - Task 3
Hierarchical Linkage Comparison (Single, Complete, Average) on 8 Songs Dataset.
Plots 3 dendrograms side-by-side and summarizes structural observations.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.preprocessing import StandardScaler

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

def compare_linkages():
    df = pd.DataFrame(songs_data)
    song_labels = df['song_title'].tolist()
    X = df[['tempo_bpm', 'energy']].values
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    linkage_methods = ['single', 'complete', 'average']
    
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))
    
    print("=" * 85)
    print("SESSION 05 - TASK 3: Hierarchical Linkage Comparison on Songs Dataset")
    print("=" * 85)
    
    for idx, method in enumerate(linkage_methods):
        Z = linkage(X_scaled, method=method)
        
        ax = axes[idx]
        dendrogram(Z, labels=song_labels, leaf_rotation=55, leaf_font_size=9, ax=ax)
        ax.set_title(f"Linkage Method: '{method.capitalize()}'", fontsize=12, fontweight='bold')
        ax.set_ylabel("Distance (Merge Height)", fontsize=10)
        ax.grid(True, axis='y', linestyle='--', alpha=0.5)
        
        print(f"• Method '{method.capitalize()}': Max merge height = {Z[-1, 2]:.4f}")

    plt.tight_layout()
    plt.savefig("Session_05/task_3_linkage_comparison.png", dpi=300)
    print("\nSaved 3-panel dendrogram plot to Session_05/task_3_linkage_comparison.png")
    plt.close()

    print("\n" + "=" * 85)
    print("STRUCTURAL OBSERVATIONS & SUMMARY:")
    print("=" * 85)
    print("""
1. Single Linkage (Minimum Distance):
   - Measures distance between the closest pair of points across two clusters.
   - Susceptible to the 'chaining effect' where loose sequences of songs merge prematurely at low heights.

2. Complete Linkage (Maximum Distance):
   - Measures distance between the furthest pair of points across two clusters.
   - Avoids chaining and enforces compact, tightly-bounded clusters, resulting in higher top-level merge heights.

3. Average Linkage (Mean Pairwise Distance):
   - Measures the average distance between all pairs of points across two clusters.
   - Provides a balanced compromise between single and complete linkage, smoothing out noise while maintaining cluster integrity.
""")

if __name__ == "__main__":
    compare_linkages()
