# Discrete Project 1: Network Analysis of Visual Structure in an Image
# Model A - Region Adjacency Graph (RAG)

                                                # Imported Libraries

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
try:
    from skimage.future import graph
except ImportError:
    from skimage import graph

import cv2
import numpy as np
import networkx as nx
from skimage import segmentation, color
import pandas as pd
import os
import networkx.algorithms.community as nx_comm
np.random.seed(0)

                                                # Utility functions
def load_image(image_path, max_size=1000):
    """Load image from file and resize if needed."""
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
   
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]
    if max(h, w) > max_size:
        scale = max_size / max(h, w)
        img = cv2.resize(img, (int(w*scale), int(h*scale)))
    return img

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

                                                # Graph model functions
def build_rag_model(img, n_segments=200, compactness=30):
    labels = segmentation.slic(img, compactness=compactness, n_segments=n_segments, start_label=1)
    labels = labels.astype(int)
    rag = graph.rag_mean_color(img, labels, mode='distance')
    
    G = nx.Graph()
    for (n1, n2, data) in rag.edges(data=True):
        G.add_edge(n1, n2, weight=data['weight'])
    return G, labels

def analyze_graph(G):
    centrality = nx.degree_centrality(G)
    communities = list(nx.algorithms.community.greedy_modularity_communities(G))
    modularity_score = nx_comm.modularity(G, communities)
    A = nx.to_numpy_array(G)
    L = np.diag(np.sum(A, axis=1)) - A
    vals = np.linalg.eigvalsh(L)
    vals = vals[vals > 1e-12]
    probs = vals / np.sum(vals)
    vn_entropy = -np.sum(probs * np.log2(probs))
    return centrality, communities, vn_entropy, modularity_score

def save_results(G, centrality, communities, img, labels, figures_dir="figures", prefix=""):
    ensure_dir(figures_dir)
    
                                                    # Segmented image
    seg_img = color.label2rgb(labels, img, kind='avg')
    plt.imsave(os.path.join(figures_dir, f"{prefix}segmented_meancolor.png"), seg_img)
    
                                                    # Graph overlay
    plt.figure(figsize=(8,8))
    plt.imshow(seg_img)
    for (n1, n2) in G.edges():
        y1, x1 = np.mean(np.argwhere(labels==n1), axis=0)[:2]
        y2, x2 = np.mean(np.argwhere(labels==n2), axis=0)[:2]
        plt.plot([x1, x2], [y1, y2], 'k-', alpha=0.3)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, f"{prefix}graph_overlay.png"))
    plt.close()
    
                                                    # Communities overlay

                                                    
    comm_colors = np.zeros((labels.max()+1, 3))
    for idx, community in enumerate(communities):
        color_rgb = np.random.randint(0,255,3)/255
        for node in community:
            comm_colors[node] = color_rgb
    comm_img = np.zeros_like(img, dtype=float)
    for i in range(labels.shape[0]):
        for j in range(labels.shape[1]):
            comm_img[i,j] = comm_colors[labels[i,j]]
    plt.imsave(os.path.join(figures_dir, f"{prefix}communities.png"), comm_img)

def annotate_top_nodes(G, labels, img, figures_dir="figures", top_n=5, prefix=""):
    ensure_dir(figures_dir)
    centrality = nx.degree_centrality(G)
    top_nodes = sorted(centrality.items(), key=lambda x: x[1], reverse=True)[:top_n]
    
    csv_rows = []
    for node, cent in top_nodes:
        mask = (labels==node)
        size = np.sum(mask)
        neighbors = len(list(G.neighbors(node)))
        mean_color = np.mean(img[mask], axis=0).astype(int)
        csv_rows.append({
            "Node": node,
            "Degree Centrality": cent,
            "Neighbors": neighbors,
            "Size(px)": size,
            "Mean Color(RGB)": tuple(mean_color)
        })
    df = pd.DataFrame(csv_rows)
    csv_path = os.path.join(figures_dir, f"{prefix}top_nodes.csv")
    df.to_csv(csv_path, index=False)
    
    overlay = img.copy()
    for node, _ in top_nodes:
        mask = (labels==node)
        overlay[mask] = [255,0,0]
    overlay_path = os.path.join(figures_dir, f"{prefix}top_nodes_overlay.png")
    plt.imsave(overlay_path, overlay)
    
    print(f"\nSaved top nodes CSV: {csv_path}")
    print(f"Saved overlay: {overlay_path}\n")
    print(f"----------- Top nodes summary ({prefix}) -----------")
    print(df.to_string(index=False))

#----------------------------------------- Main execution -------------------------------------------------

if __name__ == "__main__":
    IMAGE_PATH = os.path.join(os.path.dirname(__file__), "sunset.jpeg") #  <---------------------- IMAGE PATH HERE
    if not os.path.isfile(IMAGE_PATH):
        raise FileNotFoundError(f"Please put the input image at this path: {IMAGE_PATH}")
    FIGURES_DIR = "figures"
    
                                                            # Original Image Analysis
    print("Loading original image...")
    img = load_image(IMAGE_PATH, max_size=1000)
    print("Building RAG for original image...")
    G, labels = build_rag_model(img)
    print("Analyzing graph metrics...")
    centrality, communities, vn_entropy, modularity_score = analyze_graph(G)
    save_results(G, centrality, communities, img, labels, FIGURES_DIR, prefix="original")
    print("\n------------ Original Image Metrics ------------")
    print(f"n-nodes: {G.number_of_nodes()}, n-edges: {G.number_of_edges()}, n-communities: {len(communities)}, vn-entropy: {vn_entropy:.3f}, modularity: {modularity_score:.4f} ")
    annotate_top_nodes(G, labels, img, FIGURES_DIR, prefix="original")
    
                                                             # Grayscale Image Analysis
    print("\nLoading grayscale image...")
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    gray_rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    print("Building RAG for grayscale image...")
    G_gray, labels_gray = build_rag_model(gray_rgb)
    print("Analyzing graph metrics...")
    centrality_gray, communities_gray, vn_entropy_gray, modularity_score_gray = analyze_graph(G_gray)
    save_results(G_gray, centrality_gray, communities_gray, gray_rgb, labels_gray, FIGURES_DIR, prefix="grayscale")
    print("\n--- Grayscale Image Metrics ---")
    print(f"n-nodes: {G_gray.number_of_nodes()}, n-edges: {G_gray.number_of_edges()}, n-communities: {len(communities_gray)}, vn-entropy: {vn_entropy_gray:.3f}, modularity: {modularity_score_gray:.4f} ")
    annotate_top_nodes(G_gray, labels_gray, gray_rgb, FIGURES_DIR, prefix="grayscale")
    
                                                            # Comparative Summary
    print("\n----------- Comparative Analysis -----------")
    print(f"{'Metric':<15} {'Original':<10} {'Grayscale':<10} {'Difference':<10}")
    print(f"{'Nodes':<15} {G.number_of_nodes():<10} {G_gray.number_of_nodes():<10} {G.number_of_nodes()-G_gray.number_of_nodes():<10}")
    print(f"{'Edges':<15} {G.number_of_edges():<10} {G_gray.number_of_edges():<10} {G.number_of_edges()-G_gray.number_of_edges():<10}")
    print(f"{'Communities':<15} {len(communities):<10} {len(communities_gray):<10} {len(communities)-len(communities_gray):<10}")
    print(f"{'VN Entropy':<15} {vn_entropy:<10.3f} {vn_entropy_gray:<10.3f} {vn_entropy-vn_entropy_gray:<10.3f}")
    print(f"{'Modularity':<15} {modularity_score:<10.3f} {modularity_score_gray:<10.3f} {modularity_score - modularity_score_gray:<10.3f}") 


    print("\nInterpretation:")  # VN Entropy interpretation
    if vn_entropy > vn_entropy_gray:
        print("- VN Entropy is higher in the original image → indicates greater visual complexity and color variation.")
    elif vn_entropy < vn_entropy_gray:
        print("- VN Entropy is higher in the grayscale image → structural or tonal complexity is stronger in grayscale.")
    else:
        print("- VN Entropy values are equal → overall visual complexity is similar in both versions.")
                                # Community count interpretation
    if len(communities) > len(communities_gray):
        print(f"- Original image has more communities ({len(communities)}) than grayscale ({len(communities_gray)}) → colors create more distinct regions.")
    elif len(communities) < len(communities_gray):
        print(f"- Grayscale image has more communities ({len(communities_gray)}) than original ({len(communities)}) → brightness contrast dominates structure.")
    else:
        print(f"- Both images have the same number of communities ({len(communities)}) → segmentation unaffected by color removal.")
                                # Modularity interpretation
    if modularity_score > modularity_score_gray:
        print(f"- Modularity is higher in the original image ({modularity_score:.3f} vs {modularity_score_gray:.3f}) → color-based regions are better separated.")
    elif modularity_score < modularity_score_gray:
        print(f"- Modularity is higher in the grayscale image ({modularity_score_gray:.3f} vs {modularity_score:.3f}) → communities are more distinct by intensity.")
    else:
        print(f"- Modularity is equal ({modularity_score:.3f}) → both images show similar structural grouping.")

