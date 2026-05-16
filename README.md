# Image Network Analysis Using Region Adjacency Graphs

A Python project that applies **discrete mathematics and graph theory** to analyse the visual structure of a sunset image. The image is segmented into superpixels using the SLIC algorithm, then modelled as a Region Adjacency Graph (RAG) where regions are nodes and edges connect neighbouring regions weighted by colour distance.

Built as a group project for a Discrete Mathematics course at the American University of Ras Al Khaimah.

---

## What It Does

- Segments a colour image into ~200 superpixels using SLIC
- Constructs a Region Adjacency Graph (RAG) for both the **colour** and **grayscale** versions of the image
- Computes graph metrics: degree centrality, community structure, modularity, and von Neumann entropy
- Generates visualisations: segmented image, RAG overlay, community map, top-centrality node highlights
- Compares colour vs. grayscale graphs to understand how colour affects network structure

---

## Key Findings

| Metric | Colour | Grayscale |
|---|---|---|
| Nodes | 197 | 198 |
| Edges | 536 | 539 |
| Communities | 5 | 5 |
| Von Neumann Entropy | 7.152 | 6.999 |
| Modularity Q | 0.614 | 0.632 |

- **Colour increases structural complexity** — higher von Neumann entropy means more variation in edge weights and neighbourhood patterns
- **Large-scale segmentation is preserved** — both graphs produce the same 5 communities (sky, horizon, foreground, etc.)
- **Grayscale sharpens intensity boundaries** — slightly higher modularity means more internally cohesive communities

---

## Project Structure

```
image-network-analysis-RAG/
├── main.py           # Full analysis pipeline
├── sunset.jpeg       # Input image
├── README.md
└── figures/          # Auto-generated output folder (not tracked by git)
    ├── original_segmented_meancolor.png
    ├── original_graph_overlay.png
    ├── original_communities.png
    ├── original_top_nodes_overlay.png
    ├── original_top_nodes.csv
    ├── grayscale_segmented_meancolor.png
    ├── grayscale_graph_overlay.png
    ├── grayscale_communities.png
    ├── grayscale_top_nodes_overlay.png
    └── grayscale_top_nodes.csv
```

---

## Requirements

Install dependencies with:

```bash
pip install opencv-python scikit-image networkx matplotlib numpy pandas
```

| Library | Purpose |
|---|---|
| `opencv-python` | Image loading and colour space conversion |
| `scikit-image` | SLIC segmentation and RAG construction |
| `networkx` | Graph construction and metric computation |
| `matplotlib` | Visualisation and saving figures |
| `numpy` | Matrix operations and eigenvalue computation |
| `pandas` | Top nodes summary table and CSV export |

---

## How to Run

1. Make sure `sunset.jpeg` is in the same folder as `main.py`
2. Run:

```bash
python main.py
```

3. All output images and CSVs will be saved to a `figures/` folder that gets created automatically

---

## Output

**Console** — prints metrics for both colour and grayscale graphs plus a side-by-side comparison table with auto-interpretation.

**figures/ folder** — saves 10 files total:
- Superpixel segmentation images
- RAG overlay images
- Community structure maps
- Top 5 high-centrality node overlays
- CSV files listing top node details (degree centrality, neighbours, size, mean colour)
