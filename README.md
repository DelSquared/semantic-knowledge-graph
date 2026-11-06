# Semantic Knowledge Graph

A web application that visualizes a collection of text notes as an interactive 3D knowledge graph. Notes are embedded into a high-dimensional semantic space, projected into 3D, clustered, and displayed as coloured nodes.

## Algorithm

### 1. Embedding

Each note is encoded into a fixed-length vector using a sentence-transformer model (`all-mpnet-base-v2` by default). The model maps semantically similar text to nearby vectors in the embedding space, where similarity is measured by cosine distance.

Embeddings are cached: a `.npy` file per note stores the vector alongside an MD5 hash of the note content. On rebuild, cached embeddings are reused if the content hasn't changed.

### 2. 3D Layout via PCA

Rather than a force-directed layout, the embedding vectors are projected into 3D using Principal Component Analysis (PCA). The first 3 principal components become the x, y, z coordinates of each node. This preserves the semantic neighbourhood structure directly: notes that are close in the high-dimensional embedding space will be close in the 3D view.

### 3. Clustering

K-means clustering is applied to the embedding vectors (k = `sqrt(n)` clusters, capped at 8). Each cluster is assigned a distinct colour. Node colour, glow, and edge colour (if edges are enabled) reflect cluster membership.

### 4. Visualization

The 3D scene is rendered with Three.js. Nodes are spheres coloured by cluster. Labels are rendered as sprites above nodes. The user can orbit with left-drag, zoom with scroll, and pan with middle-drag.

## Project Structure

```
app.py          FastAPI server
config.py       Configuration (model, paths, thresholds)
graph.py        Graph engine (embedding, PCA layout, k-means, caching)
requirements.txt    Python dependencies
run_server.bat      Server launcher
notes/          Input text files (one per node)
static/
  index.html    Entry page
  css/style.css Styling
  js/
    app.js      ES module entry point
    main.js     3D visualization logic
    OrbitControls.js  Camera controls (from three.js)
    three.module.js   Three.js ES module
```

## Layout Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `EMBEDDING_MODEL` | `all-mpnet-base-v2` | Sentence-transformer model |
| k-means k | `round(sqrt(n))`, capped at 8 | Number of colour clusters |
