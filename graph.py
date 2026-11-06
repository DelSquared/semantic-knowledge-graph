"""Semantic knowledge graph engine.

Nodes represent text files containing notes. Edges are weighted by
the cosine similarity of their content embeddings. A 3D PCA layout
positions nodes by semantic similarity, and k-means clustering groups
nodes into visually distinct clusters.
"""

import hashlib
import json
import logging
import threading
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from config import (
    EMBEDDING_MODEL,
    NOTES_DIR,
    SIMILARITY_THRESHOLD,
)
from sentence_transformers import SentenceTransformer

log = logging.getLogger(__name__)


@dataclass
class Node:
    """A node in the semantic graph, representing a text file."""

    id: str
    file_path: Path
    content: str
    embedding: np.ndarray | None = field(default=None, repr=False)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "file_path": str(self.file_path),
            "content_preview": self.content[:200] + ("..." if len(self.content) > 200 else ""),
            "content": self.content,
        }


@dataclass
class Edge:
    """An edge in the semantic graph with a similarity weight."""

    source: str
    target: str
    weight: float


class SemanticGraph:
    """Manages the semantic graph: loading notes, computing embeddings,
    building weighted edges, and running 3D force-directed layout."""

    def __init__(self, notes_dir: Path = NOTES_DIR):
        self.notes_dir = notes_dir
        self.nodes: list[Node] = []
        self.edges: list[Edge] = []
        self.node_dict: dict[str, Node] = {}
        self._model: SentenceTransformer | None = None
        self._positions: dict[str, list[float]] = {}  # node_id -> [x, y, z]
        self._cluster_ids: np.ndarray = np.array([])  # node index -> cluster label
        self._positions_lock = threading.Lock()
        self._load_notes()

    def _load_model(self) -> SentenceTransformer:
        if self._model is None:
            log.info("Loading embedding model '%s' from HuggingFace...", EMBEDDING_MODEL)
            self._model = SentenceTransformer(EMBEDDING_MODEL)
            log.info("Embedding model loaded successfully (vocab size=%d).",
                     self._model.get_sentence_embedding_dimension())
        return self._model

    def _load_notes(self) -> None:
        """Load all .txt files from the notes directory."""
        if not self.notes_dir.exists():
            log.info("Notes directory does not exist yet: %s", self.notes_dir)
            self.notes_dir.mkdir(parents=True, exist_ok=True)
        txt_files = sorted(self.notes_dir.glob("*.txt"))
        total_chars = 0
        for txt_file in txt_files:
            file_id = txt_file.stem
            content = txt_file.read_text(encoding="utf-8")
            total_chars += len(content)
            node = Node(id=file_id, file_path=txt_file, content=content)
            self.nodes.append(node)
            self.node_dict[node.id] = node
        log.info("Loaded %d notes from %s (total ~%d chars).",
                 len(self.nodes), self.notes_dir, total_chars)

    def _content_hash(self, content: str) -> str:
        """Fast non-cryptographic hash of note content."""
        return hashlib.md5(content.encode("utf-8")).hexdigest()

    def _load_cached_embeddings(self) -> dict[str, np.ndarray] | None:
        """Try to load embeddings from cache. Returns None if cache is stale/missing."""
        cache_dir = Path(".cache") / "embeddings"
        cache: dict[str, np.ndarray] = {}
        for node in self.nodes:
            cached_path = cache_dir / f"{node.id}.npy"
            if not cached_path.exists():
                return None
            cached_hash = (cache_dir / f"{node.id}.hash").read_text()
            current_hash = self._content_hash(node.content)
            if cached_hash != current_hash:
                return None
            cache[node.id] = np.load(cached_path, allow_pickle=False)
        return cache

    def _save_cached_embeddings(self, cache: dict[str, np.ndarray]) -> None:
        """Persist embeddings and their content hashes to the cache directory."""
        cache_dir = Path(".cache") / "embeddings"
        cache_dir.mkdir(parents=True, exist_ok=True)
        for node_id, emb in cache.items():
            np.save(cache_dir / f"{node_id}.npy", emb)
            (cache_dir / f"{node_id}.hash").write_text(self._content_hash(self.node_dict[node_id].content))

    def _compute_embeddings(self) -> None:
        """Compute sentence embeddings, with caching by content hash."""
        cache = self._load_cached_embeddings()
        model = self._load_model()
        texts = [n.content for n in self.nodes]

        if cache is not None:
            hit = sum(1 for n in self.nodes if n.id in cache)
            log.info("Loaded %d/%d embeddings from cache.", hit, len(self.nodes))
            for i, (node, emb) in enumerate(zip(self.nodes, [cache[n.id] for n in self.nodes])):
                node.embedding = emb
                log.debug("  [%d/%d] %s (cached)", i + 1, len(self.nodes), node.id)
        else:
            # Compute missing embeddings and merge with cache hits
            log.info("Computing embeddings for %d texts with model '%s'...",
                     len(texts), EMBEDDING_MODEL)
            embeddings = model.encode(texts, show_progress_bar=True,
                                      normalize_embeddings=True)
            cache = {}
            for i, (node, emb) in enumerate(zip(self.nodes, embeddings)):
                node.embedding = emb
                cache[node.id] = emb
                log.debug("  [%d/%d] %s", i + 1, len(self.nodes), node.id)
            self._save_cached_embeddings(cache)
            log.info("All %d embeddings computed and cached.", len(self.nodes))

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))

    def build_graph(self) -> None:
        """Compute embeddings, build edges based on similarity, and run layout."""
        log.info("Building graph from %s...", self.notes_dir)

        self._compute_embeddings()

        log.info("Computing pairwise similarities (threshold=%.2f)...", SIMILARITY_THRESHOLD)
        n = len(self.nodes)
        edge_count = 0
        # Only compute upper triangle to avoid duplicates
        embeddings = np.array([n.embedding for n in self.nodes])
        for i in range(n):
            for j in range(i + 1, n):
                sim = self._cosine_similarity(embeddings[i], embeddings[j])
                if sim >= SIMILARITY_THRESHOLD:
                    self.edges.append(Edge(
                        source=self.nodes[i].id,
                        target=self.nodes[j].id,
                        weight=sim,
                    ))
                    edge_count += 1
        log.info("Created %d edges (threshold=%.2f).", edge_count, SIMILARITY_THRESHOLD)
        embeddings = np.array([n.embedding for n in self.nodes])
        positions = self._pca_layout(embeddings)
        cluster_ids = self._kmeans_cluster(embeddings)
        with self._positions_lock:
            self._positions = positions
            self._cluster_ids = cluster_ids
        log.info("Graph built: %d nodes, %d edges, %d clusters.",
                 len(self.nodes), len(self.edges), len(set(cluster_ids)))

    def _pca_layout(self, embeddings: np.ndarray) -> dict[str, list[float]]:
        """Dimensionality reduction via PCA to map embedding vectors into 3D.

        Each node's position is the first 3 principal components of its
        embedding vector. This preserves the semantic structure of the
        high-dimensional embedding space directly, rather than using
        a binary edge-weight-based force simulation.
        """
        from sklearn.decomposition import PCA

        log.info("Reducing %d embeddings (dim=%d) to 3D via PCA...",
                 len(self.nodes), embeddings.shape[1])

        pca = PCA(n_components=3)
        reduced = pca.fit_transform(embeddings)

        # Center the layout around origin
        reduced -= reduced.mean(axis=0)

        # Scale to a reasonable viewing radius
        scale = 50.0 / np.max(np.abs(reduced))
        reduced *= scale

        # Optional: flip so the densest cluster is in the positive octant
        if reduced.min() < 0:
            reduced -= reduced.min(axis=0)

        positions: dict[str, list[float]] = {}
        for i, node in enumerate(self.nodes):
            positions[node.id] = reduced[i].tolist()

        var_explained = pca.explained_variance_ratio_.sum() * 100
        log.info("PCA: %.1f%% variance explained by 3 components.", var_explained)
        log.info("Graph layout: %d nodes placed in 3D.", len(self.nodes))

        return positions

    def _kmeans_cluster(self, embeddings: np.ndarray) -> np.ndarray:
        """Cluster nodes into groups using k-means on the embedding space.

        Returns an array of integer cluster labels (0..k-1) for each node,
        in the same order as self.nodes.
        """
        from sklearn.cluster import KMeans

        n = len(self.nodes)
        n_clusters = min(8, max(2, round(n ** 0.5)))
        log.info("Running k-means clustering (%d clusters, %d points)...",
                 n_clusters, n)

        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(embeddings)

        # Build label -> index mapping for reporting
        label_counts: dict[int, int] = {}
        for l in labels:
            label_counts[l] = label_counts.get(l, 0) + 1
        log.info("Clusters: " + ", ".join(
            f"{l}:{c}" for l, c in sorted(label_counts.items())))

        return labels

    def get_graph_data(self) -> dict:
        """Return the full graph as JSON-serializable data."""
        nodes_out = []
        positions_out = {}
        with self._positions_lock:
            positions_out = dict(self._positions)

        for i, node in enumerate(self.nodes):
            pos = positions_out.get(node.id, [0.0, 0.0, 0.0])
            nodes_out.append({
                **node.to_dict(),
                "position": pos,
                "cluster": int(self._cluster_ids[i]) if len(self._cluster_ids) > i else 0,
            })

        return {
            "nodes": nodes_out,
            "edges": [
                {"source": e.source, "target": e.target, "weight": e.weight}
                for e in self.edges
            ],
        }

    def get_node(self, node_id: str) -> dict | None:
        """Get a single node by ID."""
        node = self.node_dict.get(node_id)
        if node is None:
            return None
        idx = self.node_dict.get(node_id)
        cluster = int(self._cluster_ids[idx]) if idx is not None and len(self._cluster_ids) > idx else 0
        with self._positions_lock:
            pos = self._positions.get(node_id, [0.0, 0.0, 0.0])
        return {**node.to_dict(), "position": pos, "cluster": cluster}
