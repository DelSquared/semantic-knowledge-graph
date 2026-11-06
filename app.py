"""FastAPI server for the semantic knowledge graph web interface."""

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from config import STATIC_DIR
from graph import SemanticGraph

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# Will be populated by the lifespan
_graph: SemanticGraph | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _graph
    _graph = SemanticGraph()
    _graph.build_graph()
    num_nodes = len(_graph.nodes)
    num_edges = len(_graph.edges)
    print(f"Graph built with {num_nodes} nodes and {num_edges} edges.")
    yield


app = FastAPI(title="Semantic Knowledge Graph", lifespan=lifespan)

# CORS middleware -- allows cross-origin fetch() from any origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", response_class=HTMLResponse)
async def index() -> str:
    html_path = STATIC_DIR / "index.html"
    return HTMLResponse(content=html_path.read_text(encoding="utf-8"))


@app.get("/api/graph")
async def get_graph() -> dict:
    if _graph is None:
        raise HTTPException(status_code=503, detail="Graph not yet built")
    return _graph.get_graph_data()


@app.get("/api/node/{node_id}")
async def get_node(node_id: str) -> dict:
    if _graph is None:
        raise HTTPException(status_code=503, detail="Graph not yet built")
    node = _graph.get_node(node_id)
    if node is None:
        raise HTTPException(status_code=404, detail=f"Node '{node_id}' not found")
    return node


@app.get("/api/stats")
async def get_stats() -> dict:
    if _graph is None:
        raise HTTPException(status_code=503, detail="Graph not yet built")
    return {
        "num_nodes": len(_graph.nodes),
        "num_edges": len(_graph.edges),
        "notes_dir": str(_graph.notes_dir),
    }


# Mount static files (JS, CSS, etc.)
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


if __name__ == "__main__":
    import uvicorn
    from config import HOST, PORT

    log.info("Starting server on %s:%s", HOST, PORT)
    log.info("Serving static files from %s", STATIC_DIR)
    uvicorn.run(
        app,
        host=HOST,
        port=PORT,
        log_level="info",
        access_log=True,
    )
