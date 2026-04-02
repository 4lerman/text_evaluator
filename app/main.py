import logging
from contextlib import asynccontextmanager
from pathlib import Path

import spacy
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from sentence_transformers import SentenceTransformer

from app.config import config
from app.routers.evaluate import router as evaluate_router

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load NLP models at startup and clean up at shutdown."""
    logger.info(f"Loading spaCy model '{config.SPACY_MODEL}'...")
    app.state.nlp = spacy.load(config.SPACY_MODEL)

    logger.info(f"Loading spaCy POS model '{config.POS_MODEL}'...")
    app.state.pos_nlp = spacy.load(config.POS_MODEL)

    logger.info(f"Loading spaCy RU/KZ POS model '{config.RU_POS_MODEL}'...")
    app.state.ru_pos_nlp = spacy.load(config.RU_POS_MODEL)

    logger.info(f"Loading sentence-transformer '{config.EMBEDDING_MODEL_NAME}'...")
    app.state.embedding_model = SentenceTransformer(config.EMBEDDING_MODEL_NAME)

    yield

    logger.info("Shutting down models...")
    del app.state.nlp
    del app.state.pos_nlp
    del app.state.ru_pos_nlp
    del app.state.embedding_model


app = FastAPI(title="D.R.I.V.E. Evaluator", lifespan=lifespan)

# CORS — allow all origins during development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST", "GET", "OPTIONS"],
    allow_headers=["*"],
)

app.include_router(evaluate_router)

# Serve the frontend SPA
FRONTEND_DIR = Path(__file__).parent.parent / "frontend"
if FRONTEND_DIR.exists():
    app.mount("/app", StaticFiles(directory=FRONTEND_DIR), name="frontend")

    @app.get("/", include_in_schema=False)
    async def serve_index():
        return FileResponse(FRONTEND_DIR / "index.html")


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "ok"}
