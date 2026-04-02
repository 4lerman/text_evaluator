import pytest
import spacy
from sentence_transformers import SentenceTransformer
from fastapi.testclient import TestClient

from app.main import app
from app.config import config

@pytest.fixture(scope="session")
def sample_text() -> str:
    return (
        "I consistently regulated my emotions under extreme pressure. "
        "The weather is nice today. "
        "I consistently led the strategic initiative to drive growth."
    )

@pytest.fixture(scope="session")
def nlp() -> spacy.Language:
    """Load multilingual spaCy model once for tests (sentence segmentation, handles RU + KZ)."""
    try:
        return spacy.load(config.SPACY_MODEL)
    except OSError:
        import xx_sent_ud_sm
        return xx_sent_ud_sm.load()

@pytest.fixture(scope="session")
def pos_nlp() -> spacy.Language:
    """Load English spaCy model once for POS tagging."""
    return spacy.load(config.POS_MODEL)

@pytest.fixture(scope="session")
def ru_pos_nlp() -> spacy.Language:
    """Load Russian spaCy model once for RU/KZ POS tagging."""
    return spacy.load(config.RU_POS_MODEL)

@pytest.fixture(scope="session")
def embedding_model() -> SentenceTransformer:
    """Load embedding model once for tests."""
    return SentenceTransformer(config.EMBEDDING_MODEL_NAME)

@pytest.fixture
def test_client(nlp, pos_nlp, ru_pos_nlp, embedding_model) -> TestClient:
    """Test client with loaded models in state."""
    app.state.nlp = nlp
    app.state.pos_nlp = pos_nlp
    app.state.ru_pos_nlp = ru_pos_nlp
    app.state.embedding_model = embedding_model
    with TestClient(app) as client:
        yield client
