from app.pipeline.highlighter import highlight_sentence
from app.schemas.responses import LLMVerdict

def test_highlight_sentence_extracts_tokens(pos_nlp):
    verdict = LLMVerdict(
        text="I consistently led the strategic initiative.",
        value_code="R",
        value_name="Responsible Innovation",
        confirmed=True,
        reasoning="Shows strategy and execution"
    )
    
    # Needs a real verb like "led", adverb like "consistently", noun in ASSERTIVE_NOUNS like "initiative"
    result = highlight_sentence("I consistently led the strategic initiative", verdict, pos_nlp)
    
    assert result.value_code == "R"
    
    categories = [h.pos_category for h in result.highlights]
    tokens_text = [h.token for h in result.highlights]
    
    assert "ADVERB" in categories
    assert "consistently" in tokens_text
    
    assert "ACTION_VERB" in categories
    assert "led" in tokens_text
    
    assert "ASSERTIVE_NOUN" in categories
    assert "initiative" in tokens_text

def test_highlight_sentence_ignores_common_nouns(pos_nlp):
    verdict = LLMVerdict(
        text="The cat sat on the mat.",
        value_code="I",
        value_name="Insightful Vision",
        confirmed=True,
        reasoning="Irrelevant"
    )
    
    result = highlight_sentence("The cat sat on the mat.", verdict, pos_nlp)
    
    categories = [h.pos_category for h in result.highlights]
    # No ASSERTIVE_NOUN should be found
    assert "ASSERTIVE_NOUN" not in categories
    # The verb 'sat' is present
    assert "ACTION_VERB" in categories
