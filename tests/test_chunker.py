from app.pipeline.chunker import chunk_text

def test_chunk_text_valid_sentences(nlp):
    text = (
        "This is the first valid sentence with enough words. "
        "Here is a second sentence that meets the length criteria. "
        "And finally, a third sentence to ensure we get multiple items."
    )
    
    chunks = chunk_text(text, nlp)
    
    assert len(chunks) == 3
    assert chunks[0] == "This is the first valid sentence with enough words."
    assert chunks[1] == "Here is a second sentence that meets the length criteria."
    assert chunks[2] == "And finally, a third sentence to ensure we get multiple items."

def test_chunk_text_filters_short_fragments(nlp):
    text = (
        "Yes. I agree. No way. "
        "I consistently regulated my emotions under extreme pressure. "
        "Wow. "
    )
    
    chunks = chunk_text(text, nlp)
    
    assert len(chunks) == 1
    assert chunks[0] == "I consistently regulated my emotions under extreme pressure."
