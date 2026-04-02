import spacy

def chunk_text(text: str, nlp: spacy.Language) -> list[str]:
    """
    Split text into valid sentence chunks using spaCy.
    
    Args:
        text (str): The raw text to process.
        nlp (spacy.Language): A loaded spaCy model.
        
    Returns:
        list[str]: A list of cleaned, valid sentences.
    """
    doc = nlp(text)
    chunks = []
    
    for sent in doc.sents:
        clean_sent = sent.text.strip()
        if not clean_sent:
            continue
            
        # Count words (tokens that are not whitespace or pure punctuation)
        words = [token for token in sent if not token.is_space and not token.is_punct]
        
        if len(words) >= 5:
            chunks.append(clean_sent)
            
    return chunks
