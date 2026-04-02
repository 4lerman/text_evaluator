import spacy

# STT transcripts often lack punctuation, producing run-on "sentences".
# Any spaCy sentence longer than this gets split into sub-chunks.
_MAX_WORDS = 40
_TARGET_CHUNK_WORDS = 25


def _split_long_sentence(tokens: list) -> list[list]:
    """Split a token list into sub-chunks of ~_TARGET_CHUNK_WORDS words.

    Args:
        tokens: spaCy token list for one sentence.

    Returns:
        List of token sublists, each roughly _TARGET_CHUNK_WORDS words long.
    """
    words = [t for t in tokens if not t.is_space]
    sublists = []
    i = 0
    while i < len(words):
        sublists.append(words[i : i + _TARGET_CHUNK_WORDS])
        i += _TARGET_CHUNK_WORDS
    return sublists


def chunk_text(text: str, nlp: spacy.Language) -> list[str]:
    """Split text into valid sentence chunks using spaCy.

    Long sentences (common in unpunctuated STT output) are split into
    sub-chunks of ~_TARGET_CHUNK_WORDS words so Prometheus receives
    focused, scoreable fragments rather than entire paragraphs.

    Args:
        text: The raw text to process.
        nlp: A loaded spaCy model.

    Returns:
        List of cleaned, valid sentence strings (>= 5 content words each).
    """
    doc = nlp(text)
    chunks = []

    for sent in doc.sents:
        words = [t for t in sent if not t.is_space and not t.is_punct]

        if len(words) < 5:
            continue

        if len(words) <= _MAX_WORDS:
            chunks.append(sent.text.strip())
        else:
            # STT run-on: split into focused sub-chunks
            for sublist in _split_long_sentence(list(sent)):
                sub_words = [t for t in sublist if not t.is_punct]
                if len(sub_words) >= 5:
                    chunks.append(" ".join(t.text for t in sublist).strip())

    return chunks
