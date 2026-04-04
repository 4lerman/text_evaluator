import spacy
from app.schemas.responses import LLMVerdict, HighlightedSentence, HighlightedToken

# Assertive nouns in English, Russian (lemmatised), and Kazakh.
# spaCy lemmatises each token before lookup, so Russian inflected forms
# (e.g. "стратегии", "стратегию") will resolve to "стратегия".
ASSERTIVE_NOUNS: set[str] = {
    # ── English ──────────────────────────────────────────────────────────────
    "strategy", "leadership", "initiative", "impact", "goal",
    "vision", "growth", "innovation", "resilience", "partnership",
    "decision", "opportunity", "integrity", "discipline", "insight",
    "execution", "accountability", "service", "collaboration",
    "determination",
    # ── Russian (lemma / base forms) ─────────────────────────────────────────
    "стратегия",          # strategy
    "лидерство",          # leadership
    "инициатива",         # initiative
    "влияние",            # impact
    "цель",               # goal
    "видение",            # vision
    "рост",               # growth
    "инновация",          # innovation
    "стойкость",          # resilience
    "партнёрство",        # partnership
    "решение",            # decision
    "возможность",        # opportunity
    "честность",          # integrity
    "дисциплина",         # discipline
    "проницательность",   # insight
    "исполнение",         # execution
    "ответственность",    # accountability
    "служение",           # service
    "сотрудничество",     # collaboration
    "решимость",          # determination
    # ── Kazakh (base forms) ───────────────────────────────────────────────────
    "көшбасшылық",        # leadership
    "бастама",            # initiative
    "ықпал",              # impact
    "мақсат",             # goal
    "көзқарас",           # vision
    "өсу",                # growth
    "төзімділік",         # resilience
    "серіктестік",        # partnership
    "шешім",              # decision
    "мүмкіндік",          # opportunity
    "адалдық",            # integrity
    "тәртіп",             # discipline
    "түсінік",            # insight
    "орындау",            # execution
    "жауапкершілік",      # accountability
    "қызмет",             # service
    "ынтымақтастық",      # collaboration
    "табандылық",         # determination
    # Note: "стратегия" and "инновация" appear in both Russian and Kazakh but
    # are already covered by the Russian section above (sets deduplicate).
}

def highlight_sentence(
    text: str,
    verdict: LLMVerdict,
    nlp: spacy.Language,
) -> HighlightedSentence:
    """
    Run POS tagging to extract highlighted tokens from a confirmed sentence.
    
    Args:
        text (str): The sentence text.
        verdict (LLMVerdict): The LLM verdict mapping for this sentence.
        nlp (spacy.Language): Loaded spaCy NLP model.
        
    Returns:
        HighlightedSentence: The sentence payload with structured highlights.
    """
    doc = nlp(text)
    highlights = []
    
    for token in doc:
        if token.pos_ == "VERB":
            highlights.append(HighlightedToken(
                token=token.text, 
                pos_category="ACTION_VERB",
                start=token.idx,
                end=token.idx + len(token.text)
            ))
        elif token.pos_ == "ADV":
            highlights.append(HighlightedToken(
                token=token.text, 
                pos_category="ADVERB",
                start=token.idx,
                end=token.idx + len(token.text)
            ))
        elif token.pos_ == "NOUN" and token.lemma_.lower() in ASSERTIVE_NOUNS:
            highlights.append(HighlightedToken(
                token=token.text, 
                pos_category="ASSERTIVE_NOUN",
                start=token.idx,
                end=token.idx + len(token.text)
            ))
            
    # Add the specific evidence quote from LLM if available
    if verdict.evidence_quote:
        eq = verdict.evidence_quote.strip().strip('"')
        start_idx = text.find(eq)
        if start_idx != -1:
            highlights.append(HighlightedToken(
                token=eq,
                pos_category="EVIDENCE_QUOTE",
                start=start_idx,
                end=start_idx + len(eq)
            ))
            
    return HighlightedSentence(
        text=text,
        value_code=verdict.value_code,
        value_name=verdict.value_name,
        reasoning=verdict.reasoning,
        score=verdict.score,
        highlights=highlights,
    )
