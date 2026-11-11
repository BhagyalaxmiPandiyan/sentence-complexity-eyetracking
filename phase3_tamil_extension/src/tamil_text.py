"""
Shared Tamil text helpers used by the Phase 3 pipeline.

repair_pdf_text(): fixes the font artifacts in the Tamil Nadu textbook PDFs,
where text extraction doubles pulli / vowel signs / consonants:
    அடித்்தளம்   -> அடித்தளம்      (doubled pulli)
    தொொடக்்கமும் -> தொடக்கமும்     (doubled vowel sign + pulli)
    க்்ககான      -> க்கான          (doubled pulli + doubled consonant)
Measured on the Class 6 book: 55.3% -> 76.5% of words are real Tamil words
(experiments/pilot_textbook_extraction.py).
"""
import re
import unicodedata

PULLI = "்"                                  # ்  (virama: kills the inherent vowel)
CONSONANT = "[க-ஹ]"                     # க ... ஹ
VOWEL_SIGN = "[ா-்ௗ]"              # ா ி ீ ு ூ ெ ே ை ொ ோ ௌ ் ௗ
TAMIL_WORD = re.compile("[஀-௿]+")

_DOUBLE_PULLI_CONS = re.compile(PULLI + PULLI + "(" + CONSONANT + ")\\1")
_DOUBLE_PULLI = re.compile(PULLI + PULLI)
_DOUBLE_SIGN = re.compile("(" + VOWEL_SIGN + ")\\1")


def repair_pdf_text(text):
    # NFC first: ொ may be stored decomposed as ெ + ா, which would look like a doubled sign
    text = unicodedata.normalize("NFC", text)
    text = _DOUBLE_PULLI_CONS.sub(lambda m: PULLI + m.group(1), text)
    text = _DOUBLE_PULLI.sub(PULLI, text)
    return _DOUBLE_SIGN.sub(lambda m: m.group(1), text)


def merge_split_words(sentence, max_parts=4):
    """Re-join words that justified PDF text split with spaces:
        "கிரா மத்தில்"      -> "கிராமத்தில்"
        "இருக் கை யி லேயே"  -> "இருக்கையிலேயே"
    Greedy: at each position, join the LONGEST run of 2..max_parts tokens whose
    concatenation is a known Tamil word while at least one part is not, so two
    real words are never glued together."""
    from wordfreq import zipf_frequency
    known = lambda w: zipf_frequency(w, "ta") > 0
    toks, out, i = sentence.split(), [], 0
    while i < len(toks):
        for j in range(min(len(toks), i + max_parts), i + 1, -1):
            parts = toks[i:j]
            if all(TAMIL_WORD.fullmatch(p) for p in parts) and not all(map(known, parts)) \
                    and known("".join(parts)):
                out.append("".join(parts)); i = j
                break
        else:
            out.append(toks[i]); i += 1
    return " ".join(out)


def tamil_words(text):
    return TAMIL_WORD.findall(text)
