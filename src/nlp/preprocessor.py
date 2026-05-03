import re
import unicodedata

try:
    from underthesea import word_tokenize

    UNDERTHESEA_AVAILABLE = True
except ImportError:
    UNDERTHESEA_AVAILABLE = False


STOPWORDS = {
    "va",
    "cua",
    "trong",
    "theo",
    "nhu",
    "hay",
    "hoac",
    "vi",
    "do",
    "boi",
    "da",
    "se",
    "dang",
    "van",
    "con",
    "them",
    "bi",
    "ra",
    "vao",
    "len",
    "xuong",
    "lai",
    "di",
    "den",
    "rat",
    "qua",
    "kha",
    "hoi",
    "cang",
    "nhieu",
    "it",
    "chung",
    "moi",
    "ai",
}


class VietnamesePreprocessor:
    def __init__(self, use_word_segment: bool = True, remove_stopwords: bool = True):
        self.use_word_segment = use_word_segment and UNDERTHESEA_AVAILABLE
        self.remove_stopwords = remove_stopwords

    def normalize(self, text: str) -> str:
        text = text.lower().strip()
        text = re.sub(r"[^\w\s\u00C0-\u024F\u1E00-\u1EFF]", " ", text)
        text = re.sub(r"\s+", " ", text)
        return text

    def strip_accents(self, text: str) -> str:
        nfd = unicodedata.normalize("NFD", text)
        stripped = "".join(c for c in nfd if unicodedata.category(c) != "Mn")
        return unicodedata.normalize("NFC", stripped)

    def tokenize(self, text: str) -> list[str]:
        if self.use_word_segment:
            return word_tokenize(text, format="text").split()
        return text.split()

    def filter_stopwords(self, tokens: list[str]) -> list[str]:
        return [t for t in tokens if t not in STOPWORDS and len(t) > 1]

    def process(self, text: str) -> str:
        text = self.normalize(text)
        no_accent = self.strip_accents(text)
        combined = text if text == no_accent else f"{text} {no_accent}"
        tokens = self.tokenize(combined)
        if self.remove_stopwords:
            tokens = self.filter_stopwords(tokens)
        return " ".join(tokens)

    def process_batch(self, texts: list[str]) -> list[str]:
        return [self.process(t) for t in texts]
