"""
Cyber Shield - Utility Functions
Shared helpers used across the pipeline.
"""

import os
import logging
import time
from functools import wraps

# ── Project paths ──────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASETS_DIR = os.path.join(PROJECT_ROOT, "datasets")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
REPORTS_DIR = os.path.join(PROJECT_ROOT, "reports")

# Ensure output directories exist
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

# ── Logging setup ──────────────────────────────────────────────────────────────
def get_logger(name: str) -> logging.Logger:
    """Create a configured logger with console + file output."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter(
            "%(asctime)s │ %(name)-20s │ %(levelname)-8s │ %(message)s",
            datefmt="%H:%M:%S",
        )
        # Console handler
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        logger.addHandler(ch)
        # File handler
        log_path = os.path.join(PROJECT_ROOT, "cyber_shield.log")
        fh = logging.FileHandler(log_path, encoding="utf-8")
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    return logger


# ── Timer decorator ────────────────────────────────────────────────────────────
def timer(func):
    """Decorator to time function execution."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        logger = get_logger("timer")
        start = time.time()
        logger.info(f"⏱  Starting: {func.__name__}")
        result = func(*args, **kwargs)
        elapsed = time.time() - start
        logger.info(f"✅ Finished: {func.__name__} ({elapsed:.2f}s)")
        return result
    return wrapper


# ── Slang dictionary ──────────────────────────────────────────────────────────
SLANG_DICT = {
    "u": "you",
    "ur": "your",
    "r": "are",
    "y": "why",
    "n": "and",
    "b4": "before",
    "bc": "because",
    "bf": "boyfriend",
    "gf": "girlfriend",
    "idk": "i do not know",
    "imo": "in my opinion",
    "imho": "in my humble opinion",
    "irl": "in real life",
    "jk": "just kidding",
    "lmao": "laughing my ass off",
    "lol": "laughing out loud",
    "nvm": "never mind",
    "omg": "oh my god",
    "omfg": "oh my freaking god",
    "pls": "please",
    "plz": "please",
    "rn": "right now",
    "smh": "shaking my head",
    "tbh": "to be honest",
    "thx": "thanks",
    "ty": "thank you",
    "tysm": "thank you so much",
    "wtf": "what the freak",
    "wth": "what the hell",
    "wyd": "what are you doing",
    "stfu": "shut the freak up",
    "gtfo": "get the freak out",
    "af": "as freak",
    "brb": "be right back",
    "btw": "by the way",
    "fyi": "for your information",
    "gonna": "going to",
    "gotta": "got to",
    "wanna": "want to",
    "kinda": "kind of",
    "sorta": "sort of",
    "dunno": "do not know",
    "lemme": "let me",
    "gimme": "give me",
    "cuz": "because",
    "coz": "because",
    "dat": "that",
    "dem": "them",
    "dis": "this",
    "dey": "they",
    "doe": "though",
    "fam": "family",
    "lit": "exciting",
    "salty": "upset",
    "shade": "disrespect",
    "sus": "suspicious",
    "toxic": "harmful",
    "troll": "harasser",
    "clout": "influence",
    "simp": "submissive person",
    "noob": "beginner",
    "ngl": "not going to lie",
    "ong": "on god",
    "istg": "i swear to god",
    "abt": "about",
    "w/": "with",
    "w/o": "without",
    "ppl": "people",
    "govt": "government",
    "msg": "message",
    "pic": "picture",
    "pics": "pictures",
    "info": "information",
}

# ── Contraction expansion ─────────────────────────────────────────────────────
CONTRACTIONS = {
    "ain't": "am not",
    "aren't": "are not",
    "can't": "cannot",
    "couldn't": "could not",
    "didn't": "did not",
    "doesn't": "does not",
    "don't": "do not",
    "hadn't": "had not",
    "hasn't": "has not",
    "haven't": "have not",
    "he'd": "he would",
    "he'll": "he will",
    "he's": "he is",
    "i'd": "i would",
    "i'll": "i will",
    "i'm": "i am",
    "i've": "i have",
    "isn't": "is not",
    "it'd": "it would",
    "it'll": "it will",
    "it's": "it is",
    "let's": "let us",
    "might've": "might have",
    "mustn't": "must not",
    "she'd": "she would",
    "she'll": "she will",
    "she's": "she is",
    "shouldn't": "should not",
    "that's": "that is",
    "there's": "there is",
    "they'd": "they would",
    "they'll": "they will",
    "they're": "they are",
    "they've": "they have",
    "wasn't": "was not",
    "we'd": "we would",
    "we'll": "we will",
    "we're": "we are",
    "we've": "we have",
    "weren't": "were not",
    "what'll": "what will",
    "what're": "what are",
    "what's": "what is",
    "what've": "what have",
    "where's": "where is",
    "who'd": "who would",
    "who'll": "who will",
    "who're": "who are",
    "who's": "who is",
    "who've": "who have",
    "won't": "will not",
    "wouldn't": "would not",
    "you'd": "you would",
    "you'll": "you will",
    "you're": "you are",
    "you've": "you have",
}
