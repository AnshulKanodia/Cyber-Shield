"""
Cyber Shield - Flask REST API
Serves the trained cyberbullying detection model for real-time predictions.
Enhanced with rule-based threat pattern detection for indirect bullying.
"""

import sys
import os
import re

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import joblib
import time
from src.preprocessing import clean_text
from src.feature_engineering import load_vectorizer
from src.models import predict, load_model
from src.utils import MODELS_DIR, get_logger

logger = get_logger("api_server")

# ── App Setup ──────────────────────────────────────────────────────────────────
app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})

# ── Website Directory ──────────────────────────────────────────────────────────
WEBSITE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "website")


# ══════════════════════════════════════════════════════════════════════════════
# THREAT PATTERN DETECTOR
# Catches indirect/disguised bullying that uses simple words.
# The ML model sees word frequencies; this layer sees dangerous COMBINATIONS.
# ══════════════════════════════════════════════════════════════════════════════

# Patterns are (regex_pattern, severity_score, category)
# severity_score: 0.0 to 1.0 — how confident we are this is harmful
THREAT_PATTERNS = [
    # ── Self-harm / suicide encouragement ──────────────────────────────────
    (r"\b(jump|go)\b.{0,20}\b(cliff|bridge|building|roof|window|ledge|train|truck)\b", 0.95, "bullying"),
    (r"\b(kill|end)\b.{0,15}\b(yourself|urself|ur\s*self|your\s*life)\b", 0.98, "bullying"),
    (r"\bgo\s+(die|away\s+and\s+die|hang|drown)\b", 0.97, "bullying"),
    (r"\b(kys|kms)\b", 0.95, "bullying"),
    (r"\b(drink|eat)\b.{0,15}\b(bleach|poison|acid)\b", 0.96, "bullying"),
    (r"\b(cut|slit)\b.{0,15}\b(wrist|vein|throat)\b", 0.96, "bullying"),
    (r"\bworld.{0,15}(better|without).{0,10}(you|u)\b", 0.92, "bullying"),
    (r"\bnobody.{0,10}(miss|care|notice).{0,10}(you|u|if)\b", 0.90, "bullying"),
    (r"\bshould(n.t|nt)?\s+(have\s+been\s+born|exist|be\s+alive)\b", 0.93, "bullying"),
    (r"\b(don.?t|dont|do\s+not)\b.{0,15}\b(want|need)\b.{0,15}\b(see|talk|hear)\b.{0,10}\b(you|u)\b.{0,10}\b(again|anymore|ever)\b", 0.85, "bullying"),
    (r"\b(disappear|vanish)\b.{0,15}\b(nobody|no\s*one).{0,10}(care|miss|notice)\b", 0.90, "bullying"),
    (r"\bjust\s+(die|disappear|leave|go\s+away)\b", 0.88, "bullying"),
    (r"\bno\s*(one|body)\s*(wants|likes|loves|needs|cares\s+about)\s+(you|u)\b", 0.92, "bullying"),
    (r"\bever(y\s*one|ybody)\s*(hates|despises|is\s+sick\s+of)\s+(you|u)\b", 0.90, "bullying"),

    # ── Direct threats ────────────────────────────────────────────────────
    (r"\bi.{0,5}(will|ll|m\s+gonna|m\s+going\s+to)\b.{0,20}\b(kill|hurt|beat|punch|stab|shoot|find)\b.{0,10}\b(you|u|ur)\b", 0.95, "bullying"),
    (r"\b(watch|wait).{0,15}\b(back|out)\b", 0.65, "bullying"),
    (r"\byou.{0,5}(will|ll|are\s+going\s+to|gonna).{0,10}\b(pay|regret|suffer)\b", 0.85, "bullying"),
    (r"\b(beat|bash|smash).{0,10}\b(face|head|skull|brains)\b", 0.92, "bullying"),

    # ── Exclusion / isolation ─────────────────────────────────────────────
    (r"\bnobody.{0,10}(likes|loves|wants|needs)\b.{0,10}\b(you|u)\b", 0.88, "bullying"),
    (r"\byou.{0,5}(have|got)\s*(no|zero)\s*(friends|life|future|worth|value)\b", 0.87, "bullying"),
    (r"\byou.{0,10}(don.?t|dont)\s*(belong|matter|deserve)\b", 0.85, "bullying"),
    (r"\b(waste|worthless|useless)\s*(of|piece\s+of)?\s*(space|air|oxygen|life|skin)\b", 0.92, "bullying"),
    (r"\b(get|stay)\s*(out|away)\s*(of|from).{0,10}(here|my\s+life|our)\b", 0.70, "bullying"),
    (r"\byou.{0,5}(are|r)\s*(a\s+)?(mistake|accident|burden|embarrassment|disgrace|failure)\b", 0.88, "bullying"),
    (r"\byou.{0,5}(are|r)\s*(not|never)\s*(wanted|welcome|enough|good\s+enough|loved)\b", 0.87, "bullying"),

    # ── Dehumanization / degradation ──────────────────────────────────────
    (r"\byou.{0,5}(look|are|smell)\s*(like)?\s*(a\s+)?(pig|dog|animal|rat|cockroach|insect|trash|garbage|dirt)\b", 0.85, "bullying"),
    (r"\b(ugly|fat|disgusting|pathetic|hideous)\s*(ass|face|looking|creature|thing|person)?\b", 0.75, "toxic"),
    (r"\bno\s*one\s*(will\s+ever|could\s+ever|would\s+ever)\s*(love|date|marry|like|want)\s*(you|u)\b", 0.90, "bullying"),

    # ── Sarcastic / coded bullying ────────────────────────────────────────
    (r"\b(do\s+us|the\s+world)\s*(a\s+)?(favor|favour).{0,15}(disappear|leave|shut|stop|quit)\b", 0.88, "bullying"),
    (r"\btry\s+not\s+to\s+(breathe|exist|show\s+your\s+face)\b", 0.85, "bullying"),
    (r"\b(better\s+off|be\s+better)\s*(dead|gone|without\s+you)\b", 0.93, "bullying"),
    (r"\byou.{0,5}(should|need\s+to)\s*(just\s+)?(quit|give\s+up|stop\s+trying)\b", 0.72, "bullying"),
]

# Additional harmful word combos (checked as pairs)
HARMFUL_COMBOS = [
    ({"jump", "cliff"}, 0.92),
    ({"jump", "bridge"}, 0.92),
    ({"jump", "building"}, 0.88),
    ({"jump", "off"}, 0.60),
    ({"go", "die"}, 0.95),
    ({"go", "kill"}, 0.90),
    ({"kill", "yourself"}, 0.98),
    ({"end", "life"}, 0.85),
    ({"nobody", "want", "you"}, 0.80),
    ({"nobody", "like", "you"}, 0.80),
    ({"nobody", "love", "you"}, 0.82),
    ({"nobody", "care", "you"}, 0.80),
    ({"everyone", "hate"}, 0.85),
    ({"dont", "want", "see", "you"}, 0.75),
    ({"dont", "want", "you", "again"}, 0.70),
    ({"never", "want", "see", "you"}, 0.75),
    ({"better", "off", "dead"}, 0.95),
    ({"waste", "space"}, 0.88),
    ({"waste", "oxygen"}, 0.90),
    ({"no", "friends"}, 0.70),
    ({"no", "one", "like"}, 0.82),
    ({"no", "one", "love"}, 0.82),
    ({"no", "one", "care"}, 0.80),
    ({"world", "better", "without"}, 0.92),
    ({"do", "favor", "disappear"}, 0.88),
    ({"do", "favor", "leave"}, 0.80),
    ({"shut", "up", "forever"}, 0.78),
    ({"ugly", "face"}, 0.75),
    ({"fat", "ugly"}, 0.78),
    ({"stupid", "worthless"}, 0.82),
    ({"loser", "nobody"}, 0.82),
    ({"hate", "you"}, 0.65),
    ({"hate", "your", "face"}, 0.80),
    ({"regret", "born"}, 0.90),
    ({"should", "born"}, 0.85),
    ({"mistake", "born"}, 0.88),
    ({"disappear", "forever"}, 0.85),
]


# ══════════════════════════════════════════════════════════════════════════════
# SAFE PATTERN DETECTOR
# Catches compliments, praise, and positive language the ML model misclassifies.
# Mirror of the threat detector — overrides false positives back to safe.
# ══════════════════════════════════════════════════════════════════════════════

# (regex_pattern, safety_score) — higher = more confident it's safe
SAFE_PATTERNS = [
    # ── Direct compliments ────────────────────────────────────────────────
    (r"\byou.{0,5}(are|r|re)\s*(so\s+)?(funny|amazing|awesome|beautiful|gorgeous|stunning|brilliant|talented|smart|clever|kind|sweet|lovely|wonderful|incredible|fantastic|cool|great|nice|cute|adorable|handsome|pretty|perfect|inspiring|impressive)\b", 0.90),
    (r"\b(love|adore|admire|appreciate)\s+(you|your|how\s+you|the\s+way\s+you)\b", 0.88),
    (r"\b(great|good|nice|amazing|awesome|excellent|wonderful|fantastic|brilliant|outstanding|superb|incredible)\s+(job|work|effort|presentation|performance|idea|point|question)\b", 0.92),
    (r"\b(well\s+done|bravo|kudos|congrats|congratulations|props\s+to)\b", 0.90),
    (r"\b(proud\s+of\s+you|believe\s+in\s+you|rooting\s+for\s+you|happy\s+for\s+you)\b", 0.92),
    (r"\b(keep\s+it\s+up|keep\s+going|you\s+got\s+this|you\s+can\s+do\s+it)\b", 0.88),
    (r"\bhappy\s+(birthday|anniversary|holidays|new\s+year|diwali|christmas|eid)\b", 0.95),

    # ── Positive slang / modern compliments ───────────────────────────────
    (r"\byou\s+slay\b", 0.88),
    (r"\b(slay|slaying|slayed)\s*(it|queen|king|girl|bro|dude|bestie|everything)?\b", 0.85),
    (r"\b(queen|king|legend|icon|boss|goat)\b.*\b(you|your)\b", 0.80),
    (r"\byou.{0,10}\b(queen|king|legend|icon|boss|goat)\b", 0.80),
    (r"\b(fire|lit|bussin|valid|goated|elite|chef.s\s+kiss)\b", 0.70),
    (r"\b(on\s+point|on\s+fire|nailed\s+it|killed\s+it|crushing\s+it|smashed\s+it|aced\s+it)\b", 0.88),
    (r"\b(goals|vibes|energy)\b", 0.60),

    # ── Expressions of laughter / enjoyment ───────────────────────────────
    (r"\bcan.?t\s+stop\s+(laughing|smiling|watching|listening)\b", 0.90),
    (r"\b(hilarious|so\s+funny|too\s+funny|dying\s+of\s+laughter|lmao|lol|rofl|haha)\b", 0.82),
    (r"\b(made\s+my\s+day|brightened\s+my\s+day|you\s+made\s+me\s+(laugh|smile|happy))\b", 0.90),
    (r"\b(love\s+this|love\s+it|love\s+that|this\s+is\s+(great|amazing|awesome|perfect|beautiful))\b", 0.85),

    # ── Friendly / social ─────────────────────────────────────────────────
    (r"\b(miss\s+you|see\s+you\s+soon|can.?t\s+wait\s+to\s+see\s+you)\b", 0.88),
    (r"\b(best\s+friend|bestie|bff|homie|fam|brother|sister|bro|sis)\b", 0.65),
    (r"\b(thank\s+you|thanks|thx|ty|tysm|appreciate\s+it)\b", 0.82),
    (r"\b(good\s+(morning|afternoon|evening|night)|have\s+a\s+(great|good|nice|wonderful)\s+(day|evening|weekend|time))\b", 0.90),
    (r"\b(let.?s\s+(go|hang|meet|chill|catch\s+up|grab\s+(coffee|lunch|dinner|food)))\b", 0.85),
    (r"\b(take\s+care|stay\s+safe|be\s+safe|get\s+well\s+soon)\b", 0.90),

    # ── Positive activities / neutral conversation ────────────────────────
    (r"\b(weather\s+is\s+(nice|beautiful|great|lovely|perfect))\b", 0.92),
    (r"\b(homework|assignment|project|exam|class|lecture|study)\b", 0.60),
    (r"\b(go\s+for\s+a\s+(walk|run|swim|ride|hike|drive))\b", 0.88),
    (r"\b(played\s+(really\s+)?well|great\s+(game|match|play))\b", 0.90),
    (r"\b(dinner|lunch|breakfast|food|pasta|pizza|movie|song|book)\b", 0.55),
]

# Safe word combos — when these sets of words ALL appear, text is likely safe
SAFE_COMBOS = [
    ({"love", "you"}, 0.75),
    ({"great", "job"}, 0.90),
    ({"good", "job"}, 0.90),
    ({"well", "done"}, 0.90),
    ({"proud", "you"}, 0.88),
    ({"happy", "birthday"}, 0.95),
    ({"thank", "you"}, 0.82),
    ({"miss", "you"}, 0.80),
    ({"funny", "laughing"}, 0.90),
    ({"funny", "joke"}, 0.90),
    ({"funny", "jokes"}, 0.90),
    ({"stop", "laughing"}, 0.85),
    ({"amazing", "job"}, 0.90),
    ({"slay", "queen"}, 0.88),
    ({"slay", "outfit"}, 0.88),
    ({"slay", "every"}, 0.85),
    ({"beautiful", "day"}, 0.90),
    ({"nice", "weather"}, 0.90),
    ({"good", "morning"}, 0.90),
    ({"good", "night"}, 0.88),
    ({"take", "care"}, 0.85),
    ({"see", "soon"}, 0.82),
    ({"best", "friend"}, 0.80),
    ({"keep", "going"}, 0.82),
    ({"impressive", "work"}, 0.90),
]


def detect_safe_patterns(text: str) -> dict:
    """
    Scan text for positive/safe patterns that the ML model might misclassify.
    Mirror of detect_threat_patterns — catches false positives.
    
    Returns:
        dict with 'is_safe', 'max_safety', 'matched_count'
    """
    text_lower = text.lower().strip()
    # Normalize common substitutions (same as threat detector)
    text_normalized = text_lower
    text_normalized = re.sub(r"\bu\b", "you", text_normalized)
    text_normalized = re.sub(r"\bur\b", "your", text_normalized)
    text_normalized = re.sub(r"\br\b", "are", text_normalized)
    text_normalized = re.sub(r"\bdon'?t\b", "dont", text_normalized)
    text_normalized = re.sub(r"\bcan'?t\b", "cant", text_normalized)
    text_normalized = re.sub(r"\bwon'?t\b", "wont", text_normalized)
    text_normalized = re.sub(r"\bi'?m\b", "i am", text_normalized)
    text_normalized = re.sub(r"\bi'?ll\b", "i will", text_normalized)
    text_normalized = re.sub(r"\byou'?re\b", "you are", text_normalized)
    
    matched = []
    max_safety = 0.0
    
    # Check regex patterns
    for pattern, safety in SAFE_PATTERNS:
        if re.search(pattern, text_normalized):
            matched.append((pattern, safety))
            if safety > max_safety:
                max_safety = safety
    
    # Check word combo patterns
    words_in_text = set(re.findall(r"\b\w+\b", text_normalized))
    for combo_words, safety in SAFE_COMBOS:
        if combo_words.issubset(words_in_text):
            matched.append((str(combo_words), safety))
            if safety > max_safety:
                max_safety = safety
    
    # Boost safety score when multiple safe patterns match (strong signal)
    if len(matched) >= 2:
        max_safety = min(0.98, max_safety + 0.05 * (len(matched) - 1))
    
    return {
        "is_safe": max_safety >= 0.70,
        "max_safety": max_safety,
        "matched_count": len(matched),
    }


def detect_threat_patterns(text: str) -> dict:
    """
    Scan text for harmful patterns that the ML model might miss.
    
    Returns:
        dict with 'is_threat', 'max_severity', 'matched_patterns', 'category'
    """
    text_lower = text.lower().strip()
    # Normalize common substitutions
    text_normalized = text_lower
    text_normalized = re.sub(r"\bu\b", "you", text_normalized)
    text_normalized = re.sub(r"\bur\b", "your", text_normalized)
    text_normalized = re.sub(r"\br\b", "are", text_normalized)
    text_normalized = re.sub(r"\bdon'?t\b", "dont", text_normalized)
    text_normalized = re.sub(r"\bcan'?t\b", "cant", text_normalized)
    text_normalized = re.sub(r"\bwon'?t\b", "wont", text_normalized)
    text_normalized = re.sub(r"\bshouldn'?t\b", "shouldnt", text_normalized)
    text_normalized = re.sub(r"\bwouldn'?t\b", "wouldnt", text_normalized)
    text_normalized = re.sub(r"\bi'?m\b", "i am", text_normalized)
    text_normalized = re.sub(r"\bi'?ll\b", "i will", text_normalized)
    text_normalized = re.sub(r"\byou'?re\b", "you are", text_normalized)
    
    matched = []
    max_severity = 0.0
    top_category = "safe"
    
    # Check regex patterns
    for pattern, severity, category in THREAT_PATTERNS:
        if re.search(pattern, text_normalized):
            matched.append((pattern, severity, category))
            if severity > max_severity:
                max_severity = severity
                top_category = category
    
    # Check word combo patterns
    words_in_text = set(re.findall(r"\b\w+\b", text_normalized))
    for combo_words, severity in HARMFUL_COMBOS:
        if combo_words.issubset(words_in_text):
            matched.append((str(combo_words), severity, "bullying"))
            if severity > max_severity:
                max_severity = severity
                top_category = "bullying"
    
    return {
        "is_threat": max_severity >= 0.60,
        "max_severity": max_severity,
        "matched_count": len(matched),
        "category": top_category,
    }


def enhanced_predict(text: str, cleaned: str, features) -> dict:
    """
    Combine ML model prediction with rule-based detection.
    
    Pipeline:
    1. ML model prediction
    2. Safe pattern detection (catches false positives)
    3. Threat pattern detection (catches false negatives)
    
    Priority: Threat patterns > Safe patterns > ML model
    (If text matches BOTH threat and safe, threat wins — safety first)
    """
    # Step 1: ML model prediction
    result = predict(features, model=model)
    
    # Step 2: Detect both safe and threat patterns
    safe_info = detect_safe_patterns(text)
    threat_info = detect_threat_patterns(text)
    
    # Step 3: Apply threat override (ML said safe, but rules say threat)
    if threat_info["is_threat"]:
        threat_severity = threat_info["max_severity"]
        
        if result["prediction"] == 0:
            # ML said "safe" but rules detected a threat — OVERRIDE
            logger.info(f"  ⚠️ Pattern override: ML=Safe → Rules=Threat "
                        f"(severity={threat_severity:.2f}, matches={threat_info['matched_count']})")
            
            # Blend confidence: use threat severity as new confidence
            new_confidence = max(threat_severity, result["confidence"] * 0.3 + threat_severity * 0.7)
            
            result["prediction"] = 1
            result["label"] = "Bullying"
            result["confidence"] = round(new_confidence, 4)
            result["bullying_probability"] = round(new_confidence, 4)
            result["safe_probability"] = round(1 - new_confidence, 4)
            result["pattern_override"] = True
        else:
            # ML already said "bullying" — boost confidence if rules agree
            boosted = min(0.99, result["confidence"] * 0.6 + threat_severity * 0.4)
            result["confidence"] = round(max(result["confidence"], boosted), 4)
            result["bullying_probability"] = result["confidence"]
            result["safe_probability"] = round(1 - result["confidence"], 4)
        
        # Set severity level
        if result["confidence"] >= 0.85:
            result["severity"] = "high"
        elif result["confidence"] >= 0.65:
            result["severity"] = "medium"
        else:
            result["severity"] = "low"
        
        result["threat_patterns_matched"] = threat_info["matched_count"]
    
    # Step 4: Apply safe override (ML said bullying, but text is clearly safe)
    # Only if NO threat patterns were matched (threat always wins over safe)
    elif safe_info["is_safe"] and result["prediction"] == 1 and not threat_info["is_threat"]:
        safe_score = safe_info["max_safety"]
        
        logger.info(f"  ✅ Safe override: ML=Bullying → Rules=Safe "
                    f"(safety={safe_score:.2f}, matches={safe_info['matched_count']})")
        
        # Override to safe
        new_safe_confidence = max(safe_score, result["confidence"] * 0.2 + safe_score * 0.8)
        
        result["prediction"] = 0
        result["label"] = "Non-Bullying"
        result["confidence"] = round(new_safe_confidence, 4)
        result["safe_probability"] = round(new_safe_confidence, 4)
        result["bullying_probability"] = round(1 - new_safe_confidence, 4)
        result["severity"] = "none"
        result["safe_override"] = True
        result["safe_patterns_matched"] = safe_info["matched_count"]
    
    return result

# ── Load Model & Vectorizer at Startup ─────────────────────────────────────────
model = None
vectorizer = None
model_name = None


def load_best_model():
    """Load the best performing model and vectorizer."""
    global model, vectorizer, model_name
    
    try:
        # Try loading best model info
        info_path = os.path.join(MODELS_DIR, "best_model_info.joblib")
        if os.path.exists(info_path):
            info = joblib.load(info_path)
            model_name = info["best_model_name"]
        else:
            # Default to XGBoost
            model_name = "XGBoost"
        
        model = load_model(model_name)
        vectorizer = load_vectorizer()
        logger.info(f"✅ Loaded model: {model_name}")
        logger.info(f"✅ Loaded TF-IDF vectorizer")
        return True
    except Exception as e:
        logger.error(f"❌ Failed to load model: {e}")
        return False


# ── API Routes ─────────────────────────────────────────────────────────────────

@app.route("/api/health", methods=["GET"])
def health_check():
    """Health check endpoint."""
    return jsonify({
        "status": "healthy",
        "model_loaded": model is not None,
        "model_name": model_name,
        "version": "1.0.0",
    })


@app.route("/api/predict", methods=["POST"])
def predict_single():
    """
    Predict whether a single text is bullying or not.
    
    Request body: { "text": "..." }
    Response: { "label": "Bullying"/"Non-Bullying", "confidence": 0.87, "severity": "high" }
    """
    if model is None:
        return jsonify({"error": "Model not loaded. Run train_pipeline.py first."}), 503
    
    data = request.get_json()
    if not data or "text" not in data:
        return jsonify({"error": "Missing 'text' field in request body"}), 400
    
    text = data["text"]
    if not text or not text.strip():
        return jsonify({"error": "Text cannot be empty"}), 400
    
    start_time = time.time()
    
    # Preprocess
    cleaned = clean_text(text)
    
    # Transform to features
    features = vectorizer.transform([cleaned])
    
    # Enhanced predict: ML model + threat pattern detection
    result = enhanced_predict(text, cleaned, features)
    result["original_text"] = text
    result["cleaned_text"] = cleaned
    result["processing_time_ms"] = round((time.time() - start_time) * 1000, 2)
    
    return jsonify(result)


@app.route("/api/predict/batch", methods=["POST"])
def predict_batch():
    """
    Predict on multiple texts at once.
    
    Request body: { "texts": ["text1", "text2", ...] }
    Response: { "results": [...], "summary": {...} }
    """
    if model is None:
        return jsonify({"error": "Model not loaded. Run train_pipeline.py first."}), 503
    
    data = request.get_json()
    if not data or "texts" not in data:
        return jsonify({"error": "Missing 'texts' field in request body"}), 400
    
    texts = data["texts"]
    if not isinstance(texts, list) or len(texts) == 0:
        return jsonify({"error": "'texts' must be a non-empty list"}), 400
    
    if len(texts) > 100:
        return jsonify({"error": "Maximum 100 texts per batch request"}), 400
    
    start_time = time.time()
    
    results = []
    bullying_count = 0
    
    for text in texts:
        if not text or not str(text).strip():
            results.append({"error": "Empty text", "label": "Unknown"})
            continue
        
        cleaned = clean_text(str(text))
        features = vectorizer.transform([cleaned])
        result = enhanced_predict(str(text), cleaned, features)
        result["original_text"] = text
        results.append(result)
        
        if result["prediction"] == 1:
            bullying_count += 1
    
    total_time = round((time.time() - start_time) * 1000, 2)
    
    return jsonify({
        "results": results,
        "summary": {
            "total": len(texts),
            "bullying": bullying_count,
            "non_bullying": len(texts) - bullying_count,
            "bullying_percentage": round(bullying_count / len(texts) * 100, 1),
            "processing_time_ms": total_time,
        },
    })


@app.route("/api/model/info", methods=["GET"])
def model_info():
    """Get information about the currently loaded model."""
    info_path = os.path.join(MODELS_DIR, "best_model_info.joblib")
    
    response = {
        "model_name": model_name,
        "model_loaded": model is not None,
        "vectorizer_loaded": vectorizer is not None,
    }
    
    if os.path.exists(info_path):
        info = joblib.load(info_path)
        response["metrics"] = info.get("metrics", {})
    
    return jsonify(response)


# ── Website Routes ─────────────────────────────────────────────────────────────

@app.route("/")
def serve_website():
    """Serve the standalone Cyber Shield website."""
    return send_from_directory(WEBSITE_DIR, "index.html")


@app.route("/<path:path>")
def serve_static(path):
    """Serve static assets (CSS, JS, images) for the website."""
    return send_from_directory(WEBSITE_DIR, path)


# ── Main ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    logger.info("🚀 Starting Cyber Shield API Server...")
    
    if load_best_model():
        logger.info("🌐 Server ready at http://localhost:5000")
        app.run(
            host="0.0.0.0",
            port=5000,
            debug=False,
        )
    else:
        logger.error("❌ Cannot start server without a trained model.")
        logger.error("   Run 'python train_pipeline.py' first.")
        sys.exit(1)
