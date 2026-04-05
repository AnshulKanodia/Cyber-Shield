/**
 * Cyber Shield v2 — Popup Script
 * Google Translate-style: paste text → analyze → see result
 * Also handles right-click context menu results.
 */

const API_BASE = "http://localhost:5000/api";

// ── DOM Elements ─────────────────────────────────────────────────────────────
const $ = (id) => document.getElementById(id);
const inputText = $("inputText");
const charCount = $("charCount");
const btnAnalyze = $("btnAnalyze");
const btnClear = $("btnClear");
const resultEmpty = $("resultEmpty");
const resultLoading = $("resultLoading");
const resultCard = $("resultCard");
const resultError = $("resultError");
const apiDot = $("apiDot");
const historyList = $("historyList");
const historySection = $("historySection");
const btnClearHistory = $("btnClearHistory");

let history = [];

// ── Init ─────────────────────────────────────────────────────────────────────
document.addEventListener("DOMContentLoaded", async () => {
  checkAPI();
  loadHistory();
  checkForContextMenuText();

  // Event listeners
  inputText.addEventListener("input", onInput);
  btnAnalyze.addEventListener("click", analyze);
  btnClear.addEventListener("click", clearInput);
  btnClearHistory.addEventListener("click", clearHistory);

  // Enter key to analyze (Ctrl+Enter or just Enter)
  inputText.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) {
      e.preventDefault();
      analyze();
    }
  });
});

// ── Check for text passed from context menu ──────────────────────────────────
async function checkForContextMenuText() {
  try {
    const data = await chrome.storage.local.get("pendingAnalysis");
    if (data.pendingAnalysis && data.pendingAnalysis.text) {
      inputText.value = data.pendingAnalysis.text;
      onInput();
      // Clear it
      await chrome.storage.local.remove("pendingAnalysis");
      // Auto-analyze
      analyze();
    }
  } catch (e) {
    // Not in extension context
  }
}

// ── Input handler ────────────────────────────────────────────────────────────
function onInput() {
  const len = inputText.value.length;
  charCount.textContent = `${len} chars`;
  btnAnalyze.disabled = len === 0;
}

// ── Clear input ──────────────────────────────────────────────────────────────
function clearInput() {
  inputText.value = "";
  onInput();
  showState("empty");
}

// ── Analyze text ─────────────────────────────────────────────────────────────
async function analyze() {
  const text = inputText.value.trim();
  if (!text) return;

  showState("loading");
  btnAnalyze.disabled = true;

  try {
    const response = await fetch(`${API_BASE}/predict`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text }),
      signal: AbortSignal.timeout(10000),
    });

    if (!response.ok) throw new Error(`Server error: ${response.status}`);

    const data = await response.json();
    displayResult(data);
    addToHistory(text, data);
  } catch (err) {
    showState("error");
    $("errorMsg").textContent = err.message.includes("timeout")
      ? "Request timed out"
      : err.message.includes("Failed to fetch")
        ? "Cannot connect to API server"
        : err.message;
  } finally {
    btnAnalyze.disabled = false;
  }
}

// ── Display result ───────────────────────────────────────────────────────────
function displayResult(data) {
  showState("result");

  const confidence = data.confidence || 0;
  const confPercent = Math.round(confidence * 100);
  const isBullying = data.prediction === 1;

  // Determine categories with scores
  const categories = classifyText(data);

  // Verdict — always trust the API's prediction, not the derived topCategory
  const verdictConfig = {
    safe:     { icon: "✅", label: "Safe" },
    bullying: { icon: "🚨", label: "Bullying" },
    toxic:    { icon: "☠️", label: "Toxic" },
    spam:     { icon: "📧", label: "Spam" },
    scam:     { icon: "🎣", label: "Scam" },
  };

  // Check if spam/scam keywords dominated (they override the ML verdict)
  const topCategory = categories[0];
  let verdictKey;
  if (topCategory.name === "spam" || topCategory.name === "scam") {
    verdictKey = topCategory.name;
  } else if (isBullying) {
    // Check if toxic scored higher than bullying
    const toxicCat = categories.find((c) => c.name === "toxic");
    verdictKey = (toxicCat && toxicCat.score > 0.3) ? "toxic" : "bullying";
  } else {
    verdictKey = "safe";
  }

  const verdict = verdictConfig[verdictKey] || verdictConfig.safe;
  $("verdictIcon").textContent = verdict.icon;
  $("verdictLabel").textContent = verdict.label;
  $("verdictLabel").className = `verdict-label ${verdictKey}`;

  // Confidence bar
  $("confidenceValue").textContent = `${confPercent}%`;
  const fill = $("confidenceFill");
  fill.className = `confidence-fill ${verdictKey}`;
  fill.style.width = `${confPercent}%`;

  // Category breakdown
  const catNames = ["safe", "bullying", "toxic", "spam", "scam"];
  catNames.forEach((cat) => {
    const catData = categories.find((c) => c.name === cat);
    const score = catData ? catData.score : 0;
    const scoreEl = document.getElementById(`score${cat.charAt(0).toUpperCase() + cat.slice(1)}`);
    scoreEl.textContent = `${Math.round(score * 100)}%`;

    // Highlight active category
    const item = scoreEl.closest(".category-item");
    item.classList.toggle("active", cat === verdictKey);
  });

  // Analyzed text
  $("analyzedText").textContent = data.original_text || inputText.value;
}

// ── Classify text into categories ────────────────────────────────────────────
function classifyText(data) {
  const text = (data.original_text || "").toLowerCase();
  const cleaned = (data.cleaned_text || "").toLowerCase();
  const confidence = data.confidence || 0;
  const isBullying = data.prediction === 1;

  // Spam indicators
  const spamWords = [
    "free", "win", "winner", "click", "subscribe", "buy now", "discount",
    "limited offer", "act now", "deal", "promo", "giveaway", "earn money",
    "make money", "cash", "prize", "congratulations", "claim", "offer",
    "order now", "lowest price", "save big", "100% free",
  ];

  // Scam indicators
  const scamWords = [
    "send money", "wire transfer", "bank account", "password", "ssn",
    "social security", "credit card", "verify your", "urgent", "suspended",
    "locked account", "nigerian prince", "lottery", "inheritance",
    "bitcoin", "crypto invest", "guaranteed return", "double your money",
    "send btc", "whatsapp me", "dm me for", "link in bio",
  ];

  // Toxic indicators (offensive but not direct bullying)
  const toxicWords = [
    "stupid", "idiot", "dumb", "moron", "pathetic", "disgusting",
    "trash", "garbage", "worthless", "useless", "shut up", "hate",
    "worst", "terrible", "horrible", "ugly", "fat", "loser",
  ];

  let scores = {
    safe: 0,
    bullying: 0,
    toxic: 0,
    spam: 0,
    scam: 0,
  };

  // Base scores from ML model
  if (isBullying) {
    scores.bullying = confidence;
    scores.safe = 1 - confidence;
  } else {
    scores.safe = confidence;
    scores.bullying = 1 - confidence;
    // Safeguard: when API says safe, ensure safe always scores higher
    if (scores.bullying >= scores.safe) {
      scores.safe = 0.52;
      scores.bullying = 0.48;
    }
  }

  // Keyword boosting
  const spamScore = calculateKeywordScore(text, spamWords);
  const scamScore = calculateKeywordScore(text, scamWords);
  const toxicScore = calculateKeywordScore(text, toxicWords);

  if (spamScore > 0.3) {
    scores.spam = Math.min(0.95, spamScore + 0.2);
    scores.safe *= 0.3;
  }

  if (scamScore > 0.2) {
    scores.scam = Math.min(0.95, scamScore + 0.3);
    scores.safe *= 0.2;
  }

  if (isBullying && toxicScore > 0.15) {
    // Split between bullying and toxic
    const toxicPortion = Math.min(0.4, toxicScore);
    scores.toxic = scores.bullying * toxicPortion;
    scores.bullying *= (1 - toxicPortion * 0.5);
  } else if (!isBullying && toxicScore > 0.2) {
    scores.toxic = Math.min(0.7, toxicScore + 0.1);
    scores.safe *= 0.5;
  }

  // Normalize scores
  const total = Object.values(scores).reduce((a, b) => a + b, 0);
  if (total > 0) {
    Object.keys(scores).forEach((k) => (scores[k] /= total));
  }

  // Sort by score, return as array
  return Object.entries(scores)
    .map(([name, score]) => ({ name, score }))
    .sort((a, b) => b.score - a.score);
}

function calculateKeywordScore(text, keywords) {
  let matches = 0;
  for (const word of keywords) {
    if (text.includes(word)) matches++;
  }
  return Math.min(1, matches / 3); // Normalize: 3+ matches = max
}

// ── State management ─────────────────────────────────────────────────────────
function showState(state) {
  resultEmpty.classList.toggle("hide", state !== "empty");
  resultLoading.classList.toggle("hide", state !== "loading");
  resultCard.classList.toggle("hide", state !== "result");
  resultError.classList.toggle("hide", state !== "error");
}

// ── History ──────────────────────────────────────────────────────────────────
function addToHistory(text, data) {
  const categories = classifyText(data);
  const topCat = categories[0].name;

  history.unshift({ text: text.substring(0, 100), category: topCat });
  if (history.length > 10) history.pop();

  renderHistory();
  saveHistory();
}

function renderHistory() {
  if (history.length === 0) {
    historySection.classList.add("hide");
    return;
  }
  historySection.classList.remove("hide");

  historyList.innerHTML = history
    .map(
      (item, i) => `
    <div class="history-item" data-index="${i}">
      <span class="history-badge ${item.category}">${item.category}</span>
      <span class="history-text">${escapeHtml(item.text)}</span>
    </div>
  `
    )
    .join("");

  // Click to re-analyze
  historyList.querySelectorAll(".history-item").forEach((el) => {
    el.addEventListener("click", () => {
      const idx = parseInt(el.dataset.index);
      inputText.value = history[idx].text;
      onInput();
      analyze();
    });
  });
}

async function loadHistory() {
  try {
    const data = await chrome.storage.local.get("analysisHistory");
    history = data.analysisHistory || [];
    renderHistory();
  } catch (e) {
    history = [];
  }
}

async function saveHistory() {
  try {
    await chrome.storage.local.set({ analysisHistory: history });
  } catch (e) {}
}

async function clearHistory() {
  history = [];
  renderHistory();
  try {
    await chrome.storage.local.remove("analysisHistory");
  } catch (e) {}
}

// ── API Health Check ─────────────────────────────────────────────────────────
async function checkAPI() {
  try {
    const r = await fetch(`${API_BASE}/health`, { signal: AbortSignal.timeout(3000) });
    const d = await r.json();
    apiDot.className = d.status === "healthy" ? "api-dot online" : "api-dot offline";
  } catch (e) {
    apiDot.className = "api-dot offline";
  }
}

// ── Helpers ──────────────────────────────────────────────────────────────────
function escapeHtml(str) {
  const div = document.createElement("div");
  div.textContent = str;
  return div.innerHTML;
}
