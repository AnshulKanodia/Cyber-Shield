/**
 * Cyber Shield — Standalone Website Script
 * Text analysis, history, scroll animations, API health check
 * Shares the same API backend as the Chrome extension
 */

const API_BASE = "/api";

// ── DOM Elements ──────────────────────────────────────────────────────────────
const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => document.querySelectorAll(sel);

const inputText = $("#analyzerInput");
const charCount = $("#charCount");
const btnAnalyze = $("#btnAnalyze");
const btnClear = $("#btnClearInput");
const resultEmpty = $("#resultEmpty");
const resultLoading = $("#resultLoading");
const resultCard = $("#resultCard");
const resultError = $("#resultError");
const statusDot = $("#statusDot");
const statusText = $("#statusText");
const historySection = $("#historySection");
const historyList = $("#historyList");
const historyEmpty = $("#historyEmpty");
const btnClearHistory = $("#btnClearHistory");

let history = [];
let analysisCount = 0;

// ── Init ──────────────────────────────────────────────────────────────────────
document.addEventListener("DOMContentLoaded", () => {
  checkAPI();
  loadHistory();
  setupScrollAnimations();
  setupNavbar();

  // Event listeners
  inputText.addEventListener("input", onInput);
  btnAnalyze.addEventListener("click", analyze);
  btnClear.addEventListener("click", clearInput);
  btnClearHistory.addEventListener("click", clearHistory);

  // Ctrl+Enter to analyze
  inputText.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) {
      e.preventDefault();
      analyze();
    }
  });

  // Smooth scroll for nav links
  $$(".nav-links a").forEach((link) => {
    link.addEventListener("click", (e) => {
      const href = link.getAttribute("href");
      if (href.startsWith("#")) {
        e.preventDefault();
        const target = document.querySelector(href);
        if (target) {
          target.scrollIntoView({ behavior: "smooth", block: "start" });
          // Close mobile menu
          $(".nav-links").classList.remove("mobile-open");
        }
      }
    });
  });

  // Hero CTA button
  const heroCta = $("#heroAnalyze");
  if (heroCta) {
    heroCta.addEventListener("click", (e) => {
      e.preventDefault();
      $("#analyzer").scrollIntoView({ behavior: "smooth", block: "start" });
      setTimeout(() => inputText.focus(), 500);
    });
  }

  // Periodic API health check
  setInterval(checkAPI, 30000);
});

// ── Navbar scroll effect ──────────────────────────────────────────────────────
function setupNavbar() {
  const navbar = $(".navbar");
  const hamburger = $(".nav-hamburger");

  window.addEventListener("scroll", () => {
    navbar.classList.toggle("scrolled", window.scrollY > 20);
  });

  if (hamburger) {
    hamburger.addEventListener("click", () => {
      $(".nav-links").classList.toggle("mobile-open");
    });
  }
}

// ── Scroll reveal animations ──────────────────────────────────────────────────
function setupScrollAnimations() {
  const observer = new IntersectionObserver(
    (entries) => {
      entries.forEach((entry) => {
        if (entry.isIntersecting) {
          entry.target.classList.add("visible");
        }
      });
    },
    { threshold: 0.1, rootMargin: "0px 0px -40px 0px" }
  );

  $$(".reveal").forEach((el) => observer.observe(el));
}

// ── Input handler ─────────────────────────────────────────────────────────────
function onInput() {
  const len = inputText.value.length;
  charCount.textContent = `${len} chars`;
  btnAnalyze.disabled = len === 0;
}

// ── Clear input ───────────────────────────────────────────────────────────────
function clearInput() {
  inputText.value = "";
  onInput();
  showState("empty");
}

// ── Analyze text ──────────────────────────────────────────────────────────────
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
      signal: AbortSignal.timeout(15000),
    });

    if (!response.ok) throw new Error(`Server error: ${response.status}`);

    const data = await response.json();
    displayResult(data);
    addToHistory(text, data);
    analysisCount++;
    updateStats();
  } catch (err) {
    showState("error");
    const errTitle = $("#errorTitle");
    if (errTitle) {
      errTitle.textContent = err.message.includes("timeout")
        ? "Request timed out"
        : err.message.includes("Failed to fetch")
          ? "Cannot connect to API server"
          : err.message;
    }
  } finally {
    btnAnalyze.disabled = false;
  }
}

// ── Display result ────────────────────────────────────────────────────────────
function displayResult(data) {
  showState("result");

  const confidence = data.confidence || 0;
  const confPercent = Math.round(confidence * 100);
  const isBullying = data.prediction === 1;

  // Get category scores
  const categories = classifyText(data);

  // Determine verdict
  const verdictConfig = {
    safe:     { icon: "✅", label: "Safe",     sub: "No threats detected" },
    bullying: { icon: "🚨", label: "Bullying", sub: "Harmful content detected" },
    toxic:    { icon: "☠️", label: "Toxic",    sub: "Offensive language detected" },
    spam:     { icon: "📧", label: "Spam",     sub: "Promotional content detected" },
    scam:     { icon: "🎣", label: "Scam",     sub: "Deceptive content detected" },
  };

  const topCategory = categories[0];
  let verdictKey;
  if (topCategory.name === "spam" || topCategory.name === "scam") {
    verdictKey = topCategory.name;
  } else if (isBullying) {
    const toxicCat = categories.find((c) => c.name === "toxic");
    verdictKey = toxicCat && toxicCat.score > 0.3 ? "toxic" : "bullying";
  } else {
    verdictKey = "safe";
  }

  const verdict = verdictConfig[verdictKey];

  // Update verdict
  const verdictEl = $("#verdict");
  verdictEl.className = `verdict ${verdictKey}`;
  $("#verdictIcon").textContent = verdict.icon;
  const verdictLabel = $("#verdictLabel");
  verdictLabel.textContent = verdict.label;
  verdictLabel.className = `verdict-label ${verdictKey}`;
  $("#verdictSub").textContent = verdict.sub;

  // Confidence bar
  $("#confidenceValue").textContent = `${confPercent}%`;
  const fill = $("#confidenceFill");
  fill.className = `confidence-fill ${verdictKey}`;
  setTimeout(() => { fill.style.width = `${confPercent}%`; }, 50);

  // Category breakdown with bars
  const catNames = ["safe", "bullying", "toxic", "spam", "scam"];
  catNames.forEach((cat) => {
    const catData = categories.find((c) => c.name === cat);
    const score = catData ? catData.score : 0;
    const scorePercent = Math.round(score * 100);

    const scoreEl = $(`#score${capitalize(cat)}`);
    scoreEl.textContent = `${scorePercent}%`;

    const barFill = $(`#bar${capitalize(cat)}`);
    if (barFill) {
      setTimeout(() => { barFill.style.width = `${scorePercent}%`; }, 100);
    }

    const item = scoreEl.closest(".category-item");
    item.classList.toggle("active", cat === verdictKey);
  });

  // Analyzed text
  $("#analyzedText").textContent = data.original_text || inputText.value;
}

// ── Classify text (same logic as extension popup) ─────────────────────────────
function classifyText(data) {
  const text = (data.original_text || "").toLowerCase();
  const confidence = data.confidence || 0;
  const isBullying = data.prediction === 1;

  const spamWords = [
    "free", "win", "winner", "click", "subscribe", "buy now", "discount",
    "limited offer", "act now", "deal", "promo", "giveaway", "earn money",
    "make money", "cash", "prize", "congratulations", "claim", "offer",
    "order now", "lowest price", "save big", "100% free",
  ];

  const scamWords = [
    "send money", "wire transfer", "bank account", "password", "ssn",
    "social security", "credit card", "verify your", "urgent", "suspended",
    "locked account", "nigerian prince", "lottery", "inheritance",
    "bitcoin", "crypto invest", "guaranteed return", "double your money",
    "send btc", "whatsapp me", "dm me for", "link in bio",
  ];

  const toxicWords = [
    "stupid", "idiot", "dumb", "moron", "pathetic", "disgusting",
    "trash", "garbage", "worthless", "useless", "shut up", "hate",
    "worst", "terrible", "horrible", "ugly", "fat", "loser",
  ];

  let scores = { safe: 0, bullying: 0, toxic: 0, spam: 0, scam: 0 };

  if (isBullying) {
    scores.bullying = confidence;
    scores.safe = 1 - confidence;
  } else {
    scores.safe = confidence;
    scores.bullying = 1 - confidence;
    if (scores.bullying >= scores.safe) {
      scores.safe = 0.52;
      scores.bullying = 0.48;
    }
  }

  const spamScore = calcKeywordScore(text, spamWords);
  const scamScore = calcKeywordScore(text, scamWords);
  const toxicScore = calcKeywordScore(text, toxicWords);

  if (spamScore > 0.3) {
    scores.spam = Math.min(0.95, spamScore + 0.2);
    scores.safe *= 0.3;
  }

  if (scamScore > 0.2) {
    scores.scam = Math.min(0.95, scamScore + 0.3);
    scores.safe *= 0.2;
  }

  if (isBullying && toxicScore > 0.15) {
    const toxicPortion = Math.min(0.4, toxicScore);
    scores.toxic = scores.bullying * toxicPortion;
    scores.bullying *= 1 - toxicPortion * 0.5;
  } else if (!isBullying && toxicScore > 0.2) {
    scores.toxic = Math.min(0.7, toxicScore + 0.1);
    scores.safe *= 0.5;
  }

  const total = Object.values(scores).reduce((a, b) => a + b, 0);
  if (total > 0) Object.keys(scores).forEach((k) => (scores[k] /= total));

  return Object.entries(scores)
    .map(([name, score]) => ({ name, score }))
    .sort((a, b) => b.score - a.score);
}

function calcKeywordScore(text, keywords) {
  let matches = 0;
  for (const word of keywords) {
    if (text.includes(word)) matches++;
  }
  return Math.min(1, matches / 3);
}

// ── State management ──────────────────────────────────────────────────────────
function showState(state) {
  resultEmpty.classList.toggle("hide", state !== "empty");
  resultLoading.classList.toggle("hide", state !== "loading");
  resultCard.classList.toggle("hide", state !== "result");
  resultError.classList.toggle("hide", state !== "error");
}

// ── History ───────────────────────────────────────────────────────────────────
function addToHistory(text, data) {
  const categories = classifyText(data);
  const topCat = categories[0].name;

  history.unshift({
    text: text.substring(0, 120),
    category: topCat,
    confidence: Math.round((data.confidence || 0) * 100),
    time: new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }),
  });
  if (history.length > 20) history.pop();

  renderHistory();
  saveHistory();
}

function renderHistory() {
  if (history.length === 0) {
    historySection.classList.add("hide");
    historyEmpty.classList.remove("hide");
    return;
  }
  historySection.classList.remove("hide");
  historyEmpty.classList.add("hide");

  historyList.innerHTML = history
    .map(
      (item, i) => `
    <div class="history-item" data-index="${i}">
      <span class="history-badge ${item.category}">${item.category}</span>
      <span class="history-text">${escapeHtml(item.text)}</span>
      <span class="history-time">${item.time}</span>
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
      $("#analyzer").scrollIntoView({ behavior: "smooth", block: "start" });
      setTimeout(() => analyze(), 400);
    });
  });
}

function loadHistory() {
  try {
    const stored = localStorage.getItem("cyberShieldHistory");
    history = stored ? JSON.parse(stored) : [];
    renderHistory();
  } catch (e) {
    history = [];
  }
}

function saveHistory() {
  try {
    localStorage.setItem("cyberShieldHistory", JSON.stringify(history));
  } catch (e) {}
}

function clearHistory() {
  history = [];
  renderHistory();
  localStorage.removeItem("cyberShieldHistory");
}

// ── Stats ─────────────────────────────────────────────────────────────────────
function updateStats() {
  const statEl = $("#statAnalyses");
  if (statEl) {
    const stored = parseInt(localStorage.getItem("cyberShieldAnalyses") || "0");
    const total = stored + 1;
    localStorage.setItem("cyberShieldAnalyses", total);
    statEl.textContent = total.toLocaleString();
  }
}

// ── API Health Check ──────────────────────────────────────────────────────────
async function checkAPI() {
  try {
    const r = await fetch(`${API_BASE}/health`, {
      signal: AbortSignal.timeout(4000),
    });
    const d = await r.json();
    const isOnline = d.status === "healthy";

    statusDot.className = `nav-status-dot ${isOnline ? "online" : "offline"}`;
    statusText.textContent = isOnline ? "API Online" : "API Offline";

    // Update model info
    if (d.model_name) {
      const modelEl = $("#statModel");
      if (modelEl) modelEl.textContent = d.model_name;
    }
  } catch (e) {
    statusDot.className = "nav-status-dot offline";
    statusText.textContent = "API Offline";
  }
}

// ── Helpers ───────────────────────────────────────────────────────────────────
function escapeHtml(str) {
  const div = document.createElement("div");
  div.textContent = str;
  return div.innerHTML;
}

function capitalize(s) {
  return s.charAt(0).toUpperCase() + s.slice(1);
}
