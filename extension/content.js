/**
 * Cyber Shield v2 — Content Script
 * Shows inline result tooltip when user right-clicks → Analyze with Cyber Shield
 */

const API_BASE = "http://localhost:5000/api";

// ── Category config ──────────────────────────────────────────────────────────
const CATEGORY_CONFIG = {
  safe:     { icon: "✅", label: "Safe",     color: "#10b981", bg: "rgba(16,185,129,0.12)" },
  bullying: { icon: "🚨", label: "Bullying", color: "#ef4444", bg: "rgba(239,68,68,0.12)" },
  toxic:    { icon: "☠️", label: "Toxic",    color: "#f59e0b", bg: "rgba(245,158,11,0.12)" },
  spam:     { icon: "📧", label: "Spam",     color: "#8b5cf6", bg: "rgba(139,92,246,0.12)" },
  scam:     { icon: "🎣", label: "Scam",     color: "#ec4899", bg: "rgba(236,72,153,0.12)" },
};

// ── Listen for messages from background ──────────────────────────────────────
chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  if (message.type === "SHOW_INLINE_RESULT") {
    showInlineResult(message.text);
    sendResponse({ status: "ok" });
  }
  return true;
});

// ── Show inline result near selection ────────────────────────────────────────
async function showInlineResult(text) {
  // Remove any existing tooltip
  removeTooltip();

  // Get selection position
  const selection = window.getSelection();
  if (!selection.rangeCount) return;
  const range = selection.getRangeAt(0);
  const rect = range.getBoundingClientRect();

  // Create loading tooltip
  const tooltip = createTooltip(
    rect,
    `<div style="display:flex;align-items:center;gap:8px;">
      <div class="cs-spinner"></div>
      <span>Analyzing...</span>
    </div>`
  );
  document.body.appendChild(tooltip);

  try {
    // Call API
    const response = await fetch(`${API_BASE}/predict`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text }),
      signal: AbortSignal.timeout(8000),
    });

    if (!response.ok) throw new Error("API error");
    const data = await response.json();

    // Determine category
    const category = getCategory(data, text);
    const config = CATEGORY_CONFIG[category] || CATEGORY_CONFIG.safe;
    const confidence = Math.round((data.confidence || 0) * 100);

    // Update tooltip with result
    tooltip.innerHTML = `
      <div style="display:flex;align-items:center;gap:8px;margin-bottom:6px;">
        <span style="font-size:18px;">${config.icon}</span>
        <span style="font-size:14px;font-weight:700;color:${config.color};">${config.label}</span>
        <span style="font-size:11px;color:rgba(255,255,255,0.5);margin-left:auto;">${confidence}%</span>
      </div>
      <div style="font-size:10px;color:rgba(255,255,255,0.4);overflow:hidden;text-overflow:ellipsis;white-space:nowrap;max-width:250px;">
        "${text.substring(0, 80)}${text.length > 80 ? '...' : ''}"
      </div>
      <div style="margin-top:6px;font-size:9px;color:rgba(255,255,255,0.25);">🛡️ Cyber Shield</div>
    `;
    tooltip.style.borderColor = config.color + "40";

    // Auto-dismiss after 5 seconds
    setTimeout(() => {
      tooltip.style.opacity = "0";
      tooltip.style.transform = "translateY(-4px)";
      setTimeout(() => removeTooltip(), 300);
    }, 5000);

  } catch (err) {
    tooltip.innerHTML = `
      <div style="display:flex;align-items:center;gap:6px;">
        <span>⚠️</span>
        <span style="font-size:11px;color:#ef4444;">Cannot connect to API</span>
      </div>
      <div style="font-size:9px;color:rgba(255,255,255,0.3);margin-top:4px;">
        Run: python api/server.py
      </div>
    `;
    setTimeout(() => removeTooltip(), 4000);
  }

  // Clear the badge
  try {
    chrome.runtime.sendMessage({ type: "CLEAR_BADGE" });
  } catch (e) {}
}

// ── Get category from API response ───────────────────────────────────────────
function getCategory(data, text) {
  const isBullying = data.prediction === 1;
  const lower = text.toLowerCase();

  // Spam check
  const spamWords = ["free", "win", "click", "subscribe", "buy now", "discount", "giveaway", "earn money", "promo"];
  const spamHits = spamWords.filter(w => lower.includes(w)).length;
  if (spamHits >= 2) return "spam";

  // Scam check  
  const scamWords = ["send money", "password", "verify your", "bank account", "urgent", "lottery", "inheritance", "crypto invest"];
  const scamHits = scamWords.filter(w => lower.includes(w)).length;
  if (scamHits >= 1) return "scam";

  if (isBullying) {
    // Check if more "toxic" than "bullying"
    const toxicWords = ["stupid", "idiot", "dumb", "moron", "pathetic", "trash", "garbage", "hate"];
    const bullyWords = ["loser", "kill", "die", "ugly", "fat", "nobody likes", "go away", "kys"];
    const toxicHits = toxicWords.filter(w => lower.includes(w)).length;
    const bullyHits = bullyWords.filter(w => lower.includes(w)).length;
    
    if (toxicHits > bullyHits && data.confidence < 0.8) return "toxic";
    return "bullying";
  }

  return "safe";
}

// ── Create tooltip element ───────────────────────────────────────────────────
function createTooltip(rect, content) {
  const tooltip = document.createElement("div");
  tooltip.id = "cyber-shield-tooltip";
  tooltip.innerHTML = content;

  Object.assign(tooltip.style, {
    position: "fixed",
    zIndex: "2147483647",
    top: `${rect.bottom + 8}px`,
    left: `${Math.max(10, rect.left)}px`,
    maxWidth: "320px",
    padding: "12px 16px",
    background: "rgba(12, 12, 24, 0.95)",
    backdropFilter: "blur(12px)",
    border: "1px solid rgba(99, 102, 241, 0.25)",
    borderRadius: "10px",
    boxShadow: "0 8px 32px rgba(0, 0, 0, 0.5), 0 0 0 1px rgba(255,255,255,0.05)",
    fontFamily: "'Inter', -apple-system, sans-serif",
    fontSize: "12px",
    color: "rgba(255, 255, 255, 0.85)",
    lineHeight: "1.4",
    transition: "opacity 0.3s ease, transform 0.3s ease",
    animation: "csTooltipIn 0.25s ease",
  });

  // Inject keyframes + spinner CSS
  if (!document.getElementById("cyber-shield-styles")) {
    const style = document.createElement("style");
    style.id = "cyber-shield-styles";
    style.textContent = `
      @keyframes csTooltipIn {
        from { opacity: 0; transform: translateY(4px); }
        to { opacity: 1; transform: translateY(0); }
      }
      .cs-spinner {
        width: 14px; height: 14px;
        border: 2px solid rgba(255,255,255,0.1);
        border-top-color: #6366f1;
        border-radius: 50%;
        animation: csSpin 0.7s linear infinite;
      }
      @keyframes csSpin { to { transform: rotate(360deg); } }
    `;
    document.head.appendChild(style);
  }

  // Click outside to dismiss
  setTimeout(() => {
    document.addEventListener("click", dismissOnClick, { once: true });
  }, 100);

  return tooltip;
}

function dismissOnClick(e) {
  const tooltip = document.getElementById("cyber-shield-tooltip");
  if (tooltip && !tooltip.contains(e.target)) {
    removeTooltip();
  }
}

function removeTooltip() {
  const existing = document.getElementById("cyber-shield-tooltip");
  if (existing) existing.remove();
}
