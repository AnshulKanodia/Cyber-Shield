/**
 * Cyber Shield v2 — Background Service Worker
 * Handles right-click context menu and passes selected text to popup.
 */

const API_BASE = "http://localhost:5000/api";

// ── Create Context Menu on Install ───────────────────────────────────────────
chrome.runtime.onInstalled.addListener(() => {
  chrome.contextMenus.create({
    id: "cyberShieldAnalyze",
    title: "🛡️ Analyze with Cyber Shield",
    contexts: ["selection"],
  });

  console.log("[Cyber Shield] Context menu created");
});

// ── Handle Context Menu Click ────────────────────────────────────────────────
chrome.contextMenus.onClicked.addListener(async (info, tab) => {
  if (info.menuItemId === "cyberShieldAnalyze" && info.selectionText) {
    const selectedText = info.selectionText.trim();

    if (!selectedText) return;

    // Save selected text to storage so popup can pick it up
    await chrome.storage.local.set({
      pendingAnalysis: {
        text: selectedText,
        timestamp: Date.now(),
      },
    });

    // Open the popup (by programmatically triggering the extension action)
    // Note: We can't directly open popup, so we open a small window instead
    // or notify via the action badge

    // Set badge to indicate pending analysis
    chrome.action.setBadgeText({ text: "1", tabId: tab.id });
    chrome.action.setBadgeBackgroundColor({ color: "#6366f1", tabId: tab.id });

    // Try to show a notification-like experience by opening popup
    // The user will click the extension icon to see the result
    // But we can also do an inline notification via content script

    // Send to content script for inline result display
    try {
      await chrome.tabs.sendMessage(tab.id, {
        type: "SHOW_INLINE_RESULT",
        text: selectedText,
      });
    } catch (e) {
      // Content script not loaded on this page, that's fine
      // User will see result when they click the extension icon
    }
  }
});

// ── Listen for messages ──────────────────────────────────────────────────────
chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  if (message.type === "ANALYZE_TEXT") {
    analyzeText(message.text)
      .then((result) => sendResponse(result))
      .catch((err) => sendResponse({ error: err.message }));
    return true;
  }

  if (message.type === "CLEAR_BADGE") {
    if (sender.tab) {
      chrome.action.setBadgeText({ text: "", tabId: sender.tab.id });
    }
  }
});

// ── Analyze text via API ─────────────────────────────────────────────────────
async function analyzeText(text) {
  const response = await fetch(`${API_BASE}/predict`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text }),
  });

  if (!response.ok) throw new Error(`API error: ${response.status}`);
  return await response.json();
}

console.log("[Cyber Shield] Background service worker started");
