// ===========================================
// Learning Companion - Background Service Worker
// Handles Gemini VLM integration and screen capture.
// Ported from partner's Electron main.js.
// ===========================================

// ─── Config ────────────────────────────────────────────────────────
const GEMINI_API_KEY = 'YOUR_GEMINI_API_KEY_HERE'; // Replace with your key
const GEMINI_API_URL = 'https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent';
const CAPTURE_INTERVAL_MS = 3000;

// ─── State ─────────────────────────────────────────────────────────
let captureInterval = null;
let sessionActive = false;
let contextBuffer = []; // rolling buffer of recent observations
const MAX_CONTEXT = 10;
let lastSpeechTranscript = '';
let activeTabId = null; // the tab that started the session

// ─── System prompt (from partner's main.js) ────────────────────────
const SYSTEM_PROMPT = `You are an intelligent learning assistant observing a user's screen via periodic screenshots.

Your job is to analyze each screenshot and output a concise JSON summary:
{
  "activity": "what the user is doing",
  "topic": "subject/topic they are working on",
  "mode": "CONCEPTUAL | APPLIED | CONSOLIDATION",
  "stuck": true/false,
  "notes": "any transitions, observations, or notable details"
}

Modes:
- CONCEPTUAL: reading, watching lectures, understanding theory
- APPLIED: solving problems, coding, practicing, doing exercises
- CONSOLIDATION: reviewing notes, summarizing, making flashcards, organizing

Also consider:
- If you see the same content for multiple frames, they might be stuck or deeply reading
- If content changes rapidly, they might be skimming or switching tasks
- If there's a transcript of what they said, incorporate it into your analysis

Be concise. Only output the JSON, nothing else.`;

// ─── Gemini API call via fetch ──────────────────────────────────────
async function analyzeScreen(base64Image, speechTranscript) {
  try {
    let promptText = 'Analyze this screenshot of the user\'s screen.';

    if (contextBuffer.length > 0) {
      promptText += '\n\nPrevious observations (most recent last):\n';
      promptText += contextBuffer.slice(-3).map((c, i) => `${i + 1}. ${c}`).join('\n');
    }

    if (speechTranscript) {
      promptText += `\n\nThe user just said: "${speechTranscript}"`;
    }

    const response = await fetch(`${GEMINI_API_URL}?key=${GEMINI_API_KEY}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        contents: [{
          role: 'user',
          parts: [
            { text: SYSTEM_PROMPT + '\n\n' + promptText },
            {
              inlineData: {
                mimeType: 'image/jpeg',
                data: base64Image,
              }
            }
          ]
        }]
      })
    });

    if (!response.ok) {
      const errBody = await response.text();
      throw new Error(`Gemini API error ${response.status}: ${errBody}`);
    }

    const result = await response.json();
    const text = result?.candidates?.[0]?.content?.parts?.[0]?.text || '';

    if (text) {
      console.log('[Gemini]', text.substring(0, 200));

      // Add to rolling context buffer
      contextBuffer.push(text);
      if (contextBuffer.length > MAX_CONTEXT) contextBuffer.shift();

      // Send to content script → sidebar iframe
      sendToTab('gemini-response', { text, timestamp: Date.now() });
    }
  } catch (err) {
    console.error('[Gemini] Error:', err.message);
    if (err.message.includes('API key') || err.message.includes('quota') || err.message.includes('403')) {
      sendToTab('status-update', { status: 'error', message: `Gemini: ${err.message}` });
    }
  }
}

// ─── Screen Capture ────────────────────────────────────────────────
async function captureAndAnalyze() {
  try {
    const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
    if (!tab?.id) return;

    const dataUrl = await chrome.tabs.captureVisibleTab(tab.windowId, {
      format: 'jpeg',
      quality: 70,
    });

    // Extract base64 data from data URL
    const base64Image = dataUrl.replace(/^data:image\/jpeg;base64,/, '');
    console.log('[Capture] Frame grabbed (' + Math.round(base64Image.length * 0.75 / 1024) + 'KB)');

    const transcript = lastSpeechTranscript;
    lastSpeechTranscript = '';
    analyzeScreen(base64Image, transcript);

  } catch (err) {
    console.error('[Capture] Error:', err.message);
  }
}

function startCapture() {
  captureAndAnalyze();
  captureInterval = setInterval(captureAndAnalyze, CAPTURE_INTERVAL_MS);
  console.log(`[Capture] Started — every ${CAPTURE_INTERVAL_MS}ms`);
}

function stopCapture() {
  if (captureInterval) {
    clearInterval(captureInterval);
    captureInterval = null;
    console.log('[Capture] Stopped');
  }
}

// ─── Send message to content script in the active tab ──────────────
function sendToTab(type, data) {
  if (activeTabId) {
    chrome.tabs.sendMessage(activeTabId, { type, ...data }).catch(() => {});
  }
}

// ─── Message listener ──────────────────────────────────────────────
chrome.runtime.onMessage.addListener((message, sender) => {
  if (message.type === 'toggle-session') {
    activeTabId = sender.tab?.id;
    handleToggleSession();
  } else if (message.type === 'speech-transcript') {
    console.log('[Speech]', message.transcript);
    lastSpeechTranscript = message.transcript;
  }
});

async function handleToggleSession() {
  if (sessionActive) {
    // Stop session
    stopCapture();
    sendToTab('stop-mic', {});
    sessionActive = false;
    contextBuffer = [];
    sendToTab('status-update', { status: 'ready', message: 'Session stopped. Click to start again.' });
  } else {
    // Start session — validate API key with a quick test call
    try {
      sendToTab('status-update', { status: 'connecting', message: 'Connecting to Gemini...' });

      const testResponse = await fetch(`${GEMINI_API_URL}?key=${GEMINI_API_KEY}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          contents: [{ role: 'user', parts: [{ text: 'Say OK' }] }]
        })
      });

      if (!testResponse.ok) {
        const errBody = await testResponse.text();
        throw new Error(`API error ${testResponse.status}: ${errBody}`);
      }

      sendToTab('status-update', { status: 'active', message: 'Session active — watching screen & listening.' });
      startCapture();
      sendToTab('start-mic', {});
      sessionActive = true;

    } catch (err) {
      console.error('[Session] Failed to start:', err.message);
      sendToTab('status-update', { status: 'error', message: `Failed to start: ${err.message}` });
    }
  }
}
