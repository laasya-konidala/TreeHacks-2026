const path = require('path');
require('dotenv').config({ path: path.join(__dirname, '.env') });
const { app, BrowserWindow, screen, desktopCapturer, ipcMain, systemPreferences } = require('electron');
const WebSocket = require('ws');

// ─── Config ────────────────────────────────────────────────────────
const GEMINI_API_KEY = process.env.GEMINI_API_KEY;
const BACKEND_URL = process.env.BACKEND_URL || 'http://localhost:8080';
const CAPTURE_INTERVAL_MS = 3000;
const CAPTURE_WIDTH = 1280;
const CAPTURE_HEIGHT = 720;

// ─── State ─────────────────────────────────────────────────────────
let sidebarWindow = null;
let captureInterval = null;
let sessionActive = false;
let genai = null;       // GoogleGenAI instance
let contextBuffer = []; // rolling buffer of recent observations
const MAX_CONTEXT = 10; // keep last 10 observations
let agentWs = null;     // WebSocket to Python agent backend

// ─── Create the sidebar overlay window ─────────────────────────────
function createSidebar() {
  const { width: screenWidth, height: screenHeight } = screen.getPrimaryDisplay().bounds;
  const panelWidth = 380;

  sidebarWindow = new BrowserWindow({
    width: panelWidth,
    height: screenHeight,
    x: screenWidth - panelWidth,
    y: 0,
    frame: false,
    transparent: true,
    alwaysOnTop: true,
    resizable: false,
    skipTaskbar: true,
    backgroundColor: '#00000000',
    hasShadow: false,
    webPreferences: {
      preload: path.join(__dirname, 'preload.js'),
      nodeIntegration: false,
      contextIsolation: true,
    },
  });

  sidebarWindow.loadFile(path.join(__dirname, 'overlay.html'));
  sidebarWindow.setVisibleOnAllWorkspaces(true, { visibleOnFullScreen: true });
  sidebarWindow.setIgnoreMouseEvents(false);

  sidebarWindow.once('ready-to-show', () => {
    sidebarWindow.show();
    sendStatus('ready', 'Click the toggle to start a session.');
  });
}

// ─── Gemini generateContent (regular API with vision) ──────────────
const SYSTEM_PROMPT = `You are an intelligent learning assistant observing a user's screen via periodic screenshots.

Your job is to analyze each screenshot and output a concise JSON summary:
{
  "activity": "what the user is doing",
  "topic": "subject/topic they are working on (e.g. gradient_descent, linear_algebra, calculus)",
  "subtopic": "more specific sub-topic if identifiable",
  "mode": "CONCEPTUAL | APPLIED | CONSOLIDATION",
  "stuck": true/false,
  "work_status": "correct | incorrect | incomplete | unclear",
  "content_type": "code | equation | text | diagram | mixed",
  "confused_about": ["concept1", "concept2"],
  "understands": ["concept3"],
  "error_description": null or "specific error if work is incorrect",
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
- Focus on WHAT CONCEPT they're working on and WHETHER THEIR WORK IS CORRECT

Be concise. Only output the JSON, nothing else.`;

async function analyzeScreen(base64Image, speechTranscript) {
  if (!genai) return;

  try {
    const { GoogleGenAI } = await import('@google/genai');
    if (!genai) genai = new GoogleGenAI({ apiKey: GEMINI_API_KEY });

    const contents = [];

    // Build the prompt with context
    let promptText = 'Analyze this screenshot of the user\'s screen.';

    if (contextBuffer.length > 0) {
      promptText += '\n\nPrevious observations (most recent last):\n';
      promptText += contextBuffer.slice(-3).map((c, i) => `${i + 1}. ${c}`).join('\n');
    }

    if (speechTranscript) {
      promptText += `\n\nThe user just said: "${speechTranscript}"`;
    }

    const response = await genai.models.generateContent({
      model: 'gemini-2.5-flash',
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
      }],
    });

    const text = response.text || '';
    if (text) {
      console.log('[Gemini]', text.substring(0, 200));

      // Add to rolling context buffer
      contextBuffer.push(text);
      if (contextBuffer.length > MAX_CONTEXT) contextBuffer.shift();

      // Send to sidebar for display
      if (sidebarWindow && !sidebarWindow.isDestroyed()) {
        sidebarWindow.webContents.send('gemini-response', {
          text: text,
          timestamp: Date.now(),
        });
      }

      // ─── Feed Gemini analysis to the agent backend ───
      forwardToAgentBackend(text, speechTranscript);

      // ─── ALWAYS trigger agent on every Gemini response ───
      try {
        const jsonMatch2 = text.match(/\{[\s\S]*\}/);
        let topic = 'screen activity';
        let activity = 'analyzing';
        let confused = [];
        if (jsonMatch2) {
          const parsed = JSON.parse(jsonMatch2[0]);
          topic = parsed.topic || 'screen activity';
          activity = parsed.activity || 'analyzing';
          confused = parsed.confused_about || [];
        }
        console.log('[Agent] TRIGGERED — VLM produced output');
        if (sidebarWindow && !sidebarWindow.isDestroyed()) {
          sidebarWindow.webContents.send('agent-triggered', {
            reason: 'vlm_output',
            topic: topic,
            activity: activity,
            confused_about: confused,
            timestamp: Date.now(),
          });
        }
      } catch (e) { /* ignore parse errors */ }
    }
  } catch (err) {
    console.error('[Gemini] Error:', err.message);
    // Don't spam error status for transient failures
    if (err.message.includes('API key') || err.message.includes('quota')) {
      sendStatus('error', `Gemini: ${err.message}`);
    }
  }
}

/**
 * Forward Gemini's structured analysis to the Python agent backend.
 * This replaces the Chrome extension's screenshot capture — Gemini is now
 * the VLM that feeds the confusion detector and agent routing.
 */
function forwardToAgentBackend(geminiText, speechTranscript) {
  try {
    // Parse Gemini's JSON response
    const jsonMatch = geminiText.match(/\{[\s\S]*\}/);
    if (!jsonMatch) return;

    const analysis = JSON.parse(jsonMatch[0]);

    // Build a WorkContext for the agent backend
    const ctx = {
      // Gemini vision analysis fields
      screen_content: analysis.activity || '',
      screen_content_type: analysis.content_type || 'text',
      detected_topic: analysis.topic || '',
      detected_subtopic: analysis.subtopic || '',

      // Gemini-specific fields (new)
      gemini_stuck: !!analysis.stuck,
      gemini_work_status: analysis.work_status || 'unclear',
      gemini_confused_about: analysis.confused_about || [],
      gemini_understands: analysis.understands || [],
      gemini_error: analysis.error_description || null,
      gemini_mode: analysis.mode || '',
      gemini_notes: analysis.notes || '',

      // Behavioral signals — not available from Electron screen capture,
      // Chrome extension will merge these in separately
      typing_speed_ratio: 1.0,
      deletion_rate: 0.0,
      pause_duration: 0.0,
      scroll_back_count: 0,

      // Verbal cues from speech transcript
      verbal_confusion_cues: speechTranscript ? [speechTranscript] : [],
      audio_transcript: speechTranscript || null,

      user_touched_agent: false,
      user_message: null,
      user_id: 'default',
      session_id: 'gemini_' + Date.now().toString(36),
      timestamp: new Date().toISOString(),

      // No screenshot needed — Gemini already analyzed it
      screenshot_b64: null,
    };

    fetch(`${BACKEND_URL}/context`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(ctx),
    }).catch(() => {
      // Silent — backend not running is fine for the simple agent demo
    });

  } catch (e) {
    // Gemini response wasn't valid JSON — skip silently
  }
}

// ─── Agent Backend WebSocket (optional — connects if backend is running) ──
function connectAgentWebSocket() {
  if (agentWs) {
    try { agentWs.close(); } catch (e) {}
    agentWs = null;
  }

  try {
    agentWs = new WebSocket(`ws://localhost:8080/ws`);

    agentWs.on('open', () => {
      console.log('[AgentWS] Connected to agent backend');
    });

    agentWs.on('message', (data) => {
      try {
        const msg = JSON.parse(data.toString());
        console.log(`[AgentWS] ${msg.agent_type}: ${(msg.content || '').substring(0, 100)}`);

        // Forward agent response to overlay sidebar
        if (sidebarWindow && !sidebarWindow.isDestroyed()) {
          sidebarWindow.webContents.send('agent-response', msg);
        }
      } catch (e) {
        console.warn('[AgentWS] Parse error:', e.message);
      }
    });

    agentWs.on('close', () => {
      agentWs = null;
      // Don't spam reconnects — backend is optional for the simple demo
    });

    agentWs.on('error', () => {
      // Silent — backend not running is fine for the simple agent trigger demo
    });
  } catch (e) {
    // Silent — backend not running is fine
  }
}

function disconnectAgentWebSocket() {
  if (agentWs) {
    try { agentWs.close(); } catch (e) {}
    agentWs = null;
  }
}

// ─── Screen Capture ────────────────────────────────────────────────
let lastSpeechTranscript = '';

async function captureAndAnalyze() {
  try {
    const sources = await desktopCapturer.getSources({
      types: ['screen'],
      thumbnailSize: { width: CAPTURE_WIDTH, height: CAPTURE_HEIGHT },
    });

    if (sources.length === 0) {
      console.warn('[Capture] No screen sources found — is screen recording permission granted?');
      sendStatus('error', '⚠️ No screen sources. Grant Screen Recording permission in System Settings → Privacy & Security.');
      return;
    }

    const source = sources[0];
    const thumbnail = source.thumbnail;

    if (!thumbnail || thumbnail.isEmpty()) {
      console.warn('[Capture] Empty thumbnail — screen recording permission likely not granted');
      sendStatus('error', '⚠️ Screen capture returned empty. Grant Screen Recording permission in System Settings → Privacy & Security, then restart the app.');
      return;
    }

    const jpegBuffer = thumbnail.toJPEG(70);
    const base64Image = jpegBuffer.toString('base64');

    console.log('[Capture] Frame grabbed (' + Math.round(jpegBuffer.length / 1024) + 'KB)');

    // Send to Gemini for analysis
    const transcript = lastSpeechTranscript;
    lastSpeechTranscript = ''; // consume it
    analyzeScreen(base64Image, transcript);

  } catch (err) {
    console.error('[Capture] Error:', err.message);
    sendStatus('error', `Capture error: ${err.message}`);
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

// ─── Speech transcript from renderer ───────────────────────────────
ipcMain.on('speech-transcript', (_event, transcript) => {
  console.log('[Speech]', transcript);
  lastSpeechTranscript = transcript;
});

// ─── Session toggle ────────────────────────────────────────────────
ipcMain.on('toggle-session', async () => {
  if (sessionActive) {
    stopCapture();
    disconnectAgentWebSocket();
    if (sidebarWindow && !sidebarWindow.isDestroyed()) {
      sidebarWindow.webContents.send('stop-mic');
    }
    sessionActive = false;
    contextBuffer = [];
    sendStatus('ready', 'Session stopped. Click to start again.');
  } else {
    // Initialize genai if needed
    try {
      const { GoogleGenAI } = await import('@google/genai');
      genai = new GoogleGenAI({ apiKey: GEMINI_API_KEY });

      // Quick test call to validate the API key
      sendStatus('connecting', 'Connecting to Gemini...');
      await genai.models.generateContent({
      model: 'gemini-2.5-flash',
      contents: [{ role: 'user', parts: [{ text: 'Say OK' }] }],
      });

      sendStatus('active', 'Session active — watching screen & listening.');
      startCapture();
      connectAgentWebSocket();

      if (sidebarWindow && !sidebarWindow.isDestroyed()) {
        sidebarWindow.webContents.send('start-mic');
      }

      sessionActive = true;
    } catch (err) {
      console.error('[Session] Failed to start:', err.message);
      sendStatus('error', `Failed to start: ${err.message}`);
    }
  }
});

// ─── Helpers ───────────────────────────────────────────────────────
function sendStatus(status, message) {
  if (sidebarWindow && !sidebarWindow.isDestroyed()) {
    sidebarWindow.webContents.send('status-update', { status, message });
  }
}

// ─── App lifecycle ─────────────────────────────────────────────────
app.whenReady().then(() => {
  if (process.platform === 'darwin') {
    const screenAccess = systemPreferences.getMediaAccessStatus('screen');
    console.log('[Permissions] Screen access:', screenAccess);
    if (screenAccess !== 'granted') {
      console.log('[Permissions] Screen recording permission required!');
    }
  }
  createSidebar();
});

app.on('window-all-closed', () => {
  stopCapture();
  app.quit();
});
