const path = require('path');
require('dotenv').config({ path: path.join(__dirname, '.env') });
const { app, BrowserWindow, screen, desktopCapturer, ipcMain, systemPreferences } = require('electron');

// ─── Config ────────────────────────────────────────────────────────
const GEMINI_API_KEY = process.env.GEMINI_API_KEY;
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

      // Send to sidebar
      if (sidebarWindow && !sidebarWindow.isDestroyed()) {
        sidebarWindow.webContents.send('gemini-response', {
          text: text,
          timestamp: Date.now(),
        });
      }
    }
  } catch (err) {
    console.error('[Gemini] Error:', err.message);
    // Don't spam error status for transient failures
    if (err.message.includes('API key') || err.message.includes('quota')) {
      sendStatus('error', `Gemini: ${err.message}`);
    }
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
      console.warn('[Capture] No screen sources found');
      return;
    }

    const source = sources[0];
    const thumbnail = source.thumbnail;
    const jpegBuffer = thumbnail.toJPEG(70);
    const base64Image = jpegBuffer.toString('base64');

    console.log('[Capture] Frame grabbed (' + Math.round(jpegBuffer.length / 1024) + 'KB)');

    // Send to Gemini for analysis
    const transcript = lastSpeechTranscript;
    lastSpeechTranscript = ''; // consume it
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

// ─── Speech transcript from renderer ───────────────────────────────
ipcMain.on('speech-transcript', (_event, transcript) => {
  console.log('[Speech]', transcript);
  lastSpeechTranscript = transcript;
});

// ─── Session toggle ────────────────────────────────────────────────
ipcMain.on('toggle-session', async () => {
  if (sessionActive) {
    stopCapture();
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
