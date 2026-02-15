const path = require('path');
require('dotenv').config({ path: path.join(__dirname, '.env') });
const { app, BrowserWindow, screen, desktopCapturer, ipcMain, systemPreferences } = require('electron');
const WebSocket = require('ws');
const Anthropic = require('@anthropic-ai/sdk');

// ─── Config ────────────────────────────────────────────────────────
const ANTHROPIC_API_KEY = process.env.ANTHROPIC_API_KEY;
const BACKEND_URL = process.env.BACKEND_URL || 'http://localhost:8080';
const CAPTURE_INTERVAL_MS = 8000;   // 8s — balances cost vs responsiveness
const CAPTURE_WIDTH = 1280;
const CAPTURE_HEIGHT = 720;
const VLM_MODEL = 'claude-haiku-4-5';  // cheap + fast for frequent VLM

// ─── State ─────────────────────────────────────────────────────────
let sidebarWindow = null;
let characterWindow = null;
let sidebarVisible = false;
let currentAvatar = 'plato'; // 'plato' | 'einstein'
let captureInterval = null;
let sessionActive = false;
let claude = null;       // Anthropic client
let contextBuffer = []; // rolling buffer of recent observations
const MAX_CONTEXT = 10;
let agentWs = null;     // WebSocket to Python agent backend

// ─── Create floating character window ───────────────────────────────
function createCharacter() {
  const { width: screenWidth, height: screenHeight } = screen.getPrimaryDisplay().bounds;

  characterWindow = new BrowserWindow({
    width: 100,
    height: 100,
    x: screenWidth - 490,
    y: screenHeight - 140,
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

  characterWindow.loadFile(path.join(__dirname, 'character.html'));
  characterWindow.setVisibleOnAllWorkspaces(true, { visibleOnFullScreen: true });
  characterWindow.setIgnoreMouseEvents(false);

  characterWindow.webContents.once('did-finish-load', () => {
    characterWindow.webContents.send('avatar', currentAvatar);
  });
}

// ─── Move both windows together (fixed relative position) ──────────
ipcMain.on('move-character', (_event, dx, dy) => {
  if (characterWindow && !characterWindow.isDestroyed()) {
    const [x, y] = characterWindow.getPosition();
    characterWindow.setPosition(x + dx, y + dy);
  }
  if (sidebarWindow && !sidebarWindow.isDestroyed()) {
    const [x, y] = sidebarWindow.getPosition();
    sidebarWindow.setPosition(x + dx, y + dy);
  }
});

// ─── Toggle sidebar visibility ─────────────────────────────────────
ipcMain.on('toggle-sidebar', () => {
  if (!sidebarWindow || sidebarWindow.isDestroyed()) return;
  sidebarVisible = !sidebarVisible;
  if (sidebarVisible) {
    sidebarWindow.show();
  } else {
    sidebarWindow.hide();
  }
  if (characterWindow && !characterWindow.isDestroyed()) {
    characterWindow.webContents.send('sidebar-visibility', sidebarVisible);
  }
});

// ─── Avatar selection (from sidebar dropdown) ───────────────────────
ipcMain.on('set-avatar', (_event, avatar) => {
  currentAvatar = avatar;
  if (characterWindow && !characterWindow.isDestroyed()) {
    characterWindow.webContents.send('avatar', avatar);
  }
});

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
    sidebarWindow.hide();
    sendStatus('ready', 'Click the toggle to start a session.');
  });
}

// ─── Claude Vision (screen analysis) ───────────────────────────────
const SYSTEM_PROMPT = `You are an intelligent learning assistant observing a student's screen via periodic screenshots.

Analyze each screenshot and output a JSON summary:
{
  "activity": "what the student is doing right now",
  "topic": "subject/topic (e.g. eigenvalues, gradient_descent, photosynthesis)",
  "subtopic": "more specific sub-topic if identifiable",
  "mode": "CONCEPTUAL | APPLIED | CONSOLIDATION",
  "stuck": true/false,
  "work_status": "correct | incorrect | incomplete | unclear",
  "content_type": "code | equation | text | diagram | video | mixed",
  "error_description": null or "specific error if work is incorrect",
  "natural_pause": true/false,
  "screen_details": "VERY SPECIFIC description of what is visible on screen — read out exact text, equations, variable names, code snippets, question text, diagram labels, video titles, slide headings. Be as literal as possible so a tutor who cannot see the screen knows exactly what the student is looking at."
}

Modes (pick one based on what the student is DOING):
- CONCEPTUAL: watching a video, reading notes/textbook, learning new theory
- APPLIED: solving problems, writing code, doing exercises, practicing
- CONSOLIDATION: reviewing notes, summarizing, making flashcards, organizing

Timing cues — set natural_pause to true if:
- A video appears paused
- They just finished writing something and stopped
- They scrolled to a new section/page
- There's a clear transition between activities
- They seem to be idle / not actively typing or scrolling

screen_details MUST include:
- If there's a question/problem visible: quote the EXACT question text
- If there's code: quote key lines, function names, variable names, errors
- If there's an equation: write it out (e.g. "det(A - λI) = 0")
- If there's a video: title, current slide/frame content, speaker's topic
- If there's a textbook/article: heading, key paragraph content, highlighted text
- If they wrote an answer: quote their EXACT answer so a tutor can check it
- If there's an error message: quote it exactly

Also consider:
- Same content for multiple frames → deeply reading or stuck
- Rapid content changes → skimming or switching tasks

Only output the JSON, nothing else.`;

async function analyzeScreen(base64Image, speechTranscript) {
  if (!claude) return;

  try {
    // Build the user message with context
    let promptText = 'Analyze this screenshot of the student\'s screen.';

    if (contextBuffer.length > 0) {
      promptText += '\n\nPrevious observations (most recent last):\n';
      promptText += contextBuffer.slice(-3).map((c, i) => `${i + 1}. ${c}`).join('\n');
    }

    // Speech transcript is NOT sent to VLM — it goes to the agent via the backend
    const response = await claude.messages.create({
      model: VLM_MODEL,
      max_tokens: 400,
      system: SYSTEM_PROMPT,
      messages: [{
        role: 'user',
        content: [
          {
            type: 'image',
            source: {
              type: 'base64',
              media_type: 'image/jpeg',
              data: base64Image,
            },
          },
          {
            type: 'text',
            text: promptText,
          },
        ],
      }],
    });

    const text = response.content[0]?.text || '';
    if (text) {
      console.log('[Claude VLM]', text.substring(0, 200));

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

      // Feed analysis to the agent backend
      forwardToAgentBackend(text, speechTranscript);

      // Notify sidebar of VLM activity
      try {
        const jsonMatch2 = text.match(/\{[\s\S]*\}/);
        let topic = 'screen activity';
        let activity = 'analyzing';
        let mode = '';
        if (jsonMatch2) {
          const parsed = JSON.parse(jsonMatch2[0]);
          topic = parsed.topic || 'screen activity';
          activity = parsed.activity || 'analyzing';
          mode = parsed.mode || '';
        }
        console.log(`[VLM] ${mode} — ${topic}: ${activity}`);
        if (sidebarWindow && !sidebarWindow.isDestroyed()) {
          sidebarWindow.webContents.send('agent-triggered', {
            reason: 'vlm_output',
            topic: topic,
            activity: activity,
            mode: mode,
            timestamp: Date.now(),
          });
        }
      } catch (e) { /* ignore parse errors */ }
    }
  } catch (err) {
    console.error('[Claude VLM] Error:', err.message || err);
    if (err.message && (err.message.includes('API key') || err.message.includes('credit') || err.message.includes('rate'))) {
      sendStatus('error', `Claude: ${err.message}`);
    }
  }
}

/**
 * Forward Claude's structured analysis to the Python agent backend.
 */
function forwardToAgentBackend(vlmText, speechTranscript) {
  try {
    const jsonMatch = vlmText.match(/\{[\s\S]*\}/);
    if (!jsonMatch) return;

    const analysis = JSON.parse(jsonMatch[0]);

    const ctx = {
      screen_content: analysis.activity || '',
      screen_content_type: analysis.content_type || 'text',
      detected_topic: analysis.topic || '',
      detected_subtopic: analysis.subtopic || '',

      gemini_stuck: !!analysis.stuck,
      gemini_work_status: analysis.work_status || 'unclear',
      gemini_error: analysis.error_description || null,
      gemini_mode: analysis.mode || '',
      gemini_screen_details: analysis.screen_details || '',
      gemini_natural_pause: !!analysis.natural_pause,

      audio_transcript: speechTranscript || null,

      user_id: 'default',
      session_id: 'claude_' + Date.now().toString(36),
      timestamp: new Date().toISOString(),
    };

    fetch(`${BACKEND_URL}/context`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(ctx),
    }).catch(() => {});

  } catch (e) {
    // VLM response wasn't valid JSON — skip silently
  }
}

// ─── Agent Backend WebSocket ────────────────────────────────────────
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

        if (sidebarWindow && !sidebarWindow.isDestroyed()) {
          sidebarWindow.webContents.send('agent-response', msg);
        }
      } catch (e) {
        console.warn('[AgentWS] Parse error:', e.message);
      }
    });

    agentWs.on('close', () => { agentWs = null; });
    agentWs.on('error', () => {});
  } catch (e) {}
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
      console.warn('[Capture] No screen sources found');
      sendStatus('error', '⚠️ No screen sources. Grant Screen Recording permission in System Settings → Privacy & Security.');
      return;
    }

    const source = sources[0];
    const thumbnail = source.thumbnail;

    if (!thumbnail || thumbnail.isEmpty()) {
      console.warn('[Capture] Empty thumbnail');
      sendStatus('error', '⚠️ Screen capture returned empty. Grant Screen Recording permission, then restart.');
      return;
    }

    const jpegBuffer = thumbnail.toJPEG(70);
    const base64Image = jpegBuffer.toString('base64');

    console.log('[Capture] Frame grabbed (' + Math.round(jpegBuffer.length / 1024) + 'KB)');

    const transcript = lastSpeechTranscript;
    lastSpeechTranscript = '';
    analyzeScreen(base64Image, transcript);

  } catch (err) {
    console.error('[Capture] Error:', err.message);
    sendStatus('error', `Capture error: ${err.message}`);
  }
}

function startCapture() {
  captureAndAnalyze();
  captureInterval = setInterval(captureAndAnalyze, CAPTURE_INTERVAL_MS);
  console.log(`[Capture] Started — every ${CAPTURE_INTERVAL_MS / 1000}s`);
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
    if (characterWindow && !characterWindow.isDestroyed()) {
      characterWindow.webContents.send('session-state', false);
    }
  } else {
    try {
      // Initialize Claude client
      claude = new Anthropic({ apiKey: ANTHROPIC_API_KEY });

      // Quick test call to validate the API key
      sendStatus('connecting', 'Connecting to Claude...');
      await claude.messages.create({
        model: VLM_MODEL,
        max_tokens: 10,
        messages: [{ role: 'user', content: 'Say OK' }],
      });

      sendStatus('active', 'Session active — watching screen & listening.');
      startCapture();
      connectAgentWebSocket();

      if (sidebarWindow && !sidebarWindow.isDestroyed()) {
        sidebarWindow.webContents.send('start-mic');
      }

      sessionActive = true;
      if (characterWindow && !characterWindow.isDestroyed()) {
        characterWindow.webContents.send('session-state', true);
      }
    } catch (err) {
      console.error('[Session] Failed to start:', err.message || err);
      sendStatus('error', `Failed to start: ${err.message || err}`);
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
  createCharacter();
});

app.on('window-all-closed', () => {
  stopCapture();
  app.quit();
});
