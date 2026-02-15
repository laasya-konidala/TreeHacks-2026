/**
 * Background Service Worker — receives behavioral signals from content script
 * and POSTs them to the agent backend. Screenshots are now handled by the
 * Electron app via Gemini VLM, so we only send behavioral data here.
 */

const BACKEND_URL = 'http://localhost:3000';
let latestSignals = null;

// ─── Receive signals from content script ───
chrome.runtime.onMessage.addListener((msg, sender, sendResponse) => {
    if (msg.type === 'activity_signal') {
        latestSignals = msg.data;
    }
});

// ─── Handle Ctrl+Shift+H command ───
chrome.commands.onCommand.addListener((command) => {
    if (command === 'trigger-help') {
        if (latestSignals) {
            latestSignals.user_touched_agent = true;
        }
        // Send immediate touch signal
        fetch(`${BACKEND_URL}/touch`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ message: '' })
        }).catch(err => console.warn('Touch signal failed:', err));
    }
});

// ─── Behavioral signals POST (every 3 seconds) ───
// Sends typing speed, deletion rate, pause duration, scroll-back to backend.
// These get merged with Gemini VLM analysis by the FastAPI server.
setInterval(async () => {
    if (!latestSignals) return;

    try {
        await fetch(`${BACKEND_URL}/context`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                // Behavioral signals from content.js
                screen_content: latestSignals.screen_content || '',
                screen_content_type: latestSignals.screen_content_type || 'text',
                typing_speed_ratio: latestSignals.typing_speed_ratio || 1.0,
                deletion_rate: latestSignals.deletion_rate || 0.0,
                pause_duration: latestSignals.pause_duration || 0.0,
                scroll_back_count: latestSignals.scroll_back_count || 0,
                user_touched_agent: latestSignals.user_touched_agent || false,

                // No screenshot — Gemini handles vision via Electron
                screenshot_b64: null,

                // Let the server merge with Gemini data
                detected_topic: '',
                detected_subtopic: '',
                verbal_confusion_cues: [],
                user_message: null,
                user_id: 'default',
                session_id: '',
                timestamp: new Date().toISOString(),

                // Flag this as behavioral-only data
                _source: 'chrome_extension',
            })
        });

        // Reset touch flag after sending
        if (latestSignals.user_touched_agent) {
            latestSignals.user_touched_agent = false;
        }
    } catch (e) {
        // Silent — backend may not be running
    }
}, 3000);

// ─── Initialization ───
console.log('[Ambient Learning] Background service worker started (behavioral signals only — Gemini handles vision)');
