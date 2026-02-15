/**
 * Content Script — injected into every page.
 * Monitors user activity (typing, scrolling, content) and sends signals
 * to background.js every 2 seconds.
 */

// ─── Activity Tracking State ───
let keyTimestamps = [];
let deleteCount = 0;
let lastKeystroke = Date.now();
let scrollBackCount = 0;
let lastScrollY = window.scrollY;
let baselineSpeed = null;
let speedSamples = [];

// ─── Keyboard Monitoring ───
document.addEventListener('keydown', (e) => {
    const now = Date.now();
    keyTimestamps.push(now);
    lastKeystroke = now;

    // Track deletions
    if (e.key === 'Backspace' || e.key === 'Delete') {
        deleteCount++;
    }

    // Keep last 60 seconds of keystrokes
    const cutoff = now - 60000;
    keyTimestamps = keyTimestamps.filter(t => t > cutoff);
});

// ─── Scroll Monitoring ───
document.addEventListener('scroll', () => {
    if (window.scrollY < lastScrollY - 50) {
        // Scrolled up significantly — re-reading behavior
        scrollBackCount++;
    }
    lastScrollY = window.scrollY;
});

// ─── Content Extraction ───
function extractContent() {
    // Try to find code blocks, equations, or main content
    const codeBlocks = document.querySelectorAll(
        'pre, code, .CodeMirror, .monaco-editor, .cm-editor, .ace_editor'
    );
    const mathElements = document.querySelectorAll(
        '.MathJax, .katex, math, [data-latex], .mjx-chtml, .mjx-math'
    );

    let contentType = 'text';
    let content = '';

    if (codeBlocks.length > 0) {
        contentType = 'code';
        content = Array.from(codeBlocks)
            .map(el => el.innerText)
            .join('\n')
            .slice(0, 3000);
    } else if (mathElements.length > 0) {
        contentType = 'equation';
        content = Array.from(mathElements)
            .map(el => el.innerText || el.textContent)
            .join('\n')
            .slice(0, 3000);
    } else {
        // Get focused element content or visible page content
        const active = document.activeElement;
        if (active && (active.tagName === 'TEXTAREA' || active.isContentEditable)) {
            content = (active.value || active.innerText || '').slice(0, 3000);
        } else {
            // Get visible content (rough approximation)
            content = document.body.innerText.slice(0, 3000);
        }
    }

    return { content, contentType };
}

// ─── Signal Collection (every 2 seconds) ───
setInterval(() => {
    const now = Date.now();
    const recentKeys = keyTimestamps.filter(t => t > now - 10000); // last 10s
    const currentSpeed = recentKeys.length; // keys per 10 seconds

    // Update baseline (rolling average over ~5 min)
    speedSamples.push(currentSpeed);
    if (speedSamples.length > 150) speedSamples.shift();
    if (speedSamples.length > 10) {
        baselineSpeed = speedSamples.reduce((a, b) => a + b, 0) / speedSamples.length;
    }

    const typingRatio = baselineSpeed ? currentSpeed / Math.max(baselineSpeed, 1) : 1.0;
    const pauseDuration = (now - lastKeystroke) / 1000;
    const { content, contentType } = extractContent();

    // Send to background.js
    try {
        chrome.runtime.sendMessage({
            type: 'activity_signal',
            data: {
                screen_content: content,
                screen_content_type: contentType,
                typing_speed_ratio: Math.round(typingRatio * 100) / 100,
                deletion_rate: deleteCount * (60000 / 10000), // normalize to per minute
                pause_duration: Math.round(pauseDuration * 10) / 10,
                scroll_back_count: scrollBackCount,
                timestamp: new Date().toISOString(),
                url: window.location.href,
            }
        });
    } catch (e) {
        // Extension context invalidated — page outlived the extension
    }

    // Reset per-interval counters
    deleteCount = 0;
    scrollBackCount = 0;
}, 2000);

// ─── Initialization ───
console.log('[Ambient Learning] Content script loaded');
