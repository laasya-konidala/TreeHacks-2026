/**
 * Overlay Renderer — handles agent messages, panel display, and user interaction.
 * Vanilla JS, no frameworks.
 */

// ─── DOM References ───
const statusDot = document.getElementById('status-dot');
const statusText = document.getElementById('status-text');
const touchTarget = document.getElementById('touch-target');
const panelContainer = document.getElementById('panel-container');
const panelContent = document.getElementById('panel-content');

// ─── State ───
let currentSession = null;
let dialogueHistory = [];

// ─── Status Updates ───
window.api.onStatusUpdate((data) => {
    updateStatusUI(data.status);
});

function updateStatusUI(status) {
    statusDot.className = `status-dot ${status}`;
    statusText.textContent = status;
}

// ─── Touch Target (Help Button) ───
touchTarget.addEventListener('click', () => {
    window.api.requestHelp('');
    // Visual feedback
    touchTarget.style.transform = 'scale(0.9)';
    setTimeout(() => { touchTarget.style.transform = ''; }, 200);
});

// ─── Agent Message Handling ───
window.api.onAgentMessage((msg) => {
    console.log('[Overlay] Agent message:', msg.agent_type, msg.content_type);

    if (msg.content_type === 'text' && msg.dialogue_state) {
        showDeepDivePanel(msg);
    } else if (msg.content_type === 'challenge') {
        showAssessmentPanel(msg);
    } else if (msg.content_type === 'visualization') {
        showVisualizerPanel(msg);
    } else if (msg.content_type === 'hint') {
        showHintPanel(msg);
    }
});

// ─── Hide Panel ───
window.api.onStatusUpdate && document.addEventListener('hide-panel-event', () => {
    hidePanel();
});

function hidePanel() {
    panelContainer.classList.remove('visible');
    setTimeout(() => {
        panelContent.innerHTML = '';
        currentSession = null;
        dialogueHistory = [];
    }, 400);
    window.api.hidePanel();
}

function showPanel() {
    panelContainer.classList.add('visible');
}

// ─── Deep Dive Panel (Multi-turn Dialogue) ───
function showDeepDivePanel(msg) {
    currentSession = msg.session_id;

    if (msg.turn_number <= 1) {
        // New conversation
        dialogueHistory = [];
        panelContent.innerHTML = buildDeepDiveHTML(msg);
    }

    // Add agent message
    addDialogueMessage('agent', msg.content);

    // Show reply input unless closing
    const replyContainer = document.querySelector('.reply-container');
    if (replyContainer) {
        if (msg.metadata && msg.metadata.should_close) {
            replyContainer.classList.remove('active');
        } else {
            replyContainer.classList.add('active');
        }
    }

    showPanel();
    scrollToBottom();
}

function buildDeepDiveHTML(msg) {
    const concept = (msg.metadata && msg.metadata.concept) || 'this concept';
    return `
        <div class="panel">
            <div class="panel-header">
                <span class="panel-title">💡 Deep Dive</span>
                <button class="panel-close" onclick="hidePanel()">×</button>
            </div>
            <div class="panel-content" id="dialogue-messages"></div>
            <div class="reply-container active">
                <textarea class="reply-input" id="reply-input" 
                    placeholder="Type your response..." rows="2"
                    onkeydown="handleReplyKeydown(event)"></textarea>
                <button class="reply-send" onclick="sendDialogueReply()">
                    <svg viewBox="0 0 24 24"><path d="M2.01 21L23 12 2.01 3 2 10l15 2-15 2z"/></svg>
                </button>
            </div>
        </div>
    `;
}

function addDialogueMessage(role, content) {
    dialogueHistory.push({ role, content });
    const container = document.getElementById('dialogue-messages');
    if (!container) return;

    const div = document.createElement('div');
    div.className = `message ${role}`;
    div.textContent = content;
    container.appendChild(div);
}

function sendDialogueReply() {
    const input = document.getElementById('reply-input');
    if (!input || !input.value.trim()) return;

    const message = input.value.trim();
    input.value = '';

    // Show user message locally
    addDialogueMessage('user', message);
    scrollToBottom();

    // Send to backend
    window.api.sendReply({
        message: message,
        session_id: currentSession,
    });
}

function handleReplyKeydown(event) {
    if (event.key === 'Enter' && !event.shiftKey) {
        event.preventDefault();
        sendDialogueReply();
    }
}

function scrollToBottom() {
    const container = document.getElementById('dialogue-messages');
    if (container) {
        container.scrollTop = container.scrollHeight;
    }
    panelContainer.scrollTop = panelContainer.scrollHeight;
}

// ─── Assessment Panel (Contrastive Challenge) ───
function showAssessmentPanel(msg) {
    const meta = msg.metadata || {};
    panelContent.innerHTML = `
        <div class="panel">
            <div class="panel-header">
                <span class="panel-title">🎯 Challenge</span>
                <button class="panel-close" onclick="hidePanel()">×</button>
            </div>
            <div class="panel-content">
                <div class="challenge-card">
                    <div class="challenge-label">Think about this</div>
                    <p>${escapeHTML(msg.content)}</p>
                </div>
                <button class="challenge-reveal" onclick="revealAnswer()">
                    Reveal Insight →
                </button>
                <div class="challenge-answer" id="challenge-answer">
                    <p><strong>What changes:</strong> ${escapeHTML(meta.what_changes || '')}</p>
                    <p style="margin-top: 8px;"><strong>Key insight:</strong> ${escapeHTML(meta.expected_insight || '')}</p>
                    ${meta.connects_to ? `<p style="margin-top: 8px; color: #9ca3af; font-size: 12px;">Related: ${escapeHTML(meta.connects_to)}</p>` : ''}
                </div>
            </div>
        </div>
    `;
    showPanel();
}

function revealAnswer() {
    const answer = document.getElementById('challenge-answer');
    if (answer) {
        answer.classList.add('revealed');
    }
}

// ─── Visualizer Panel ───
function showVisualizerPanel(msg) {
    const meta = msg.metadata || {};
    const scene = meta.scene || {};
    const params = scene.interactive_params || [];

    let paramsHTML = '';
    if (params.length > 0) {
        paramsHTML = `
            <div class="viz-params">
                ${params.map(p => `
                    <div class="viz-param">
                        <label>${escapeHTML(p.label || p.name)}</label>
                        ${p.type === 'slider' ? 
                            `<input type="range" min="${p.min}" max="${p.max}" value="${p.default}" step="0.01">` :
                            `<span>${p.default || ''}</span>`
                        }
                    </div>
                `).join('')}
            </div>
        `;
    }

    panelContent.innerHTML = `
        <div class="panel">
            <div class="panel-header">
                <span class="panel-title">🎨 Visualization</span>
                <button class="panel-close" onclick="hidePanel()">×</button>
            </div>
            <div class="panel-content">
                <h3 style="font-size: 16px; margin-bottom: 12px; color: #f0f0f0;">
                    ${escapeHTML(scene.title || 'Visualization')}
                </h3>
                <div class="viz-container" id="viz-canvas">
                    <p style="color: #6b7280; text-align: center; font-size: 13px;">
                        ${scene.type === '3d_surface' ? '🧊 3D Surface' : 
                          scene.type === 'interactive_graph' ? '📈 Interactive Graph' :
                          scene.type === 'comparison' ? '⚖️ Comparison View' :
                          '🎬 Animation'}
                        <br><br>
                        <span style="font-size: 11px;">
                            ${(scene.elements || []).length} elements · 
                            ${(scene.animations || []).length} animations
                        </span>
                    </p>
                </div>
                <div class="viz-narration">${escapeHTML(msg.content || scene.narration || '')}</div>
                ${paramsHTML}
            </div>
        </div>
    `;
    showPanel();
}

// ─── Hint Panel (Quick procedural hint) ───
function showHintPanel(msg) {
    panelContent.innerHTML = `
        <div class="panel">
            <div class="panel-header">
                <span class="panel-title">💬 Hint</span>
                <button class="panel-close" onclick="hidePanel()">×</button>
            </div>
            <div class="panel-content">
                <div class="message agent">
                    ${escapeHTML(msg.content)}
                </div>
            </div>
        </div>
    `;
    showPanel();

    // Auto-hide hint after 15 seconds
    setTimeout(() => {
        if (panelContent.querySelector('.panel-title')?.textContent.includes('Hint')) {
            hidePanel();
        }
    }, 15000);
}

// ─── Utilities ───
function escapeHTML(str) {
    if (!str) return '';
    const div = document.createElement('div');
    div.textContent = str;
    return div.innerHTML;
}

// ─── Initialization ───
console.log('[Overlay] Renderer loaded');
updateStatusUI('dormant');
