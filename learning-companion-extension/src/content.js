// ===========================================
// Learning Companion - Content Script
// Injects the floating avatar and sidebar iframe into every page.
// Relays messages between sidebar iframe and background service worker.
// ===========================================

(function () {
  // Guard against double-injection
  if (document.getElementById('lc-avatar')) return;

  // ---------- Create overlay (dims page when sidebar opens) ----------
  const overlay = document.createElement('div');
  overlay.id = 'lc-overlay';
  document.body.appendChild(overlay);

  // ---------- Create sidebar container with iframe ----------
  const sidebar = document.createElement('div');
  sidebar.id = 'lc-sidebar';

  const iframe = document.createElement('iframe');
  iframe.src = chrome.runtime.getURL('src/sidebar.html');
  iframe.allow = 'microphone'; // needed for speech recognition
  sidebar.appendChild(iframe);
  document.body.appendChild(sidebar);

  // ---------- Create floating avatar button with animation ----------
  const avatar = document.createElement('button');
  avatar.id = 'lc-avatar';
  avatar.title = 'Learning Companion';

  const FRAME_COUNT = 51; // frames 00–50
  const FRAME_RATE = 100; // ms per frame (~10 fps)

  // Build array of frame URLs and preload them
  const frameUrls = [];
  const preloadedImages = [];
  for (let i = 0; i < FRAME_COUNT; i++) {
    const padded = String(i).padStart(2, '0');
    frameUrls.push(chrome.runtime.getURL(`assets/plato/frame_${padded}.svg`));
  }

  const avatarImg = document.createElement('img');
  avatarImg.src = frameUrls[0];
  avatarImg.alt = 'Learning Companion';
  avatar.appendChild(avatarImg);
  document.body.appendChild(avatar);

  // Preload all frames to avoid flicker
  frameUrls.forEach((url) => {
    const img = new Image();
    img.src = url;
    preloadedImages.push(img);
  });

  // Animate: cycle through frames in a loop
  let currentFrame = 0;
  setInterval(() => {
    currentFrame = (currentFrame + 1) % FRAME_COUNT;
    avatarImg.src = frameUrls[currentFrame];
  }, FRAME_RATE);

  // ---------- Toggle sidebar open/close ----------
  let isOpen = false;

  function toggleSidebar() {
    isOpen = !isOpen;
    sidebar.classList.toggle('lc-open', isOpen);
    overlay.classList.toggle('lc-visible', isOpen);
    avatar.classList.toggle('lc-active', isOpen);
  }

  avatar.addEventListener('click', toggleSidebar);
  overlay.addEventListener('click', toggleSidebar);

  // ---------- Message relay: sidebar iframe → background ----------
  window.addEventListener('message', (event) => {
    const msg = event.data;
    if (!msg || !msg.type) return;

    if (msg.type === 'lc-close-sidebar') {
      if (isOpen) toggleSidebar();
    } else if (msg.type === 'toggle-session' || msg.type === 'speech-transcript') {
      // Forward to background service worker
      chrome.runtime.sendMessage(msg);
    }
  });

  // ---------- Message relay: background → sidebar iframe ----------
  chrome.runtime.onMessage.addListener((message) => {
    if (!message || !message.type) return;

    // Forward these message types from background to the sidebar iframe
    const forwardTypes = ['gemini-response', 'status-update', 'start-mic', 'stop-mic'];
    if (forwardTypes.includes(message.type)) {
      iframe.contentWindow.postMessage(message, '*');
    }
  });
})();
