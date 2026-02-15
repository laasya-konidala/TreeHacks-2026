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

  const PLATO_FRAME_COUNT = 51; // frames 00–50
  const PLATO2_FRAME_START = 28;
  const PLATO2_FRAME_END = 101;
  const PLATO2_FRAME_COUNT = PLATO2_FRAME_END - PLATO2_FRAME_START + 1; // 74 frames
  const PLATO2_BASE = '20260214_1659_01khfcprk3f0r9hkg0knh35eqq_';
  const FRAME_RATE = 100; // ms per frame (~10 fps)

  // Build plato frame URLs (sidebar closed)
  const platoFrameUrls = [];
  for (let i = 0; i < PLATO_FRAME_COUNT; i++) {
    const padded = String(i).padStart(2, '0');
    platoFrameUrls.push(chrome.runtime.getURL(`assets/plato/frame_${padded}.svg`));
  }

  // Build plato2 frame URLs (sidebar open)
  const plato2FrameUrls = [];
  for (let i = PLATO2_FRAME_START; i <= PLATO2_FRAME_END; i++) {
    const padded = String(i).padStart(3, '0');
    plato2FrameUrls.push(chrome.runtime.getURL(`assets/plato2/${PLATO2_BASE}${padded}.svg`));
  }

  const avatarImg = document.createElement('img');
  avatarImg.src = platoFrameUrls[0];
  avatarImg.alt = 'Learning Companion';
  avatar.appendChild(avatarImg);

  document.body.appendChild(avatar);

  // Preload all frames to avoid flicker
  [...platoFrameUrls, ...plato2FrameUrls].forEach((url) => {
    const img = new Image();
    img.src = url;
  });

  // ---------- Toggle sidebar open/close (isOpen needed for animation) ----------
  let isOpen = false;
  let currentAvatar = 'plato'; // 'plato' | 'einstein'

  function getCurrentAvatarSrc() {
    if (currentAvatar === 'einstein') return plato2FrameUrls[currentPlato2Frame];
    return isOpen ? plato2FrameUrls[currentPlato2Frame] : platoFrameUrls[currentPlatoFrame];
  }

  // Animate: plato or einstein (plato2 frames)
  let currentPlatoFrame = 0;
  let currentPlato2Frame = 0;
  setInterval(() => {
    if (currentAvatar === 'einstein') {
      currentPlato2Frame = (currentPlato2Frame + 1) % PLATO2_FRAME_COUNT;
      avatarImg.src = plato2FrameUrls[currentPlato2Frame];
    } else if (isOpen) {
      currentPlato2Frame = (currentPlato2Frame + 1) % PLATO2_FRAME_COUNT;
      avatarImg.src = plato2FrameUrls[currentPlato2Frame];
    } else {
      currentPlatoFrame = (currentPlatoFrame + 1) % PLATO_FRAME_COUNT;
      avatarImg.src = platoFrameUrls[currentPlatoFrame];
    }
  }, FRAME_RATE);

  function toggleSidebar() {
    isOpen = !isOpen;
    sidebar.classList.toggle('lc-open', isOpen);
    overlay.classList.toggle('lc-visible', isOpen);
    avatar.classList.toggle('lc-active', isOpen);
    avatarImg.src = getCurrentAvatarSrc();
  }

  function setAvatar(av) {
    currentAvatar = av;
    avatarImg.src = getCurrentAvatarSrc();
    avatar.classList.toggle('lc-einstein', currentAvatar === 'einstein');
  }

  avatar.addEventListener('click', toggleSidebar);
  overlay.addEventListener('click', toggleSidebar);

  // ---------- Message relay: sidebar iframe → background ----------
  window.addEventListener('message', (event) => {
    const msg = event.data;
    if (!msg || !msg.type) return;

    if (msg.type === 'lc-close-sidebar') {
      if (isOpen) toggleSidebar();
    } else if (msg.type === 'lc-set-avatar') {
      setAvatar(msg.avatar || 'plato');
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
