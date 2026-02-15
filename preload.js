const { contextBridge, ipcRenderer } = require('electron');

contextBridge.exposeInMainWorld('api', {
  // Receive Gemini text responses in the sidebar
  onGeminiResponse: (callback) => {
    ipcRenderer.on('gemini-response', (_event, data) => callback(data));
  },

  // Receive status updates
  onStatusUpdate: (callback) => {
    ipcRenderer.on('status-update', (_event, data) => callback(data));
  },

  // Send speech transcript to main process
  sendSpeechTranscript: (transcript) => {
    ipcRenderer.send('speech-transcript', transcript);
  },

  // Tell main process to start/stop the session
  toggleSession: () => {
    ipcRenderer.send('toggle-session');
  },

  // Toggle sidebar visibility (from character window)
  toggleSidebar: () => {
    ipcRenderer.send('toggle-sidebar');
  },

  // Move the character window by a delta (for manual drag)
  moveWindow: (dx, dy) => {
    ipcRenderer.send('move-character', dx, dy);
  },

  // Mic start/stop signals from main
  onStartMic: (callback) => {
    ipcRenderer.on('start-mic', (_event) => callback());
  },
  onStopMic: (callback) => {
    ipcRenderer.on('stop-mic', (_event) => callback());
  },

  // Receive agent responses forwarded from the Python backend via WebSocket
  onAgentResponse: (callback) => {
    ipcRenderer.on('agent-response', (_event, data) => callback(data));
  },

  // Simple agent trigger (Gemini detected user is stuck/confused)
  onAgentTriggered: (callback) => {
    ipcRenderer.on('agent-triggered', (_event, data) => callback(data));
  },

  // Session state for character pulse animation
  onSessionState: (callback) => {
    ipcRenderer.on('session-state', (_event, active) => callback(active));
  },

  // Sidebar open/closed so character can switch plato vs plato2 animation
  onSidebarVisibility: (callback) => {
    ipcRenderer.on('sidebar-visibility', (_event, open) => callback(open));
  },

  // Avatar choice for character window (plato | socrates)
  onAvatar: (callback) => {
    ipcRenderer.on('avatar', (_event, avatar) => callback(avatar));
  },

  // Sidebar: send avatar selection to main
  setAvatar: (avatar) => {
    ipcRenderer.send('set-avatar', avatar);
  },

  // Zoom meeting
  openExternal: (url) => {
    ipcRenderer.send('open-external', url);
  },
  launchZoomMeeting: (url) => {
    ipcRenderer.send('launch-zoom-meeting', url);
  },
  closeZoomMeeting: () => {
    ipcRenderer.send('close-zoom-meeting');
  },
  onZoomMeetingClosed: (callback) => {
    ipcRenderer.on('zoom-meeting-closed', () => callback());
  },
});
