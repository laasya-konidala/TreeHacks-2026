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

  // Session state for character pulse animation
  onSessionState: (callback) => {
    ipcRenderer.on('session-state', (_event, active) => callback(active));
  },
});
