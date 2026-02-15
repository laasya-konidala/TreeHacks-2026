const { contextBridge, ipcRenderer } = require('electron');

contextBridge.exposeInMainWorld('api', {
  // Receive Gemini text responses in the sidebar
  onGeminiResponse: (callback) => {
    ipcRenderer.on('gemini-response', (_event, data) => callback(data));
  },

  // Receive status updates (connected, disconnected, capturing, etc.)
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

  // Mic start/stop signals from main
  onStartMic: (callback) => {
    ipcRenderer.on('start-mic', (_event) => callback());
  },
  onStopMic: (callback) => {
    ipcRenderer.on('stop-mic', (_event) => callback());
  },
});
