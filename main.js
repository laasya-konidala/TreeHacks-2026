const { app, BrowserWindow, screen } = require('electron');
const path = require('path');

let overlayWindow = null;

function createOverlay() {
  const { width: screenWidth, height: screenHeight } = screen.getPrimaryDisplay().bounds;
  const panelWidth = 340;

  overlayWindow = new BrowserWindow({
    width: panelWidth,
    height: screenHeight,
    x: screenWidth - panelWidth,
    y: 0,
    frame: false,
    transparent: true,
    alwaysOnTop: true,
    resizable: false,
    backgroundColor: '#00000000',
    webPreferences: {
      nodeIntegration: false,
      contextIsolation: true,
    },
  });

  overlayWindow.loadFile(path.join(__dirname, 'overlay.html'));
  overlayWindow.once('ready-to-show', () => overlayWindow.show());
}

app.whenReady().then(() => {
  createOverlay();
});
