const { app, BrowserWindow } = require('electron');
const fs = require('fs');
const path = require('path');

const boundsFile = path.join(app.getPath('userData'), 'window-bounds.json');
let mainWin;

/* ---- helpers ---------------------------------------------------- */
const loadBounds = () => {
  try { return JSON.parse(fs.readFileSync(boundsFile, 'utf8')); }
  catch { return { width: 1200, height: 800 }; }
};

const saveBounds = () => {
  if (mainWin?.isDestroyed()) return;
  fs.writeFileSync(boundsFile, JSON.stringify(mainWin.getBounds()));
};

/* ---- window ----------------------------------------------------- */
function createWindow() {
  mainWin = new BrowserWindow({
    ...loadBounds(),
    webPreferences: {
      preload: path.join(__dirname, 'src', 'preload.js'),
      contextIsolation: true,
      nodeIntegration: true,
    }
  });

  mainWin.setMenu(null);
  mainWin.loadFile('src/index.html');
  mainWin.webContents.openDevTools();

  // debounce spammy events
  let t;
  const bump = () => { clearTimeout(t); t = setTimeout(saveBounds, 250); };
  mainWin.on('move', bump);
  mainWin.on('resize', bump);
}

app.whenReady().then(() => {
  createWindow();
  app.on('activate', () => BrowserWindow.getAllWindows().length || createWindow());
});

app.on('window-all-closed', () => process.platform !== 'darwin' && app.quit());
