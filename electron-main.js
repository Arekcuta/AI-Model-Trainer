const { app, BrowserWindow, ipcMain } = require('electron');
const path = require('path');
const { spawn } = require('child_process');
const fs = require('fs');

app.commandLine.appendSwitch('enable-features', 'SharedArrayBuffer');

function createWindow() {
  const win = new BrowserWindow({
    width: 800,
    height: 600,
    webPreferences: {
      preload: path.join(__dirname, 'src/preload.js'),
      nodeIntegration: false,
      contextIsolation: true
    }
  });

  win.setMenu(null);
  win.loadFile('src/index.html');
  win.webContents.openDevTools();
}

app.whenReady().then(() => {
  createWindow();

  app.on('activate', function () {
    if (BrowserWindow.getAllWindows().length === 0) createWindow();
  });
});

app.on('window-all-closed', function () {
  if (process.platform !== 'darwin') app.quit();
});

// Example: Listen for IPC call to run trainer
ipcMain.on('start-training', (event, args) => {
  const { dataFile, outputFile, epochs, numLayers, nodesPerLayer, activation } = args;

  const pythonArgs = ['trainer/train.py'];
  if (dataFile) pythonArgs.push('--data', dataFile);
  if (outputFile) pythonArgs.push('--output', outputFile);
  if (epochs) pythonArgs.push('--epochs', epochs.toString());
  if (numLayers) pythonArgs.push('--layers', numLayers.toString());
  if (nodesPerLayer) pythonArgs.push('--nodes', nodesPerLayer.toString());
  if (activation) pythonArgs.push('--activation', activation);

  console.log("Launching Python trainer with args:", pythonArgs);

  const pyProc = spawn(
    'C:\\Users\\tuckj\\AppData\\Local\\Programs\\Python\\Python311\\python.exe',
    pythonArgs,
    { stdio: 'inherit' }
  );

  pyProc.on('close', (code) => {
    console.log(`Python trainer exited with code ${code}`);
  });
});

ipcMain.on('renderer-log', (event, msg) => {
  console.log(`[Renderer] ${msg}`);
});

ipcMain.on('save-weights', (event, { filePath, contents }) => {
  fs.writeFile(filePath, contents, 'utf8', (err) => {
    if (err) {
      console.error('Error saving weights:', err);
    } else {
      console.log('Weights saved to', filePath);
    }
  });
});