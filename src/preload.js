const { contextBridge, ipcRenderer } = require('electron');

contextBridge.exposeInMainWorld('electronAPI', {
    sendTraining: (args) => ipcRenderer.send('start-training', args),
    forwardLog: (msg) => ipcRenderer.send('renderer-log', msg),
    saveWeights: (filePath, contents) => ipcRenderer.send('save-weights', { filePath, contents })
});