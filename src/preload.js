// src/preload.js  – CommonJS, isolated world
const { contextBridge, ipcRenderer } = require('electron');
const ort = require('onnxruntime-node');
const fs = require('fs');
const path = require('path');
const { spawn } = require('child_process');

const cfgPath = path.join(__dirname, '..', 'cfg.json');
let max_text_len = 0,
    max_lab_len = 0,
    session = null;

async function initialiseDefaultModel() {
    if (fs.existsSync(cfgPath)) {
        const cfg = JSON.parse(fs.readFileSync(cfgPath, 'utf8'));
        max_text_len = cfg.max_text_len;
        max_lab_len = cfg.max_lab_len;
        session = await ort.InferenceSession.create(
            path.join(__dirname, '..', 'models', 'model.onnx')
        );
    }
}

/* ----- TRAIN ----------------------------------------------------- */
function train(cfg) {
    console.log('Training with config:', cfg);
    const cfgFile = path.join(__dirname, '..', 'cfg.json');
    fs.writeFileSync(cfgFile, JSON.stringify(cfg, null, 2));
    const py = spawn(
        'python',
        [path.join(__dirname, 'trainer.py'), cfgFile],
        { cwd: path.join(__dirname) }
    );
    py.stdout.on('data', d => ipcRenderer.send('train-log', d.toString()));
    py.stderr.on('data', d => ipcRenderer.send('train-log', d.toString()));
    py.on('close', c => ipcRenderer.send('train-exit', c));
}

async function loadModel(cfgFile, modelPath) {
    const cfg = JSON.parse(fs.readFileSync(cfgFile, 'utf8'));
    max_text_len = cfg.max_text_len;
    max_lab_len = cfg.max_lab_len;
    session = await ort.InferenceSession.create(modelPath);
}

/* ----- helpers --------------------------------------------------- */
const vecText = txt => {
    const out = new Float32Array(max_text_len);
    for (let i = 0; i < Math.min(txt.length, max_text_len); i++)
        out[i] = txt.charCodeAt(i) / 255;
    return out;
};
const clamp = (x, lo, hi) => Math.min(Math.max(x, lo), hi);
function bits(floatArr) {
    return floatArr
        .slice(0, max_lab_len)
        .map(v => clamp(v, -.5, 1).toFixed(4))
        .join(', ');
}

/* ----- PREDICT --------------------------------------------------- */
async function predict(text) {
    if (!session) throw new Error('Model not loaded—train first.');
    const tensor = new ort.Tensor('float32', vecText(text), [1, max_text_len]);
    const { logits } = await session.run({ input: tensor });
    return bits(Array.from(logits.data));
}

/* ----- expose to renderer --------------------------------------- */
contextBridge.exposeInMainWorld('fileAPI', {
    readJSON: targetPath => {
        if (!path.isAbsolute(targetPath))
            targetPath = path.join(__dirname, targetPath);
        return JSON.parse(fs.readFileSync(targetPath, 'utf8'));
    },
    writeDataset: (name, data) => {
        const dest = path.join(__dirname, '..', 'datasets', name);
        fs.writeFileSync(dest, data);
        return dest;
    }
});
contextBridge.exposeInMainWorld('aiBridge', { train, predict, loadModel });
contextBridge.exposeInMainWorld('trainEvents', {
    onLog: fn => ipcRenderer.on('train-log', (_e, m) => fn(m)),
    onExit: fn => ipcRenderer.on('train-exit', (_e, c) => fn(c))
});

initialiseDefaultModel();

