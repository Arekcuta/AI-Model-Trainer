// src/preload.js  – CommonJS, isolated world
(async () => {
    const { contextBridge, ipcRenderer } = require('electron');
    const ort = require('onnxruntime-node');
    const fs = require('fs');
    const path = require('path');
    const { spawn } = require('child_process');

    /* ----- TRAIN ----------------------------------------------------- */
    function train(cfg) {
        console.log('Training with config:', cfg);
        fs.writeFileSync('cfg.json', JSON.stringify(cfg, null, 2));
        const py = spawn('python', ['./trainer.py', './../cfg.json']);
        py.stdout.on('data', d => ipcRenderer.send('train-log', d.toString()));
        py.stderr.on('data', d => ipcRenderer.send('train-log', d.toString()));
        py.on('close', c => ipcRenderer.send('train-exit', c));
    }

    /* ----- LOAD MODEL & CFG ----------------------------------------- */
    const cfgPath = path.join(__dirname, '..', 'cfg.json');
    let max_text_len = 0,
        max_lab_len = 0,
        session = null;

    if (fs.existsSync(cfgPath)) {
        const cfg = JSON.parse(fs.readFileSync(cfgPath, 'utf8'));
        max_text_len = cfg.max_text_len;
        max_lab_len = cfg.max_lab_len;

        session = await ort.InferenceSession.create(
            path.join(__dirname, '..', 'models', 'model.onnx')
        );
    }

    /* ----- helpers --------------------------------------------------- */
    const vecText = txt => {
        const out = new Float32Array(max_text_len);
        for (let i = 0; i < Math.min(txt.length, max_text_len); i++)
            out[i] = txt.charCodeAt(i) / 255;
        return out;
    };
    const bits = f32 => [...f32].map(v => (v > 0 ? '1' : '0')).join('');

    /* ----- PREDICT --------------------------------------------------- */
    async function predict(text) {
        if (!session) throw new Error('Model not loaded—train first.');
        const tensor = new ort.Tensor('float32', vecText(text), [1, max_text_len]);
        const { logits } = await session.run({ input: tensor });
        return bits(logits.data);
    }

    /* ----- expose to renderer --------------------------------------- */
    contextBridge.exposeInMainWorld('fileAPI', {
        readJSON: relPath =>
            JSON.parse(fs.readFileSync(path.join(__dirname, relPath), 'utf8'))
    });
    contextBridge.exposeInMainWorld('aiBridge', { train, predict });
})();
