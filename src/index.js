
/* ------------- TRAIN ------------------------------------------------ */
document.getElementById('trainBtn').addEventListener('click', () => {
    const text = document.getElementById('cfgInput').value;
    let cfg;
    try { cfg = JSON.parse(text); }
    catch (e) {
        alert('Invalid JSON configuration');
        return;
    }
    console.log('Training config:', cfg);
    window.aiBridge.train(cfg);
});

async function readJSONFile(file) {
    if (file.path)
        return window.fileAPI.readJSON(file.path);
    const text = await file.text();
    return JSON.parse(text);
}

async function ensureDatasetPath(file) {
    if (file.path) return file.path;
    const text = await file.text();
    return window.fileAPI.writeDataset(file.name || 'dataset.json', text);
}


document.getElementById('loadCfgBtn').addEventListener('click', async () => {
    const f = document.getElementById('cfgFile').files[0];
    if (!f) return;
    const cfgObj = await readJSONFile(f);
    document.getElementById('cfgInput').value = JSON.stringify(cfgObj, null, 2);
});

document.getElementById('toJsonBtn').addEventListener('click', async () => {
    const dataFile = document.getElementById('dataFile').files[0];
    if (!dataFile) { alert('Choose data file'); return; }
    const rows = await readJSONFile(dataFile);
    let maxText = 0, maxLab = 0;
    for (const r of rows) {
        const t = r.text || '';
        const l = r.label || '';
        if (t.length > maxText) maxText = t.length;
        if (l.length > maxLab) maxLab = l.length;
    }
    const layers = document.getElementById('layers').value
        .split(',').map(s => parseInt(s.trim(), 10)).filter(n => !isNaN(n));
    const cfg = {
        data: await ensureDatasetPath(dataFile),
        max_text_len: maxText,
        max_lab_len: maxLab,
        model: document.getElementById('modelType').value,
        layers: [maxText, ...layers, maxLab],
        activ: document.getElementById('activ').value,
        epochs: parseInt(document.getElementById('epochs').value, 10),
        batch: parseInt(document.getElementById('batch').value, 10),
        lr: parseFloat(document.getElementById('lr').value)
    };
    document.getElementById('cfgInput').value = JSON.stringify(cfg, null, 2);
});

if (window.trainEvents) {
    window.trainEvents.onLog(msg => {
        const el = document.getElementById('trainLog');
        el.value += msg;
        el.scrollTop = el.scrollHeight;
    });
    window.trainEvents.onExit(code => {
        const el = document.getElementById('trainLog');
        el.value += `\nprocess exited ${code}\n`;
        el.scrollTop = el.scrollHeight;
    });
}
