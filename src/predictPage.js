/* Load model and run predictions */
document.getElementById('loadBtn').addEventListener('click', async () => {
    const cfg = document.getElementById('cfgFile').files[0];
    const onnx = document.getElementById('onnxFile').files[0];
    if (!cfg || !onnx) return;
    await window.aiBridge.loadModel(cfg.path, onnx.path);
});

document.getElementById('predictBtn').addEventListener('click', async () => {
    const txt = document.getElementById('predInput').value;
    const out = await window.aiBridge.predict(txt);
    document.getElementById('predOut').textContent = out;
});
