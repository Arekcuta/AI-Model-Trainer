
/* ------------- TRAIN ------------------------------------------------ */
document.getElementById('trainBtn').addEventListener('click', () => {
    console.log('Training started...');
    const epochs = parseInt(document.getElementById('epochs').value, 10);

    const dataFileInput = document.getElementById('jsonFileInput');
    const filePath = dataFileInput.files[0]?.path;          // real path in Electron
    const dataFile = filePath || './../datasets/cmu_stress.json';

    const rows = window.fileAPI.readJSON(dataFile);
    let maxTextLen = 0, maxLabelLen = 0;
    for (const { text = '', label = '' } of rows) {
        if (text.length > maxTextLen) maxTextLen = text.length;
        if (label.length > maxLabelLen) maxLabelLen = label.length;
    }

    const cfg = {
        max_text_len: maxTextLen,
        max_lab_len: maxLabelLen,
        layers: [maxTextLen, 256, 128, maxLabelLen],
        activ: 'relu',
        epochs: epochs,
        batch: 512,
        lr: 1e-3,
        data: dataFile
    };
    console.log('Training config:', cfg);
    window.aiBridge.train(cfg);      // call preload proxy
});

/* ------------- PREDICT --------------------------------------------- */
document.getElementById('predictBtn').addEventListener('click', async () => {
    const txt = document.getElementById('aiPrediction').value;
    const bits = await window.aiBridge.predict(txt);
    console.log(`"${txt}" ➜ ${bits}`);
});
