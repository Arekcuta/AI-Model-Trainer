
/* ------------- TRAIN ------------------------------------------------ */
document.getElementById('trainBtn').addEventListener('click', () => {
    const rows = window.fileAPI.readJSON('datasets/cmu_stress.json');
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
        epochs: 50,
        batch: 512,
        lr: 1e-3,
        data: './datasets/cmu_stress.json'
    };

    window.aiBridge.train(cfg);      // call preload proxy
});

/* ------------- PREDICT --------------------------------------------- */
document.getElementById('predictBtn').addEventListener('click', async () => {
    const txt = document.getElementById('aiPrediction').value;
    const bits = await window.aiBridge.predict(txt);
    console.log(`"${txt}" ➜ ${bits}`);
});
