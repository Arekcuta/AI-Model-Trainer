import * as ort from 'onnxruntime-gpu';
import fs from 'fs';
import path from 'path';

/* ── 1 · load cfg so we know padding lengths ─────────────────────── */
const cfg = JSON.parse(fs.readFileSync('cfg.json', 'utf8'));
const { max_text_len, max_lab_len } = cfg;

/* ── 2 · helper: vectorise text exactly like trainer_v3.py ────────── */
function vecText(str) {
    const arr = new Float32Array(max_text_len);
    for (let i = 0; i < Math.min(str.length, max_text_len); i++) {
        arr[i] = str.charCodeAt(i) / 255;
    }
    return arr;
}

/* ── 3 · post‑process logits → "10101…" string ────────────────────── */
function logitsToBits(floatArr) {
    let out = '';
    for (let i = 0; i < max_lab_len; i++) {
        out += floatArr[i] > 0 ? '1' : '0';        // sigmoid(x)>0.5 ⇔ x>0
    }
    return out;
}

/* ── 4 · create ONNX session (GPU)  once at startup ───────────────── */
const session = await ort.InferenceSession.create(path.resolve('../models/model.onnx'));

/* ── 5 · public predict function ──────────────────────────────────── */
export async function predict(text) {
    const input = vecText(text);
    const tensor = new ort.Tensor('float32', input, [1, max_text_len]);
    const { logits } = await session.run({ input: tensor });
    return logitsToBits(logits.data);
}
