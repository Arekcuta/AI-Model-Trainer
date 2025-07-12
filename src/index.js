import { WebGPUTrainer } from './webgpu-trainer.js';

const clog = (...a) => { console.log(...a); try {
  window.electronAPI.forwardLog(a.join(' '));
} catch {} };

/* ───────── WebGPU setup ───────── */
let device, trainer;
(async () => {
  if (!navigator.gpu) return alert('WebGPU not supported');
  device   = await (await navigator.gpu.requestAdapter()).requestDevice();
  trainer  = new WebGPUTrainer(device);
  clog('WebGPU ready');
})();

/* ───────── helpers ───────── */
const textVec   = t => [...t].map(c => c.charCodeAt(0) / 255);
const padRight  = (arr,len) => arr.length>=len ? arr.slice(0,len)
                                               : arr.concat(Array(len-arr.length).fill(0));
const bitLabel  = (s,K) => s.slice(0,K).split('').map(Number)
                            .concat(Array(Math.max(0, K-s.length)).fill(0));

/* ───────── load JSON training data ───────── */
async function loadDataset(){
  const f   = document.getElementById('jsonFileInput');
  const txt = document.getElementById('manualJsonInput').value.trim();
  const raw = f.files.length ? JSON.parse(await f.files[0].text())
                             : (txt ? JSON.parse(txt) : null);
  if(!raw) throw 'no dataset';

  const K = raw.reduce((m,r)=>Math.max(m,r.label.length),1);
  const H = raw.reduce((m,r)=>Math.max(m,r.text.length),1);
  const samples = raw.map(r=>({
      input : padRight(textVec(r.text), H),
      label : bitLabel(r.label, K)
  }));
  return {samples: samples, K, H};
}

/* ───────── TRAIN ───────── */
document.getElementById('trainBtn').onclick = async () => {
  try{
    const {samples, K, H} = await loadDataset();
    const M = +document.getElementById('hidden').value || 32;
    const E = +document.getElementById('epochs').value || 1000;

    await trainer.init({inputSize:H, nodesPerLayer:M, outputSize:K});
    await trainer.train(samples, E, (ep,total)=>clog(`epoch ${ep}/${total}`));
    clog('training done');
  }catch(e){ alert(e); }
};

/* ───────── SAVE ───────── */
document.getElementById('saveWeightsBtn').onclick = async () => {
  let file = document.getElementById('saveAs').value.trim();
  if (!file) { alert('set filename first'); return; }

  // always save into the models/ folder
  if (!file.endsWith('.json')) file += '.json';
  const path = `models/${file}`;

  const w = await trainer.exportWeights();
  try {
    window.electronAPI.saveWeights(path, JSON.stringify(w));
    clog('weights saved →', path);
  } catch (e) { console.warn('save bridge missing'); }
};

/* ───────── LOAD ───────── */
document.getElementById('loadWeightsBtn').onclick = async () => {
  const wf = document.getElementById('weightsLoadFile');
  if(!wf.files.length) return alert('choose file');
  const obj = JSON.parse(await wf.files[0].text());
  await trainer.loadWeights(obj);
  clog('weights loaded');
};

/* ───────── PREDICT ───────── */
document.getElementById('predictBtn').onclick = async () => {
  const t = document.getElementById('aiPrediction').value.trim();
  if(!t) return alert('enter text');
  const H = trainer.modelSpec.inputSize;
  const vec = padRight(textVec(t), H);
  const out = await trainer.predictCPU(vec);
  clog('pred:', out.map(v=>v.toFixed(3)));
};
