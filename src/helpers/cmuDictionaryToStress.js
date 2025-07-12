// cmu‑to‑stress.js
// ------------------------------------------
// 1.  put the CMU dictionary text file in the
//     same folder and name it “cmudict.txt”.
// 2.  run   node cmu-to-stress.js
// 3.  you’ll get “cmu_stress.json” in the same
//     folder, formatted exactly like:
//
//     [
//       { "text": "prediction",  "label": "010" },
//       { "text": "predictions", "label": "010" },
//       { "text": "predictive",  "label": "010" },
//       ...
//     ]
//
//  ‑ “0”  = unstressed (CMU 0 or 2)
//  ‑ “1”  = primary‑stress (CMU 1)
// ------------------------------------------

'use strict';
const fs   = require('fs');
const path = require('path');

const dictPath = process.argv[2]              // optional CLI arg
              || path.resolve(__dirname, '..', 'datasets', 'cmudict-0.7b');

const raw = fs.readFileSync(dictPath, 'utf8');
const entries = [];

raw.split('\n').forEach(l => {
  if (!l || l.startsWith(';;;')) return;

  const parts   = l.trim().split(/\s+/);
  const wordRaw = parts.shift();

  // 1 or 2 → '1' (stressed)  |  0 → '0' (unstressed)
  const stress = parts
    .map(p => p.match(/\d/))
    .filter(Boolean)
    .map(m => (m[0] === '0' ? '0' : '1'))   // <‑‑ tweaked line
    .join('');

  if (!stress) return;

  const word = wordRaw.replace(/\(\d+\)$/, '').toLowerCase();
  entries.push({ text: word, label: stress });
});

fs.writeFileSync('cmu_stress.json', JSON.stringify(entries, null, 2));
console.log(`Done – ${entries.length} items ➜ cmu_stress.json`);

