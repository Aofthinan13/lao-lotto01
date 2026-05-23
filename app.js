let data = [];
let lstmModel, gruModel;

let freqMap = {};
let heatMap = {};

// =====================
// LOAD DATA
// =====================
async function loadData() {
  const res = await fetch("data.json");
  data = await res.json();
  log("โหลดข้อมูลแล้ว: " + data.length);
}

// =====================
// ENCODE
// =====================
function encode(num) {
  return num.split("").map(n => Number(n) / 9);
}

// =====================
// FREQUENCY
// =====================
function buildFrequency() {
  freqMap = {};
  data.forEach(d => {
    for (let n of d.last3) {
      freqMap[n] = (freqMap[n] || 0) + 1;
    }
  });
}

// =====================
// HEATMAP
// =====================
function buildHeatmap() {
  heatMap = {};
  data.forEach(d => {
    heatMap[d.last3] = (heatMap[d.last3] || 0) + 1;
  });
}

// =====================
// LSTM MODEL
// =====================
function createLSTM() {
  const model = tf.sequential();

  model.add(tf.layers.lstm({
    units: 128,
    returnSequences: true,
    inputShape: [5, 3]
  }));

  model.add(tf.layers.lstm({ units: 64 }));

  model.add(tf.layers.dense({ units: 32, activation: "relu" }));

  model.add(tf.layers.dense({ units: 3, activation: "sigmoid" }));

  model.compile({
    optimizer: "adam",
    loss: "meanSquaredError"
  });

  return model;
}

// =====================
// GRU MODEL
// =====================
function createGRU() {
  const model = tf.sequential();

  model.add(tf.layers.gru({
    units: 64,
    returnSequences: true,
    inputShape: [5, 3]
  }));

  model.add(tf.layers.gru({ units: 32 }));

  model.add(tf.layers.dense({ units: 16, activation: "relu" }));

  model.add(tf.layers.dense({ units: 3, activation: "sigmoid" }));

  model.compile({
    optimizer: "adam",
    loss: "meanSquaredError"
  });

  return model;
}

// =====================
// DATASET
// =====================
function makeDataset() {
  const xs = [];
  const ys = [];
  const window = 5;

  for (let i = 0; i < data.length - window; i++) {

    let seq = [];

    for (let j = 0; j < window; j++) {
      seq.push(...encode(data[i + j].last3));
    }

    xs.push(seq);
    ys.push(encode(data[i + window].last3));
  }

  return {
    xs: tf.tensor3d(xs, [xs.length, window, 3]),
    ys: tf.tensor2d(ys)
  };
}

// =====================
// TRAIN ALL
// =====================
async function trainAll() {

  await loadData();

  buildFrequency();
  buildHeatmap();

  lstmModel = createLSTM();
  gruModel = createGRU();

  const { xs, ys } = makeDataset();

  log("Training LSTM...");
  await lstmModel.fit(xs, ys, { epochs: 80 });

  log("Training GRU...");
  await gruModel.fit(xs, ys, { epochs: 80 });

  log("Train complete!");
}

// =====================
// PREDICT (ENSEMBLE)
// =====================
function predict() {

  const window = 5;
  let input = [];

  for (let i = data.length - window; i < data.length; i++) {
    input.push(...encode(data[i].last3));
  }

  const tensor = tf.tensor3d([input], [1, window, 3]);

  const lstmOut = lstmModel.predict(tensor);
  const gruOut = gruModel.predict(tensor);

  const lstm = Array.from(lstmOut.dataSync());
  const gru = Array.from(gruOut.dataSync());

  let merged = lstm.map((v, i) => (v * 0.6 + gru[i] * 0.4));

  let nums = merged.map(v => Math.floor(v * 9));

  let scored = nums.map(n => {
    let freq = freqMap[n] || 1;
    let heat = heatMap[n] || 1;

    return {
      num: n,
      score: freq + heat
    };
  });

  scored.sort((a, b) => b.score - a.score);

  let result = scored.slice(0, 3).map(x => x.num).join("");

  let confidence = Math.min(100, scored[0].score * 10);

  document.getElementById("result").innerHTML =
    `🎯 ${result}<br>📊 confidence: ${confidence.toFixed(0)}%`;
}

// =====================
// LOG
// =====================
function log(t) {
  document.getElementById("log").innerText += t + "\n";
}
