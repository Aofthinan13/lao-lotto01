let model;
let data = [];
let freqMap = {};

// 🔥 1) โหลดข้อมูลจริง (API + fallback)
async function fetchData() {
  try {
    const res = await fetch("https://api.apilotto.com/api/v1/laolottohistory", {
      headers: {
        "x-api-key": "YOUR_API_KEY"
      }
    });

    const json = await res.json();

    data = json.data.map(x => ({
      date: x.date,
      last3: x.laohistory3
    }));

    log("📡 ใช้ข้อมูล API จริง");

  } catch (e) {
    const res = await fetch("data.json");
    data = await res.json();

    log("⚠️ ใช้ fallback data.json");
  }
}


// 🔢 encode
function encode(num) {
  return num.split("").map(n => Number(n) / 9);
}


// 📊 frequency analysis (เลขออกบ่อย)
function buildFrequency() {
  freqMap = {};

  data.forEach(d => {
    for (let n of d.last3) {
      freqMap[n] = (freqMap[n] || 0) + 1;
    }
  });

  log("📊 built frequency map");
}


// 🧠 LSTM MODEL (V2 optimized)
function createModel() {
  const model = tf.sequential();

  model.add(tf.layers.lstm({
    units: 128,
    returnSequences: true,
    inputShape: [5, 3]
  }));

  model.add(tf.layers.dropout({ rate: 0.2 }));

  model.add(tf.layers.lstm({
    units: 64
  }));

  model.add(tf.layers.dense({
    units: 32,
    activation: "relu"
  }));

  model.add(tf.layers.dense({
    units: 3,
    activation: "sigmoid"
  }));

  model.compile({
    optimizer: "adam",
    loss: "meanSquaredError"
  });

  return model;
}


// 📊 dataset
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


// 🚀 TRAIN + AUTO UPDATE
async function loadAndTrain() {

  await fetchData();

  buildFrequency();

  model = createModel();

  const { xs, ys } = makeDataset();

  log("🚀 training V2 AI...");

  await model.fit(xs, ys, {
    epochs: 120,
    batchSize: 8,
    shuffle: true
  });

  log("✅ train complete");
}


// 🎯 HYBRID PREDICT (LSTM + Frequency)
function predict() {

  if (!model) return alert("ต้อง train ก่อน");

  const window = 5;
  let input = [];

  for (let i = data.length - window; i < data.length; i++) {
    input.push(...encode(data[i].last3));
  }

  const tensor = tf.tensor3d([input], [1, window, 3]);

  const out = model.predict(tensor);

  let lstmResult = Array.from(out.dataSync())
    .map(v => Math.floor(v * 9));

  // 🔥 frequency boost
  let boosted = lstmResult.map(n => {
    let freq = freqMap[n] || 1;
    return { num: n, score: freq };
  });

  boosted.sort((a, b) => b.score - a.score);

  let final = boosted.slice(0, 3).map(x => x.num).join("");

  let confidence =
    Math.min(100, boosted[0].score * 10).toFixed(0);

  document.getElementById("result").innerHTML =
    `🎯 ${final} <br>📊 confidence: ${confidence}%`;
}


// 🧾 log
function log(t) {
  document.getElementById("log").innerText += t + "\n";
}
