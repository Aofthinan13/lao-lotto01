let data = [];
let model = {};

// 🔥 ดึงข้อมูลจริงจาก API
async function loadData() {
  try {
    const res = await fetch("https://api.apilotto.com/api/v1/laolottohistory", {
      headers: {
        "x-api-key": "YOUR_API_KEY"
      }
    });

    const json = await res.json();

    // แปลงข้อมูลให้ใช้กับ AI
    data = json.data.map(item => ({
      date: item.date,
      last4: item.laohistory4,
      last3: item.laohistory3,
      last2: item.laohistory2.top
    }));

    document.getElementById("history").innerText =
      JSON.stringify(data.slice(0, 10), null, 2);

  } catch (err) {
    console.log(err);
    alert("ดึงข้อมูลไม่สำเร็จ (เช็ค API KEY)");
  }
}

loadData();


// 🧠 AI TRAIN
function trainAI() {
  model = {};

  data.forEach(item => {
    let key = item.last3.slice(0, 2);

    if (!model[key]) model[key] = [];

    model[key].push(item.last3);
  });

  alert("AI วิเคราะห์ข้อมูลจริงเสร็จแล้ว!");
}


// 🎯 PREDICT
function predict() {
  let keys = Object.keys(model);
  if (keys.length === 0) {
    alert("ต้องกดวิเคราะห์ก่อน");
    return;
  }

  let key = keys[Math.floor(Math.random() * keys.length)];
  let list = model[key];

  let result = list[Math.floor(Math.random() * list.length)];

  document.getElementById("result").innerText =
    "🎯 เลขทำนาย: " + result;
}
