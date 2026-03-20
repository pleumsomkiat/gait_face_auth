const express = require("express");
const path = require("path");
const { MongoClient } = require("mongodb");
const cors = require("cors");

const app = express();
app.use(express.json({ limit: "1mb" }));
app.use(cors());
app.use(express.static(path.join(__dirname, "public")));

const url = process.env.MONGO_URI;
const client = url ? new MongoClient(url) : null;

let db;

const previewEvents = [];
const PREVIEW_LIMIT = 30;

function normalizeConfidence(value) {
  if (value === undefined || value === null || value === "") {
    return null;
  }

  const numeric = Number(value);
  if (!Number.isFinite(numeric)) {
    return null;
  }

  return Math.max(0, Math.min(1, numeric));
}

function getNowParts() {
  const now = new Date();
  return {
    now,
    today: now.toISOString().slice(0, 10),
    timeNow: now.toTimeString().slice(0, 8),
  };
}

function addPreviewEvent(payload = {}, meta = {}) {
  const event = {
    id: `evt_${Date.now()}_${Math.random().toString(36).slice(2, 8)}`,
    received_at: new Date().toISOString(),
    user_id: payload.user_id || null,
    display_name: payload.display_name || null,
    confidence_score: normalizeConfidence(payload.confidence_score),
    source: payload.source || "scanner",
    preview_only: Boolean(payload.preview_only),
    status: meta.status || "รอประมวลผล",
    storage: meta.storage || "pending",
    message: meta.message || "",
    attend_date: meta.attend_date || null,
    time: meta.time || null,
  };

  previewEvents.unshift(event);
  if (previewEvents.length > PREVIEW_LIMIT) {
    previewEvents.length = PREVIEW_LIMIT;
  }

  return event;
}

function updatePreviewEvent(event, patch = {}) {
  Object.assign(event, patch);
  return event;
}

function isDbReady() {
  return Boolean(db);
}

async function start() {
  if (!client) {
    console.warn("MONGO_URI is not set. Preview mode will still work.");
    return;
  }

  try {
    await client.connect();
    db = client.db("attendanceDB");
    console.log("MongoDB connected");
  } catch (err) {
    console.error("MongoDB error", err);
  }
}
start();

async function getReportData() {
  if (!isDbReady()) {
    return [];
  }

  return db
    .collection("attendance")
    .aggregate([
      {
        $lookup: {
          from: "users",
          localField: "user_id",
          foreignField: "_id",
          as: "user",
        },
      },
      {
        $unwind: {
          path: "$user",
          preserveNullAndEmptyArrays: true,
        },
      },
      {
        $project: {
          user_id: 1,
          display_name: {
            $ifNull: ["$user.full_name", "$display_name"],
          },
          attend_date: 1,
          time: 1,
          status: 1,
          confidence_score: 1,
          source: 1,
        },
      },
      {
        $sort: {
          attend_date: -1,
          time: -1,
          _id: -1,
        },
      },
    ])
    .toArray();
}

app.post("/checkin", async (req, res) => {
  const payload = req.body || {};
  const { user_id, display_name, confidence_score, source, preview_only } = payload;

  if (!user_id) {
    return res.status(400).send({ message: "กรุณาส่ง user_id" });
  }

  const { now, today, timeNow } = getNowParts();
  const previewEvent = addPreviewEvent(payload, {
    status: "รับข้อมูลแล้ว",
    storage: preview_only ? "preview_only" : "pending",
    attend_date: today,
    time: timeNow,
  });

  if (preview_only) {
    updatePreviewEvent(previewEvent, {
      message: "รับข้อมูล preview แล้ว",
      status: "Preview",
    });

    return res.send({
      message: "รับข้อมูล preview แล้ว",
      stored: false,
      preview: previewEvent,
    });
  }

  if (!isDbReady()) {
    updatePreviewEvent(previewEvent, {
      storage: "db_unavailable",
      message: "ยังไม่พร้อมเชื่อมฐานข้อมูล",
      status: "Preview only",
    });

    return res.status(503).send({
      message: "ยังไม่พร้อมเชื่อมฐานข้อมูล",
      stored: false,
      preview: previewEvent,
    });
  }

  try {
    const already = await db
      .collection("attendance")
      .findOne({ user_id, attend_date: today });

    if (already) {
      updatePreviewEvent(previewEvent, {
        storage: "duplicate",
        message: "วันนี้เช็คชื่อแล้ว",
        status: already.status || "ซ้ำ",
        time: already.time || timeNow,
      });

      return res.send({
        message: "วันนี้เช็คชื่อแล้ว",
        status: already.status,
        stored: false,
        preview: previewEvent,
      });
    }

    const status = timeNow > "09:00:00" ? "สาย" : "ตรงเวลา";

    await db.collection("attendance").insertOne({
      user_id,
      display_name: display_name || user_id,
      attend_date: today,
      time: timeNow,
      status,
      confidence_score: normalizeConfidence(confidence_score),
      source: source || "scanner",
      created_at: now,
    });

    updatePreviewEvent(previewEvent, {
      storage: "stored",
      message: "เช็คชื่อสำเร็จ",
      status,
      time: timeNow,
    });

    res.send({
      message: "เช็คชื่อสำเร็จ",
      status,
      stored: true,
      preview: previewEvent,
    });
  } catch (err) {
    updatePreviewEvent(previewEvent, {
      storage: "error",
      message: err.message,
      status: "Error",
    });

    res.status(500).send({ error: err.message, preview: previewEvent });
  }
});

app.get("/api/preview", (req, res) => {
  res.send(previewEvents);
});

app.get("/api/report", async (req, res) => {
  try {
    const data = await getReportData();
    res.send(data);
  } catch (err) {
    res.status(500).send({ error: err.message });
  }
});

app.get("/report", async (req, res) => {
  try {
    const data = await getReportData();
    res.send(data);
  } catch (err) {
    res.status(500).send({ error: err.message });
  }
});

app.get("/", (req, res) => {
  res.sendFile(path.join(__dirname, "public", "index.html"));
});

const PORT = process.env.PORT || 3000;

app.listen(PORT, () => {
  console.log(`Server running on port ${PORT}`);
});
