const refreshButton = document.getElementById("refreshButton");
const previewFeed = document.getElementById("previewFeed");
const reportTableBody = document.getElementById("reportTableBody");
const previewCount = document.getElementById("previewCount");
const storedCount = document.getElementById("storedCount");
const previewOnlyCount = document.getElementById("previewOnlyCount");
const systemStatus = document.getElementById("systemStatus");
const lastRefresh = document.getElementById("lastRefresh");
const checkinForm = document.getElementById("checkinForm");
const submitResult = document.getElementById("submitResult");

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

function formatConfidence(value) {
  if (value === undefined || value === null || value === "") {
    return "-";
  }

  const numeric = Number(value);
  if (!Number.isFinite(numeric)) {
    return "-";
  }

  return numeric.toFixed(2);
}

function formatTime(value) {
  return value || "-";
}

function badgeClass(storage) {
  if (storage === "stored") {
    return "stored";
  }

  if (storage === "error" || storage === "db_unavailable") {
    return "error";
  }

  return "preview";
}

function renderPreview(previewItems) {
  previewCount.textContent = String(previewItems.length);
  previewOnlyCount.textContent = String(
    previewItems.filter((item) => item.preview_only || item.storage === "preview_only").length
  );

  if (!previewItems.length) {
    previewFeed.innerHTML = '<div class="empty-state">ยังไม่มีข้อมูลที่ยิงเข้ามา</div>';
    return;
  }

  previewFeed.innerHTML = previewItems
    .map((item) => {
      const displayName = item.display_name || item.user_id || "Unknown";
      return `
        <article class="feed-item">
          <div class="feed-top">
            <div>
              <div class="feed-name">${escapeHtml(displayName)}</div>
              <div class="feed-sub">user_id: ${escapeHtml(item.user_id || "-")}</div>
            </div>
            <span class="badge ${badgeClass(item.storage)}">${escapeHtml(item.storage || "pending")}</span>
          </div>
          <div class="feed-meta">
            <span class="badge ${badgeClass(item.storage)}">${escapeHtml(item.status || "-")}</span>
            <span class="muted">เวลา ${escapeHtml(formatTime(item.time))}</span>
            <span class="muted">ความมั่นใจ ${escapeHtml(formatConfidence(item.confidence_score))}</span>
            <span class="muted">source ${escapeHtml(item.source || "-")}</span>
          </div>
          <p class="muted">${escapeHtml(item.message || "รับข้อมูลจาก API แล้ว")}</p>
        </article>
      `;
    })
    .join("");
}

function renderReport(reportItems) {
  storedCount.textContent = String(reportItems.length);

  if (!reportItems.length) {
    reportTableBody.innerHTML = `
      <tr>
        <td colspan="7" class="empty-state">ยังไม่มีข้อมูลที่ถูกบันทึก</td>
      </tr>
    `;
    return;
  }

  reportTableBody.innerHTML = reportItems
    .map((item) => {
      const displayName = item.display_name || item.full_name || item.user_id || "-";
      return `
        <tr>
          <td>${escapeHtml(displayName)}</td>
          <td>${escapeHtml(item.user_id || "-")}</td>
          <td>${escapeHtml(item.attend_date || "-")}</td>
          <td>${escapeHtml(formatTime(item.time))}</td>
          <td>${escapeHtml(item.status || "-")}</td>
          <td>${escapeHtml(item.source || "-")}</td>
          <td>${escapeHtml(formatConfidence(item.confidence_score))}</td>
        </tr>
      `;
    })
    .join("");
}

async function loadDashboard() {
  try {
    const [previewResp, reportResp] = await Promise.all([
      fetch("/api/preview"),
      fetch("/api/report"),
    ]);

    if (!previewResp.ok || !reportResp.ok) {
      throw new Error("โหลดข้อมูลจาก API ไม่สำเร็จ");
    }

    const previewItems = await previewResp.json();
    const reportItems = await reportResp.json();

    renderPreview(previewItems);
    renderReport(reportItems);
    systemStatus.textContent = "พร้อมใช้งาน";
    lastRefresh.textContent = new Date().toLocaleTimeString("th-TH");
  } catch (err) {
    systemStatus.textContent = "เชื่อมต่อไม่สำเร็จ";
    lastRefresh.textContent = "-";
    previewFeed.innerHTML = `<div class="empty-state">${escapeHtml(err.message)}</div>`;
    reportTableBody.innerHTML = `
      <tr>
        <td colspan="7" class="empty-state">${escapeHtml(err.message)}</td>
      </tr>
    `;
  }
}

checkinForm.addEventListener("submit", async (event) => {
  event.preventDefault();

  const payload = {
    user_id: document.getElementById("userIdInput").value.trim(),
    display_name: document.getElementById("displayNameInput").value.trim(),
    confidence_score: document.getElementById("confidenceInput").value,
    source: document.getElementById("sourceInput").value.trim() || "dashboard-mock",
    preview_only: document.getElementById("previewOnlyInput").checked,
  };

  submitResult.textContent = "กำลังส่ง...";

  try {
    const response = await fetch("/checkin", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(payload),
    });

    const body = await response.json();
    submitResult.textContent = JSON.stringify(body, null, 2);
    await loadDashboard();
  } catch (err) {
    submitResult.textContent = err.message;
  }
});

refreshButton.addEventListener("click", loadDashboard);

loadDashboard();
window.setInterval(loadDashboard, 5000);
