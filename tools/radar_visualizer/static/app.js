const state = {
  meta: null,
  frame: null,
  framePosition: 0,
  playing: false,
  playTimer: null,
  layers: {
    radar_front: true,
    lidar_front_2: true,
    lidar_front: true,
    annotations: true,
    detections: true,
    tracks: true,
  },
  colorMode: "doppler",
  range: 150,
  bevZoom: 1,
  bevPanX: 0,
  bevPanY: 0,
  bevDragging: false,
  bevPointerX: 0,
  bevPointerY: 0,
};

const colors = {
  radar_front: "#ffb547",
  lidar_front_2: "#42d6ff",
  lidar_front: "#816cff",
  annotations: "#65f09a",
  detections: "#ff5bbd",
  tracks: "#f4d35e",
};
const classColors = {
  Car: "#65f09a", Truck: "#ffb547", Bus: "#ff7d6e", Pedestrian: "#ff78c6",
  Cyclist: "#65d8ff", Tricycle: "#af8cff", Cone: "#f3e45d",
};
const $ = (id) => document.getElementById(id);

async function init() {
  bindControls();
  const response = await fetch("/api/meta");
  state.meta = await response.json();
  $("datasetStatus").textContent =
    `${state.meta.total_pkl_frames} PKL帧 · ${state.meta.available_frame_count} 帧有本地数据 · ${state.meta.complete_frame_count} 帧完整`;
  populateFrames();
  await loadFrame(0);
  window.addEventListener("resize", renderAll);
}

function bindControls() {
  $("prevButton").addEventListener("click", () => stepFrame(-1));
  $("nextButton").addEventListener("click", () => stepFrame(1));
  $("playButton").addEventListener("click", togglePlay);
  $("frameSelect").addEventListener("change", (event) => loadFrame(Number(event.target.value)));
  $("searchButton").addEventListener("click", searchFrame);
  $("frameSearch").addEventListener("keydown", (event) => {
    if (event.key === "Enter") searchFrame();
  });
  $("rangeSelect").addEventListener("change", (event) => {
    state.range = Number(event.target.value);
    resetBevView();
  });
  $("colorMode").addEventListener("change", (event) => { state.colorMode = event.target.value; renderAll(); });
  $("radarProjectionToggle").addEventListener("change", renderImageOverlay);
  $("zoomOutButton").addEventListener("click", () => setBevZoom(state.bevZoom / 1.25));
  $("zoomResetButton").addEventListener("click", resetBevView);
  $("zoomInButton").addEventListener("click", () => setBevZoom(state.bevZoom * 1.25));
  const bevCanvas = $("bevCanvas");
  bevCanvas.addEventListener("wheel", (event) => {
    event.preventDefault();
    setBevZoom(
      state.bevZoom * (event.deltaY < 0 ? 1.15 : 1 / 1.15),
      { x: event.offsetX, y: event.offsetY },
    );
  }, { passive: false });
  bevCanvas.addEventListener("pointerdown", (event) => {
    state.bevDragging = true;
    state.bevPointerX = event.clientX;
    state.bevPointerY = event.clientY;
    bevCanvas.setPointerCapture(event.pointerId);
    bevCanvas.classList.add("dragging");
  });
  bevCanvas.addEventListener("pointermove", (event) => {
    if (!state.bevDragging) return;
    state.bevPanX += event.clientX - state.bevPointerX;
    state.bevPanY += event.clientY - state.bevPointerY;
    state.bevPointerX = event.clientX;
    state.bevPointerY = event.clientY;
    renderBev();
  });
  const stopDragging = (event) => {
    state.bevDragging = false;
    if (bevCanvas.hasPointerCapture(event.pointerId)) bevCanvas.releasePointerCapture(event.pointerId);
    bevCanvas.classList.remove("dragging");
  };
  bevCanvas.addEventListener("pointerup", stopDragging);
  bevCanvas.addEventListener("pointercancel", stopDragging);
  document.querySelectorAll("[data-layer]").forEach((input) => {
    input.addEventListener("change", () => {
      state.layers[input.dataset.layer] = input.checked;
      renderAll();
    });
  });
}

function populateFrames() {
  $("frameSelect").innerHTML = state.meta.frames.map((frame, position) => {
    const available = frame.complete ? "完整" : Object.entries(frame.available).filter(([, ok]) => ok).map(([name]) => name).join(", ");
    return `<option value="${position}">PKL #${frame.index} · ${frame.sequence_id} · ${frame.boxes} GT · ${available}</option>`;
  }).join("");
}

async function loadFrame(position) {
  if (!state.meta.frames.length) return;
  state.framePosition = (position + state.meta.frames.length) % state.meta.frames.length;
  $("frameSelect").value = state.framePosition;
  $("loading").style.display = "flex";
  const index = state.meta.frames[state.framePosition].index;
  try {
    const response = await fetch(`/api/frame/${index}`);
    const frame = await response.json();
    if (!response.ok) throw new Error(frame.error || "读取帧失败");
    state.frame = frame;
    updateText();
    await updateImage();
    renderAll();
  } catch (error) {
    console.error(error);
    $("datasetStatus").textContent = error.message;
  } finally {
    $("loading").style.display = "none";
  }
}

function stepFrame(delta) { loadFrame(state.framePosition + delta); }
function searchFrame() {
  const query = $("frameSearch").value.trim();
  if (!query) {
    showToast("请输入 frame_id");
    return;
  }
  const exact = state.meta.frames.findIndex((frame) => frame.frame_id === query);
  if (exact >= 0) {
    loadFrame(exact);
    return;
  }
  const matches = state.meta.frames
    .map((frame, index) => ({ frame, index }))
    .filter(({ frame }) => frame.frame_id.includes(query));
  if (matches.length === 1) {
    loadFrame(matches[0].index);
    return;
  }
  showToast(matches.length > 1 ? `匹配到 ${matches.length} 帧，请输入更完整的 frame_id` : "未找到有前广图像的 frame_id");
}

function showToast(message) {
  const toast = $("toast");
  toast.textContent = message;
  toast.classList.add("visible");
  clearTimeout(showToast.timer);
  showToast.timer = setTimeout(() => toast.classList.remove("visible"), 2400);
}

function setBevZoom(value, focus = null) {
  const zoom = Math.max(.5, Math.min(8, value));
  const oldGeometry = getBevGeometry(state.bevZoom, state.bevPanX, state.bevPanY);
  const newGeometry = getBevGeometry(zoom, 0, 0);
  const anchor = focus || { x: oldGeometry.width / 2, y: oldGeometry.height / 2 };
  const ratio = newGeometry.scale / oldGeometry.scale;
  const newOriginX = anchor.x - (anchor.x - oldGeometry.originX) * ratio;
  const newOriginY = anchor.y - (anchor.y - oldGeometry.originY) * ratio;
  state.bevZoom = zoom;
  state.bevPanX = newOriginX - newGeometry.baseOriginX;
  state.bevPanY = newOriginY - newGeometry.baseOriginY;
  $("zoomResetButton").textContent = `${state.bevZoom.toFixed(1)}×`;
  renderBev();
}

function resetBevView() {
  state.bevZoom = 1;
  state.bevPanX = 0;
  state.bevPanY = 0;
  $("zoomResetButton").textContent = "1.0×";
  renderBev();
}

function togglePlay() {
  state.playing = !state.playing;
  $("playButton").textContent = state.playing ? "暂停" : "播放";
  clearInterval(state.playTimer);
  if (state.playing) state.playTimer = setInterval(() => stepFrame(1), 1400);
}

function updateText() {
  const frame = state.frame;
  const items = [
    ["PKL 索引", `#${frame.index}`],
    ["Sequence", frame.sequence_id],
    ["Frame ID", frame.frame_id],
    ["FW 相机时间", frame.camera_timestamp?.toFixed(3) ?? "-"],
    ["最终坐标系", frame.coordinate_frame || "-"],
    ["GT 时间差", `${number(frame.synchronization?.annotation_offset_ms)} ms`],
    ["Radar 时间差", syncOffset(frame.sensors.radar_front)],
    ["ATX 时间差", syncOffset(frame.sensors.lidar_front_2)],
    ["EM4 时间差", syncOffset(frame.sensors.lidar_front)],
    ["GT 数量", String(frame.annotations.length)],
    ["Overlay", overlaySummary(frame.overlays)],
    ["Tags", frame.tags.join(", ") || "-"],
  ];
  $("frameInfo").innerHTML = items.map(([key, value]) =>
    `<div class="info-row"><span>${key}</span><span title="${value}">${value}</span></div>`).join("");

  const statConfig = [
    ["radar_front", "4D Radar", "radar_front"],
    ["lidar_front_2", "ATX", "lidar_front_2"],
    ["lidar_front", "EM4", "lidar_front"],
  ];
  const stats = statConfig.map(([key, title, name]) => {
    const sensor = frame.sensors[key];
    return `<div class="stat"><span>${title}<b>${name}</b></span><strong style="color:${colors[key]}">${sensor.available ? formatCount(sensor.display_count) : "—"}</strong></div>`;
  });
  stats.push(`<div class="stat"><span>GROUND TRUTH<b>3D Boxes</b></span><strong style="color:${colors.annotations}">${frame.annotations.length}</strong></div>`);
  if (state.meta.overlay?.enabled) {
    const overlayStats = overlayCounts(frame.overlays);
    stats.push(`<div class="stat"><span>DET / TRACK<b>${state.meta.overlay.frame_count} overlay frames</b></span><strong style="color:${colors.tracks}">${overlayStats.detections}/${overlayStats.tracks}</strong></div>`);
  }
  $("stats").innerHTML = stats.join("");
  $("legend").innerHTML = Object.entries(colors).map(([name, color]) =>
    `<span><i class="dot" style="display:inline-block;background:${color};margin-right:4px"></i>${name}</span>`).join("");
  updateFeatureStats();
}

function updateFeatureStats() {
  const radar = state.frame?.sensors.radar_front;
  const mode = state.colorMode;
  let stats = radar?.feature_stats?.[mode];
  if (mode === "z" && radar?.available) stats = summarize(radar.points.z);
  $("featureName").textContent = mode;
  $("featureMin").textContent = number(stats?.min);
  $("featureMedian").textContent = number(stats?.median);
  $("featureMax").textContent = number(stats?.max);
  $("colorMin").textContent = number(stats?.min);
  $("colorMax").textContent = number(stats?.max);
}

async function updateImage() {
  const image = $("cameraImage");
  const empty = $("imageEmpty");
  if (!state.frame.image.available) {
    image.removeAttribute("src");
    image.style.display = "none";
    empty.style.display = "block";
    return;
  }
  image.style.display = "block";
  empty.style.display = "none";
  await new Promise((resolve) => {
    image.onload = resolve;
    image.onerror = resolve;
    image.src = `${state.frame.image.url}?t=${state.frame.index}`;
  });
}

function renderAll() {
  if (!state.frame) return;
  updateFeatureStats();
  renderImageOverlay();
  renderBev();
}

function setupCanvas(canvas, cssWidth, cssHeight) {
  const dpr = window.devicePixelRatio || 1;
  canvas.style.width = `${cssWidth}px`;
  canvas.style.height = `${cssHeight}px`;
  canvas.width = Math.max(1, Math.round(cssWidth * dpr));
  canvas.height = Math.max(1, Math.round(cssHeight * dpr));
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  return ctx;
}

function renderImageOverlay() {
  const image = $("cameraImage");
  const canvas = $("imageCanvas");
  if (!state.frame?.image.available || !image.clientWidth || !image.clientHeight) {
    setupCanvas(canvas, 1, 1).clearRect(0, 0, 1, 1);
    return;
  }
  const width = image.clientWidth;
  const height = image.clientHeight;
  const ctx = setupCanvas(canvas, width, height);
  const scaleX = width / state.frame.image.width;
  const scaleY = height / state.frame.image.height;
  ctx.clearRect(0, 0, width, height);

  if ($("radarProjectionToggle").checked && state.layers.radar_front) {
    const projection = state.frame.radar_projection;
    const values = projectionFeature(projection);
    const domain = robustDomain(values);
    ctx.globalAlpha = .78;
    for (let i = 0; i < projection.u.length; i += 1) {
      ctx.fillStyle = heatColor(values[i], domain);
      const size = Math.max(1.1, 3.1 - projection.depth[i] / 170);
      ctx.fillRect(projection.u[i] * scaleX, projection.v[i] * scaleY, size, size);
    }
    ctx.globalAlpha = 1;
  }

  if (state.layers.annotations) {
    ctx.lineWidth = 1.25;
    ctx.font = "10px ui-monospace, monospace";
    state.frame.annotations.forEach((annotation) => {
      const color = classColors[annotation.name] || colors.annotations;
      ctx.strokeStyle = color;
      ctx.fillStyle = color;
      annotation.image_segments.forEach((segment) => {
        ctx.beginPath();
        ctx.moveTo(segment[0][0] * scaleX, segment[0][1] * scaleY);
        ctx.lineTo(segment[1][0] * scaleX, segment[1][1] * scaleY);
        ctx.stroke();
      });
      const visible = annotation.image_segments.flat().find(([u, v]) =>
        u >= 0 && v >= 0 && u < state.frame.image.width && v < state.frame.image.height);
      if (visible) ctx.fillText(annotation.name, visible[0] * scaleX + 3, visible[1] * scaleY - 3);
    });
  }
  drawOverlayImage(ctx, scaleX, scaleY);
}

function projectionFeature(projection) {
  if (state.colorMode === "z") return projection.depth;
  return projection.features[state.colorMode] || projection.depth;
}

function renderBev() {
  const canvas = $("bevCanvas");
  const geometry = getBevGeometry();
  const { width, height, scale, originX, originY, viewRange } = geometry;
  const ctx = setupCanvas(canvas, width, height);
  ctx.fillStyle = "#03090b";
  ctx.fillRect(0, 0, width, height);

  const toScreen = (x, y) => [originX - y * scale, originY - x * scale];
  drawGrid(ctx, width, height, originX, originY, scale, viewRange);

  drawPointLayer(ctx, "lidar_front", toScreen, "rgba(129,108,255,.42)", 1.0);
  drawPointLayer(ctx, "lidar_front_2", toScreen, "rgba(66,214,255,.52)", 1.0);
  drawRadar(ctx, toScreen);
  if (state.layers.annotations) drawAnnotations(ctx, toScreen, scale);
  drawOverlays(ctx, toScreen, scale);
  drawEgo(ctx, originX, originY, scale);
}

function getBevGeometry(zoom = state.bevZoom, panX = state.bevPanX, panY = state.bevPanY) {
  const canvas = $("bevCanvas");
  const rect = canvas.parentElement.getBoundingClientRect();
  const width = Math.max(320, rect.width);
  const height = Math.max(300, rect.height);
  const margin = { top: 22, right: 24, bottom: 30, left: 24 };
  const viewRange = state.range / zoom;
  const lateral = viewRange <= 80 ? 42 : viewRange <= 150 ? 70 : viewRange <= 200 ? 90 : 120;
  const scale = Math.min((height - margin.top - margin.bottom) / viewRange, (width - margin.left - margin.right) / (lateral * 2));
  const baseOriginX = width / 2;
  const baseOriginY = height - margin.bottom;
  return {
    width, height, scale, viewRange, baseOriginX, baseOriginY,
    originX: baseOriginX + panX,
    originY: baseOriginY + panY,
  };
}

function drawGrid(ctx, width, height, originX, originY, scale, viewRange) {
  const interval = viewRange <= 80 ? 10 : viewRange <= 200 ? 25 : 50;
  ctx.lineWidth = 1;
  ctx.font = "9px ui-monospace, monospace";
  const minForward = Math.floor((originY - height) / scale / interval) * interval;
  const maxForward = Math.ceil(originY / scale / interval) * interval;
  for (let x = minForward; x <= maxForward; x += interval) {
    const y = originY - x * scale;
    ctx.strokeStyle = x === 0 ? "#53666c" : "#142329";
    ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(width, y); ctx.stroke();
    ctx.fillStyle = "#5e747b"; ctx.fillText(`${x}m`, 5, y - 4);
  }
  const minLateral = Math.floor((originX - width) / scale / interval) * interval;
  const maxLateral = Math.ceil(originX / scale / interval) * interval;
  for (let lateralValue = minLateral; lateralValue <= maxLateral; lateralValue += interval) {
    const x = originX - lateralValue * scale;
    ctx.strokeStyle = lateralValue === 0 ? "#354b52" : "#102025";
    ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, height); ctx.stroke();
  }
}

function drawPointLayer(ctx, key, toScreen, color, size) {
  if (!state.layers[key]) return;
  const sensor = state.frame.sensors[key];
  if (!sensor?.available) return;
  const { x, y } = sensor.points;
  ctx.fillStyle = color;
  for (let i = 0; i < x.length; i += 1) {
    const [sx, sy] = toScreen(x[i], y[i]);
    ctx.fillRect(sx, sy, size, size);
  }
}

function drawRadar(ctx, toScreen) {
  if (!state.layers.radar_front) return;
  const radar = state.frame.sensors.radar_front;
  if (!radar?.available) return;
  const values = state.colorMode === "z" ? radar.points.z : (radar.features[state.colorMode] || radar.points.z);
  const domain = robustDomain(values);
  const { x, y } = radar.points;
  for (let i = 0; i < x.length; i += 1) {
    const [sx, sy] = toScreen(x[i], y[i]);
    ctx.fillStyle = heatColor(values[i], domain);
    ctx.fillRect(sx - 1, sy - 1, 2.2, 2.2);
  }
}

function drawAnnotations(ctx, toScreen, scale) {
  ctx.font = "9px ui-monospace, monospace";
  ctx.lineWidth = 1.5;
  state.frame.annotations.forEach((annotation) => {
    const [x, y, , length, width, , yaw] = annotation.box;
    const [cx, cy] = toScreen(x, y);
    const color = classColors[annotation.name] || colors.annotations;
    ctx.save();
    ctx.translate(cx, cy);
    ctx.rotate(-yaw);
    ctx.strokeStyle = color;
    ctx.strokeRect(-width * scale / 2, -length * scale / 2, width * scale, length * scale);
    ctx.beginPath();
    ctx.moveTo(0, -length * scale / 2);
    ctx.lineTo(0, -length * scale / 2 - Math.min(8, length * scale / 2));
    ctx.stroke();
    ctx.restore();
    ctx.fillStyle = color;
    ctx.fillText(annotation.name, cx + 4, cy - 4);
  });
}

function drawOverlays(ctx, toScreen, scale) {
  (state.frame.overlays || []).forEach((overlay) => {
    const isTrack = isTrackOverlay(overlay);
    if (isTrack && !state.layers.tracks) return;
    if (!isTrack && !state.layers.detections) return;
    const [x, y, , length, width, , yaw] = overlay.box;
    const [cx, cy] = toScreen(x, y);
    const color = isTrack ? colors.tracks : colors.detections;
    ctx.save();
    ctx.translate(cx, cy);
    ctx.rotate(-yaw);
    ctx.strokeStyle = color;
    ctx.lineWidth = isTrack ? 2.0 : 1.4;
    ctx.setLineDash(isTrack ? [] : [5, 4]);
    ctx.strokeRect(-width * scale / 2, -length * scale / 2, width * scale, length * scale);
    ctx.beginPath();
    ctx.moveTo(0, -length * scale / 2);
    ctx.lineTo(0, -length * scale / 2 - Math.min(10, length * scale / 2));
    ctx.stroke();
    ctx.restore();
    ctx.setLineDash([]);
    ctx.fillStyle = color;
    ctx.font = "9px ui-monospace, monospace";
    const score = Number.isFinite(overlay.score) ? ` ${overlay.score.toFixed(2)}` : "";
    const track = overlay.track_id ? ` #${overlay.track_id}` : "";
    ctx.fillText(`${overlay.name}${track}${score}`, cx + 4, cy + 8);
  });
}

function drawOverlayImage(ctx, scaleX, scaleY) {
  ctx.font = "10px ui-monospace, monospace";
  (state.frame.overlays || []).forEach((overlay) => {
    const isTrack = isTrackOverlay(overlay);
    if (isTrack && !state.layers.tracks) return;
    if (!isTrack && !state.layers.detections) return;
    const color = isTrack ? colors.tracks : colors.detections;
    ctx.strokeStyle = color;
    ctx.fillStyle = color;
    ctx.lineWidth = isTrack ? 1.8 : 1.2;
    ctx.setLineDash(isTrack ? [] : [5, 4]);
    overlay.image_segments.forEach((segment) => {
      ctx.beginPath();
      ctx.moveTo(segment[0][0] * scaleX, segment[0][1] * scaleY);
      ctx.lineTo(segment[1][0] * scaleX, segment[1][1] * scaleY);
      ctx.stroke();
    });
    ctx.setLineDash([]);
    const visible = overlay.image_segments.flat().find(([u, v]) =>
      u >= 0 && v >= 0 && u < state.frame.image.width && v < state.frame.image.height);
    if (visible) {
      const track = overlay.track_id ? ` #${overlay.track_id}` : "";
      ctx.fillText(`${overlay.name}${track}`, visible[0] * scaleX + 3, visible[1] * scaleY + 11);
    }
  });
}

function drawEgo(ctx, x, y, scale) {
  const width = Math.max(7, 1.9 * scale);
  const length = Math.max(14, 4.8 * scale);
  ctx.fillStyle = "#dfeaed";
  ctx.strokeStyle = "#071013";
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.roundRect(x - width / 2, y - length * .8, width, length, 2);
  ctx.fill(); ctx.stroke();
  ctx.fillStyle = "#071013";
  ctx.beginPath();
  ctx.moveTo(x, y - length * .86);
  ctx.lineTo(x - 3, y - length * .66);
  ctx.lineTo(x + 3, y - length * .66);
  ctx.closePath(); ctx.fill();
}

function robustDomain(values) {
  if (!values?.length) return [0, 1];
  const sampled = [];
  const step = Math.max(1, Math.floor(values.length / 2000));
  for (let i = 0; i < values.length; i += step) if (Number.isFinite(values[i])) sampled.push(values[i]);
  sampled.sort((a, b) => a - b);
  if (!sampled.length) return [0, 1];
  const low = sampled[Math.floor(sampled.length * .02)];
  const high = sampled[Math.floor(sampled.length * .98)];
  return high > low ? [low, high] : [low - 1, high + 1];
}

function heatColor(value, [low, high]) {
  const t = Math.max(0, Math.min(1, (value - low) / (high - low || 1)));
  const stops = [[115, 93, 255], [52, 213, 255], [245, 223, 77], [255, 85, 85]];
  const scaled = t * (stops.length - 1);
  const index = Math.min(stops.length - 2, Math.floor(scaled));
  const f = scaled - index;
  const rgb = stops[index].map((channel, i) => Math.round(channel + (stops[index + 1][i] - channel) * f));
  return `rgb(${rgb.join(",")})`;
}

function summarize(values) {
  const sorted = [...values].filter(Number.isFinite).sort((a, b) => a - b);
  if (!sorted.length) return null;
  return { min: sorted[0], median: sorted[Math.floor(sorted.length / 2)], max: sorted.at(-1) };
}
function isTrackOverlay(overlay) {
  return Boolean(overlay.track_id) || overlay.source === "track" || overlay.source === "det_track";
}
function overlayCounts(overlays = []) {
  return overlays.reduce((counts, overlay) => {
    if (isTrackOverlay(overlay)) counts.tracks += 1;
    else counts.detections += 1;
    return counts;
  }, { detections: 0, tracks: 0 });
}
function overlaySummary(overlays = []) {
  if (!state.meta.overlay?.enabled) return "-";
  const counts = overlayCounts(overlays);
  return `${counts.detections} det / ${counts.tracks} track`;
}
function syncOffset(sensor) {
  const offset = sensor?.synchronization?.offset_ms;
  if (!offset) return "-";
  if (Math.abs(offset.max - offset.min) < .01) return `${number(offset.median)} ms`;
  return `${number(offset.min)} ~ ${number(offset.max)} ms`;
}
function number(value) { return Number.isFinite(value) ? value.toFixed(2) : "-"; }
function formatCount(value) { return new Intl.NumberFormat("zh-CN", { notation: "compact", maximumFractionDigits: 1 }).format(value); }

init().catch((error) => {
  console.error(error);
  $("datasetStatus").textContent = error.message;
});
