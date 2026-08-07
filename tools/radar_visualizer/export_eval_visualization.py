#!/usr/bin/env python3
"""Export an interactive HR-4D evaluation review bundle.

The bundle is static HTML plus JSON/JPEG assets. It highlights FP, FN,
localization-error and TP context while preserving the original visualizer's
camera, Radar, ATX LiDAR and EM4 LiDAR views.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import pickle
import sys
import types
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SERVER_PATH = Path(__file__).resolve().with_name("server.py")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.radar_visualizer.eval_diff import (  # noqa: E402
    EvalDiffConfig,
    build_eval_report,
    overlay_frames_from_report,
    select_review_frames,
)


DEFAULT_INFOS = ROOT / "data" / "1000_original_data" / "splits" / "hr4d_1000_v1" / "infos_test_200.pkl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=ROOT / "data" / "1000_original_data")
    parser.add_argument("--infos", type=Path, default=DEFAULT_INFOS)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "output" / "weikang_eval_review")
    parser.add_argument("--score-threshold", type=float, default=0.1)
    parser.add_argument("--match-lateral-threshold", type=float, default=2.0)
    parser.add_argument("--loc-warning-threshold", type=float, default=1.0)
    parser.add_argument("--max-cases", type=int, default=80)
    parser.add_argument("--max-frames", type=int, default=40)
    parser.add_argument("--indent-json", action="store_true")
    return parser.parse_args()


def load_pickle(path: Path):
    with path.open("rb") as stream:
        return pickle.load(stream)


def load_visualizer_server():
    if "flask" not in sys.modules:
        flask = types.ModuleType("flask")
        flask.Flask = object
        flask.Response = object
        flask.abort = lambda *args, **kwargs: None
        flask.send_from_directory = lambda *args, **kwargs: None
        sys.modules["flask"] = flask
    spec = importlib.util.spec_from_file_location("hr4d_radar_visualizer_server", SERVER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to import {SERVER_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_json(path: Path, payload, indent: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, allow_nan=False, indent=2 if indent else None)


def frame_asset_id(frame_index: int) -> str:
    return f"frame_{frame_index:06d}"


def build_bundle(args: argparse.Namespace) -> dict:
    infos = load_pickle(args.infos)
    predictions = load_pickle(args.predictions)
    config = EvalDiffConfig(
        score_threshold=args.score_threshold,
        match_lateral_threshold=args.match_lateral_threshold,
        loc_warning_threshold=args.loc_warning_threshold,
        max_cases=args.max_cases,
        max_frames=args.max_frames,
    )
    report = build_eval_report(infos, predictions, config)
    selected_cases, selected_frame_indices = select_review_frames(report, args.max_cases, args.max_frames)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    assets_dir = args.output_dir / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)
    overlay_path = args.output_dir / "eval_overlay.json"
    write_json(overlay_path, overlay_frames_from_report(report), indent=args.indent_json)

    server = load_visualizer_server()
    dataset = server.Dataset(args.data_root, args.infos, overlay_path)
    frame_by_index = {frame["frame_index"]: frame for frame in report["frames"]}
    cases_by_frame: dict[int, list[dict]] = {}
    for case in selected_cases:
        cases_by_frame.setdefault(case["frame_index"], []).append(case)

    review_frames = []
    for frame_index in selected_frame_indices:
        asset_id = frame_asset_id(frame_index)
        frame_payload = dataset.frame(frame_index)
        frame_payload["eval"] = frame_by_index[frame_index]
        frame_payload["review_cases"] = cases_by_frame.get(frame_index, [])
        image_bytes = dataset.image_bytes(frame_index)
        image_path = assets_dir / f"{asset_id}.jpg"
        image_path.write_bytes(image_bytes)
        frame_payload["image"]["asset"] = f"assets/{asset_id}.jpg"
        write_json(assets_dir / f"{asset_id}.json", frame_payload, indent=args.indent_json)
        review_frames.append(
            {
                "asset_id": asset_id,
                "frame_index": frame_index,
                "frame_id": frame_payload["frame_id"],
                "sequence_id": frame_payload["sequence_id"],
                "summary": frame_by_index[frame_index]["summary"],
                "cases": cases_by_frame.get(frame_index, []),
                "image": f"assets/{asset_id}.jpg",
                "frame_json": f"assets/{asset_id}.json",
            }
        )

    index = {
        "schema": "hr4d_eval_review_bundle_v1",
        "created_from": {
            "infos": str(args.infos),
            "predictions": str(args.predictions),
            "data_root": str(args.data_root),
        },
        "config": config.__dict__,
        "summary": report["summary"],
        "review_summary": {
            "selected_cases": len(selected_cases),
            "selected_frames": len(review_frames),
        },
        "frames": review_frames,
        "cases": selected_cases,
        "all_case_count": len(report["cases"]),
    }
    write_json(assets_dir / "index.json", index, indent=True)
    write_json(args.output_dir / "eval_diff.json", report, indent=args.indent_json)
    (args.output_dir / "index.html").write_text(HTML_TEMPLATE, encoding="utf-8")
    return index


HTML_TEMPLATE = r"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>HR-4D Eval Review</title>
  <style>
    :root {
      color-scheme: dark;
      --bg: #111317;
      --panel: #1b1e24;
      --line: #313743;
      --muted: #9aa3b2;
      --text: #eef2f7;
      --gt: #59d98e;
      --tp: #37d6ff;
      --fp: #ff4f9a;
      --fn: #ff6b4a;
      --loc: #f4d35e;
      --ignore: #7b8494;
      --radar: #ffd24a;
      --atx: #4cc9ff;
      --em4: #b388ff;
    }
    * { box-sizing: border-box; }
    body { margin: 0; background: var(--bg); color: var(--text); font: 14px/1.4 Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }
    header { height: 58px; display: flex; align-items: center; justify-content: space-between; padding: 0 18px; border-bottom: 1px solid var(--line); background: #171a20; }
    h1 { margin: 0; font-size: 18px; font-weight: 650; letter-spacing: 0; }
    button, select, label.toggle { border: 1px solid var(--line); background: #222832; color: var(--text); border-radius: 6px; padding: 7px 9px; }
    button { cursor: pointer; }
    button.active { outline: 2px solid #64748b; }
    main { height: calc(100vh - 58px); display: grid; grid-template-columns: 360px 1fr; min-height: 720px; }
    aside { border-right: 1px solid var(--line); overflow: hidden; display: grid; grid-template-rows: auto auto 1fr; }
    .summary { padding: 14px; display: grid; grid-template-columns: repeat(3, 1fr); gap: 8px; border-bottom: 1px solid var(--line); }
    .metric { background: var(--panel); border: 1px solid var(--line); border-radius: 8px; padding: 8px; }
    .metric b { display: block; font-size: 20px; }
    .metric span { color: var(--muted); font-size: 11px; text-transform: uppercase; }
    .filters { padding: 10px 12px; display: flex; gap: 6px; flex-wrap: wrap; border-bottom: 1px solid var(--line); }
    .case-list { overflow: auto; padding: 10px; }
    .case { width: 100%; text-align: left; margin-bottom: 7px; border-radius: 8px; background: var(--panel); }
    .case strong { display: flex; justify-content: space-between; gap: 8px; }
    .case small { color: var(--muted); display: block; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
    .case[data-type="FP"] { border-color: color-mix(in srgb, var(--fp), #000 40%); }
    .case[data-type="FN"] { border-color: color-mix(in srgb, var(--fn), #000 40%); }
    .case[data-type="LOC"] { border-color: color-mix(in srgb, var(--loc), #000 40%); }
    .content { display: grid; grid-template-rows: auto 1fr 220px; min-width: 900px; }
    .toolbar { padding: 10px 12px; border-bottom: 1px solid var(--line); display: flex; align-items: center; gap: 8px; flex-wrap: wrap; }
    .toolbar .spacer { flex: 1; }
    .stage { display: grid; grid-template-columns: minmax(360px, .72fr) minmax(680px, 1.6fr); grid-template-rows: minmax(320px, 1fr) minmax(150px, .34fr); gap: 10px; padding: 10px; min-height: 0; }
    .image-panel { grid-column: 1; grid-row: 1; }
    .view3d-panel { grid-column: 2; grid-row: 1 / span 2; }
    .bev-panel { grid-column: 1; grid-row: 2; }
    .panel { position: relative; background: var(--panel); border: 1px solid var(--line); border-radius: 8px; min-height: 0; overflow: hidden; }
    .panel-title { height: 34px; display: flex; align-items: center; justify-content: space-between; gap: 8px; padding: 0 10px; color: var(--muted); border-bottom: 1px solid var(--line); font-size: 12px; }
    .panel-actions { display: flex; align-items: center; gap: 4px; }
    .view-button { min-width: 34px; padding: 3px 6px; border-radius: 5px; font-size: 11px; line-height: 1.1; }
    .view-button.active { outline: 1px solid #8aa2c8; background: #2c3544; color: var(--text); }
    .view3d-controls { height: 66px; display: grid; grid-template-columns: auto repeat(4, minmax(96px, 1fr)); gap: 8px; align-items: center; padding: 7px 10px; border-bottom: 1px solid var(--line); background: #171b22; }
    .view3d-mode { display: flex; gap: 4px; }
    .slider-control { display: grid; grid-template-columns: auto 1fr 34px; align-items: center; gap: 6px; color: var(--muted); font-size: 11px; }
    .slider-control input { min-width: 0; accent-color: #8aa2c8; }
    .slider-value { color: var(--text); text-align: right; font-variant-numeric: tabular-nums; }
    canvas { display: block; width: 100%; height: calc(100% - 34px); background: #101216; touch-action: none; }
    .view3d-panel canvas { height: calc(100% - 100px); }
    .details { border-top: 1px solid var(--line); padding: 10px; display: grid; grid-template-columns: 1fr 1fr; gap: 10px; overflow: hidden; }
    .info { background: var(--panel); border: 1px solid var(--line); border-radius: 8px; padding: 10px; overflow: auto; }
    .info h2 { margin: 0 0 8px; font-size: 14px; }
    .info-row { display: flex; justify-content: space-between; gap: 12px; padding: 3px 0; border-bottom: 1px solid #252b34; }
    .info-row span:first-child { color: var(--muted); }
    .legend { display: flex; gap: 10px; align-items: center; color: var(--muted); }
    .dot { width: 10px; height: 10px; border-radius: 50%; display: inline-block; margin-right: 4px; }
    .empty { color: var(--muted); padding: 20px; }
  </style>
</head>
<body>
  <header>
    <h1>HR-4D Eval Review</h1>
    <div class="legend">
      <span><i class="dot" style="background:var(--gt)"></i>GT</span>
      <span><i class="dot" style="background:var(--tp)"></i>TP Pred</span>
      <span><i class="dot" style="background:var(--fp)"></i>FP Pred</span>
      <span><i class="dot" style="background:var(--fn)"></i>FN GT</span>
      <span><i class="dot" style="background:var(--ignore)"></i>IGNORE GT</span>
      <span><i class="dot" style="background:var(--radar)"></i>Radar</span>
      <span><i class="dot" style="background:var(--atx)"></i>ATX</span>
      <span><i class="dot" style="background:var(--em4)"></i>EM4</span>
    </div>
  </header>
  <main>
    <aside>
      <section class="summary" id="summary"></section>
      <section class="filters">
        <button data-filter="ALL" class="active">ALL</button>
        <button data-filter="FN">FN</button>
        <button data-filter="FP">FP</button>
        <button data-filter="LOC">LOC</button>
      </section>
      <section class="case-list" id="caseList"></section>
    </aside>
    <section class="content">
      <div class="toolbar">
        <button id="prevFrame">‹ Frame</button>
        <select id="frameSelect"></select>
        <button id="nextFrame">Frame ›</button>
        <button id="resetView">Reset BEV</button>
        <button id="reset3dView">Reset 3D</button>
        <select id="rangePreset">
          <option value="80">80m</option>
          <option value="150">150m</option>
          <option value="220" selected>220m</option>
        </select>
        <select id="radarColorMode" title="Radar point color mode">
          <option value="fixed" selected>Radar color</option>
          <option value="rcs">Radar RCS</option>
          <option value="doppler">Radar Doppler</option>
          <option value="speed">Radar AbsV</option>
        </select>
        <span class="spacer"></span>
        <label class="toggle"><input type="checkbox" data-layer="radar" checked> Radar</label>
        <label class="toggle"><input type="checkbox" data-layer="atx" checked> ATX</label>
        <label class="toggle"><input type="checkbox" data-layer="em4" checked> EM4</label>
        <label class="toggle"><input type="checkbox" data-layer="gt" checked> GT</label>
        <label class="toggle"><input type="checkbox" data-layer="pred" checked> Pred</label>
        <label class="toggle"><input type="checkbox" data-layer="projection" checked> Image Projection</label>
      </div>
      <div class="stage">
        <article class="panel image-panel"><div class="panel-title">Camera + Radar Projection + GT/Pred Projection</div><canvas id="imageCanvas"></canvas></article>
        <article class="panel view3d-panel">
          <div class="panel-title">
            <span>Primary 3D Point Cloud + GT/Pred Boxes</span>
            <span class="panel-actions">
              <button class="view-button active" data-view3d="iso" title="Isometric view">ISO</button>
              <button class="view-button" data-view3d="top" title="Top view">TOP</button>
              <button class="view-button" data-view3d="front" title="Front view">FRONT</button>
              <button class="view-button" data-view3d="left" title="Left view">LEFT</button>
              <button class="view-button" data-view3d="fit" title="Fit frame">FIT</button>
              <button class="view-button" id="focus3dView" title="Focus selected case">FOCUS</button>
            </span>
          </div>
          <div class="view3d-controls">
            <span class="view3d-mode">
              <button class="view-button active" data-view3d-mode="rotate">ORBIT</button>
              <button class="view-button" data-view3d-mode="pan">PAN</button>
            </span>
            <label class="slider-control">Yaw <input id="view3dYaw" type="range" min="-180" max="180" step="1"><span class="slider-value" id="view3dYawValue">0</span></label>
            <label class="slider-control">Pitch <input id="view3dPitch" type="range" min="-20" max="87" step="1"><span class="slider-value" id="view3dPitchValue">0</span></label>
            <label class="slider-control">Zoom <input id="view3dDistance" type="range" min="25" max="520" step="5"><span class="slider-value" id="view3dDistanceValue">0</span></label>
            <label class="slider-control">Z <input id="view3dZScale" type="range" min="1" max="4" step="0.25"><span class="slider-value" id="view3dZScaleValue">1x</span></label>
          </div>
          <canvas id="view3dCanvas"></canvas>
        </article>
        <article class="panel bev-panel"><div class="panel-title">Aux BEV Context</div><canvas id="bevCanvas"></canvas></article>
      </div>
      <div class="details">
        <article class="info"><h2>Frame / Case</h2><div id="frameInfo"></div></article>
        <article class="info"><h2>Selected Case Detail</h2><pre id="caseDetail" class="empty">Select a case</pre></article>
      </div>
    </section>
  </main>
  <script>
    const state = {
      index: null, frame: null, image: null, framePos: 0, selectedCase: null, filter: "ALL",
      layers: { radar: true, atx: true, em4: true, gt: true, pred: true, projection: true },
      radarColorMode: "fixed",
      view: { range: 220, zoom: 1, panX: 0, panY: 0, dragging: false, lastX: 0, lastY: 0 },
      view3d: { preset: "iso", mode: "rotate", yaw: -0.92, pitch: 0.34, distance: 135, targetX: 55, targetY: 0, targetZ: 1.5, zScale: 1.8, dragging: false, dragMode: "rotate", lastX: 0, lastY: 0 },
    };
    const colors = { gt: "#59d98e", tp: "#37d6ff", fp: "#ff4f9a", fn: "#ff6b4a", loc: "#f4d35e", ignore: "#7b8494", radar: "#ffd24a", atx: "#4cc9ff", em4: "#b388ff" };
    const view3dPresets = {
      iso: { yaw: -0.92, pitch: 0.34, distance: 135, targetX: 55, targetY: 0, targetZ: 1.5, zScale: 1.8 },
      top: { yaw: -1.5708, pitch: 1.48, distance: 190, targetX: 58, targetY: 0, targetZ: 0, zScale: 1 },
      front: { yaw: 0, pitch: 0.04, distance: 150, targetX: 70, targetY: 0, targetZ: 2.0, zScale: 2 },
      left: { yaw: -1.5708, pitch: 0.04, distance: 145, targetX: 58, targetY: 0, targetZ: 2.0, zScale: 2 },
    };
    const $ = (id) => document.getElementById(id);

    async function init() {
      state.index = await (await fetch("assets/index.json")).json();
      renderSummary();
      bind();
      populateFrames();
      renderCaseList();
      await loadFrame(0);
    }

    function bind() {
      document.querySelectorAll("[data-filter]").forEach(btn => btn.onclick = () => {
        document.querySelectorAll("[data-filter]").forEach(b => b.classList.remove("active"));
        btn.classList.add("active"); state.filter = btn.dataset.filter; renderCaseList();
      });
      document.querySelectorAll("[data-layer]").forEach(input => input.onchange = () => {
        state.layers[input.dataset.layer] = input.checked; renderAll();
      });
      $("frameSelect").onchange = e => loadFrame(Number(e.target.value));
      $("prevFrame").onclick = () => loadFrame(state.framePos - 1);
      $("nextFrame").onclick = () => loadFrame(state.framePos + 1);
      $("resetView").onclick = () => { state.view.zoom = 1; state.view.panX = 0; state.view.panY = 0; renderBev(); };
      $("reset3dView").onclick = () => { set3dView("iso"); };
      $("focus3dView").onclick = () => { focus3dOnSelected(); };
      document.querySelectorAll("[data-view3d]").forEach(btn => btn.onclick = () => {
        if (btn.dataset.view3d === "fit") fit3dToFrame();
        else set3dView(btn.dataset.view3d);
      });
      bind3dControlInputs();
      $("radarColorMode").onchange = e => { state.radarColorMode = e.target.value; renderAll(); };
      $("rangePreset").onchange = e => { state.view.range = Number(e.target.value); state.view.zoom = 1; state.view.panX = 0; state.view.panY = 0; render3d(); renderBev(); };
      bindBevControls();
      bind3dControls();
      update3dViewButtons();
      window.addEventListener("resize", renderAll);
    }

    function bindBevControls() {
      const canvas = $("bevCanvas");
      canvas.addEventListener("wheel", e => { e.preventDefault(); state.view.zoom = Math.max(.5, Math.min(10, state.view.zoom * (e.deltaY < 0 ? 1.15 : 1/1.15))); renderBev(); }, { passive: false });
      canvas.addEventListener("pointerdown", e => { state.view.dragging = true; state.view.lastX = e.clientX; state.view.lastY = e.clientY; canvas.setPointerCapture(e.pointerId); });
      canvas.addEventListener("pointermove", e => { if (!state.view.dragging) return; state.view.panX += e.clientX - state.view.lastX; state.view.panY += e.clientY - state.view.lastY; state.view.lastX = e.clientX; state.view.lastY = e.clientY; renderBev(); });
      canvas.addEventListener("pointerup", e => { state.view.dragging = false; if (canvas.hasPointerCapture(e.pointerId)) canvas.releasePointerCapture(e.pointerId); });
    }

    function bind3dControls() {
      const canvas = $("view3dCanvas");
      canvas.addEventListener("contextmenu", e => e.preventDefault());
      canvas.addEventListener("wheel", e => {
        e.preventDefault();
        state.view3d.distance = clamp(state.view3d.distance * (e.deltaY < 0 ? 0.9 : 1.1), 25, 520);
        state.view3d.preset = "custom";
        update3dViewButtons();
        render3d();
      }, { passive: false });
      canvas.addEventListener("pointerdown", e => {
        e.preventDefault();
        state.view3d.dragging = true;
        state.view3d.dragMode = e.button === 1 || e.button === 2 || e.shiftKey ? "pan" : state.view3d.mode;
        state.view3d.lastX = e.clientX;
        state.view3d.lastY = e.clientY;
        canvas.setPointerCapture(e.pointerId);
      });
      canvas.addEventListener("pointermove", e => {
        if (!state.view3d.dragging) return;
        const dx = e.clientX - state.view3d.lastX, dy = e.clientY - state.view3d.lastY;
        state.view3d.lastX = e.clientX; state.view3d.lastY = e.clientY;
        if (state.view3d.dragMode === "pan") {
          pan3d(dx, dy);
        } else {
          state.view3d.yaw -= dx * 0.006;
          state.view3d.pitch = clamp(state.view3d.pitch - dy * 0.006, -0.35, 1.52);
        }
        state.view3d.preset = "custom";
        update3dViewButtons();
        render3d();
      });
      canvas.addEventListener("pointerup", e => { state.view3d.dragging = false; if (canvas.hasPointerCapture(e.pointerId)) canvas.releasePointerCapture(e.pointerId); });
      canvas.addEventListener("pointercancel", e => { state.view3d.dragging = false; if (canvas.hasPointerCapture(e.pointerId)) canvas.releasePointerCapture(e.pointerId); });
    }

    function bind3dControlInputs() {
      document.querySelectorAll("[data-view3d-mode]").forEach(btn => btn.onclick = () => {
        state.view3d.mode = btn.dataset.view3dMode;
        update3dViewButtons();
      });
      $("view3dYaw").oninput = e => { state.view3d.yaw = degToRad(Number(e.target.value)); setCustom3dView(); };
      $("view3dPitch").oninput = e => { state.view3d.pitch = degToRad(Number(e.target.value)); setCustom3dView(); };
      $("view3dDistance").oninput = e => { state.view3d.distance = Number(e.target.value); setCustom3dView(); };
      $("view3dZScale").oninput = e => { state.view3d.zScale = Number(e.target.value); setCustom3dView(); };
    }

    function renderSummary() {
      const s = state.index.summary;
      $("summary").innerHTML = [
        ["Frames", s.frames], ["TP", s.tp], ["FP", s.fp], ["FN", s.fn], ["LOC", s.loc], ["Review", `${state.index.review_summary.selected_cases}/${state.index.all_case_count}`],
      ].map(([k, v]) => `<div class="metric"><span>${k}</span><b>${v}</b></div>`).join("");
    }

    function populateFrames() {
      $("frameSelect").innerHTML = state.index.frames.map((f, i) => `<option value="${i}">#${f.frame_index} ${f.frame_id.slice(0,8)} · FP ${f.summary.fp} FN ${f.summary.fn} LOC ${f.summary.loc}</option>`).join("");
    }

    function renderCaseList() {
      const cases = state.index.cases.filter(c => state.filter === "ALL" || c.type === state.filter);
      $("caseList").innerHTML = cases.map(c => `<button class="case" data-case="${c.eval_id}" data-type="${c.type}">
        <strong><span>${c.type} · ${c.class_name}</span><span>${Number(c.case_score).toFixed(1)}</span></strong>
        <small>${c.reason || "matched"} · frame ${c.frame_index} · ${c.frame_id.slice(0,8)}</small>
      </button>`).join("") || `<div class="empty">No cases for ${state.filter}</div>`;
      document.querySelectorAll(".case").forEach(btn => btn.onclick = async () => {
        const item = state.index.cases.find(c => c.eval_id === btn.dataset.case);
        const framePos = state.index.frames.findIndex(f => f.frame_index === item.frame_index);
        state.selectedCase = item;
        await loadFrame(framePos, item.eval_id);
      });
    }

    async function loadFrame(position, caseId = null) {
      const frames = state.index.frames;
      state.framePos = (position + frames.length) % frames.length;
      const meta = frames[state.framePos];
      $("frameSelect").value = state.framePos;
      state.frame = await (await fetch(meta.frame_json)).json();
      state.image = await loadImage(meta.image);
      if (caseId) state.selectedCase = state.index.cases.find(c => c.eval_id === caseId);
      if (caseId) focus3dOnSelected(false);
      renderAll();
    }

    function loadImage(src) {
      return new Promise(resolve => { const img = new Image(); img.onload = () => resolve(img); img.src = src; });
    }

    function renderAll() { renderInfo(); renderImage(); render3d(); renderBev(); }

    function renderInfo() {
      const f = state.frame, e = f.eval.summary;
      const rows = [
        ["Frame", `#${f.index} ${f.frame_id}`], ["Sequence", f.sequence_id], ["Timestamp", Number(f.timestamp).toFixed(3)],
        ["GT / Pred", `${e.gt} / ${e.pred}`], ["TP / FP / FN / LOC", `${e.tp} / ${e.fp} / ${e.fn} / ${e.loc}`],
        ["Radar points", `${f.sensors.radar_front.display_count} / ${f.sensors.radar_front.source_count}`],
        ["ATX points", `${f.sensors.lidar_front_2.display_count} / ${f.sensors.lidar_front_2.source_count}`],
        ["EM4 points", `${f.sensors.lidar_front.display_count} / ${f.sensors.lidar_front.source_count}`],
        ["Radar color", radarColorLabel()],
      ];
      const radarScale = radarColorStatsLabel();
      if (radarScale) rows.push(["Radar scale", radarScale]);
      $("frameInfo").innerHTML = rows.map(([k, v]) => `<div class="info-row"><span>${k}</span><span>${v}</span></div>`).join("");
      $("caseDetail").textContent = state.selectedCase ? JSON.stringify(state.selectedCase, null, 2) : "Select a case";
    }

    function fitCanvas(canvas) {
      const rect = canvas.getBoundingClientRect();
      const dpr = window.devicePixelRatio || 1;
      canvas.width = Math.max(1, Math.round(rect.width * dpr));
      canvas.height = Math.max(1, Math.round(rect.height * dpr));
      const ctx = canvas.getContext("2d");
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      return { ctx, width: rect.width, height: rect.height };
    }

    function renderImage() {
      const { ctx, width, height } = fitCanvas($("imageCanvas"));
      ctx.fillStyle = "#101216"; ctx.fillRect(0, 0, width, height);
      if (!state.image) return;
      const scale = Math.min(width / state.image.naturalWidth, height / state.image.naturalHeight);
      const w = state.image.naturalWidth * scale, h = state.image.naturalHeight * scale;
      const x0 = (width - w) / 2, y0 = (height - h) / 2;
      ctx.drawImage(state.image, x0, y0, w, h);
      ctx.save(); ctx.translate(x0, y0); ctx.scale(scale, scale);
      if (state.layers.projection) drawRadarProjection(ctx);
      if (state.layers.gt) drawImageGt(ctx);
      if (state.layers.pred) drawImagePred(ctx);
      ctx.restore();
    }

    function drawRadarProjection(ctx) {
      const proj = state.frame.radar_projection || {};
      const features = proj.features || {};
      const stats = state.frame.sensors.radar_front?.feature_stats || {};
      (proj.u || []).forEach((u, i) => {
        const d = proj.depth[i];
        if (d < 180) {
          ctx.fillStyle = radarPointColor(features, stats, i);
          ctx.fillRect(u - 1, proj.v[i] - 1, 2, 2);
        }
      });
    }
    function drawImageGt(ctx) {
      state.frame.annotations.forEach((gt, i) => {
        const style = styleForGt(i);
        drawSegments(ctx, gt.image_segments, style.color, style.width);
      });
    }
    function drawImagePred(ctx) {
      state.frame.overlays.forEach(pred => drawSegments(ctx, pred.image_segments, colorForPred(pred), selected(pred.eval_id) ? 4 : 2));
    }
    function drawSegments(ctx, segments, color, lineWidth) {
      ctx.strokeStyle = color; ctx.lineWidth = lineWidth;
      (segments || []).forEach(seg => { ctx.beginPath(); ctx.moveTo(seg[0][0], seg[0][1]); ctx.lineTo(seg[1][0], seg[1][1]); ctx.stroke(); });
    }

    function reset3dView() {
      set3dView("iso");
    }

    function set3dView(name) {
      const preset = view3dPresets[name] || view3dPresets.iso;
      Object.assign(state.view3d, preset, { preset: name });
      update3dViewButtons();
      render3d();
    }

    function update3dViewButtons() {
      document.querySelectorAll("[data-view3d]").forEach(btn => btn.classList.toggle("active", btn.dataset.view3d === state.view3d.preset));
      document.querySelectorAll("[data-view3d-mode]").forEach(btn => btn.classList.toggle("active", btn.dataset.view3dMode === state.view3d.mode));
      $("view3dYaw").value = String(Math.round(radToDeg(state.view3d.yaw)));
      $("view3dPitch").value = String(Math.round(radToDeg(state.view3d.pitch)));
      $("view3dDistance").value = String(Math.round(state.view3d.distance));
      $("view3dZScale").value = String(state.view3d.zScale);
      $("view3dYawValue").textContent = `${Math.round(radToDeg(state.view3d.yaw))}°`;
      $("view3dPitchValue").textContent = `${Math.round(radToDeg(state.view3d.pitch))}°`;
      $("view3dDistanceValue").textContent = `${Math.round(state.view3d.distance)}m`;
      $("view3dZScaleValue").textContent = `${Number(state.view3d.zScale).toFixed(2)}x`;
    }

    function setCustom3dView() {
      state.view3d.preset = "custom";
      update3dViewButtons();
      render3d();
    }

    function fit3dToFrame() {
      const bounds = frame3dBounds();
      if (!bounds) return;
      const cx = (bounds.minX + bounds.maxX) / 2, cy = (bounds.minY + bounds.maxY) / 2, cz = (bounds.minZ + bounds.maxZ) / 2;
      const spanX = bounds.maxX - bounds.minX, spanY = bounds.maxY - bounds.minY, spanZ = bounds.maxZ - bounds.minZ;
      Object.assign(state.view3d, {
        preset: "fit",
        yaw: -0.72,
        pitch: 0.66,
        distance: clamp(Math.max(spanX, spanY, spanZ) * 1.55, 70, 520),
        targetX: cx,
        targetY: cy,
        targetZ: cz,
        zScale: 1.8,
      });
      update3dViewButtons();
      render3d();
    }

    function focus3dOnSelected(doRender = true) {
      const object = selected3dObject();
      if (!object) return;
      const center = boxCenter3d(object);
      Object.assign(state.view3d, {
        preset: "custom",
        targetX: center.x,
        targetY: center.y,
        targetZ: center.z,
        distance: clamp(Math.max(object.box?.[3] || 6, object.box?.[4] || 3, object.box?.[5] || 2) * 18, 40, 160),
      });
      update3dViewButtons();
      if (doRender) render3d();
    }

    function selected3dObject() {
      if (!state.selectedCase || !state.frame) return null;
      if (state.selectedCase.pred_index !== undefined && state.selectedCase.pred_index !== null) {
        const pred = state.frame.overlays.find(item => item.pred_index === state.selectedCase.pred_index || item.eval_id === state.selectedCase.eval_id);
        if (pred) return pred;
      }
      if (state.selectedCase.gt_index !== undefined && state.selectedCase.gt_index !== null) {
        const gt = state.frame.annotations[state.selectedCase.gt_index];
        if (gt) return gt;
      }
      return state.frame.overlays.find(item => item.eval_id === state.selectedCase.eval_id) || null;
    }

    function frame3dBounds() {
      if (!state.frame) return null;
      const bounds = { minX: Infinity, minY: Infinity, minZ: Infinity, maxX: -Infinity, maxY: -Infinity, maxZ: -Infinity };
      const add = (x, y, z) => {
        if (!Number.isFinite(x) || !Number.isFinite(y) || !Number.isFinite(z)) return;
        bounds.minX = Math.min(bounds.minX, x); bounds.minY = Math.min(bounds.minY, y); bounds.minZ = Math.min(bounds.minZ, z);
        bounds.maxX = Math.max(bounds.maxX, x); bounds.maxY = Math.max(bounds.maxY, y); bounds.maxZ = Math.max(bounds.maxZ, z);
      };
      ["radar_front", "lidar_front_2", "lidar_front"].forEach(key => {
        const sensor = state.frame.sensors[key];
        if (!sensor || !sensor.points) return;
        const pts = sensor.points, xs = pts.x || [], ys = pts.y || [], zs = pts.z || [];
        const stride = Math.max(1, Math.ceil(xs.length / 8000));
        for (let i = 0; i < xs.length; i += stride) {
          if (xs[i] < -10 || xs[i] > state.view.range || ys[i] < -75 || ys[i] > 75) continue;
          add(xs[i], ys[i], zs[i] || 0);
        }
      });
      state.frame.annotations.forEach(item => boxCorners3d(item).forEach(([x, y, z]) => add(x, y, z)));
      state.frame.overlays.forEach(item => boxCorners3d(item).forEach(([x, y, z]) => add(x, y, z)));
      if (!Number.isFinite(bounds.minX)) return null;
      return bounds;
    }

    function boxCenter3d(item) {
      if (Array.isArray(item.box) && item.box.length >= 3) return { x: item.box[0], y: item.box[1], z: item.box[2] };
      const corners = boxCorners3d(item);
      return corners.reduce((acc, [x, y, z]) => ({ x: acc.x + x / corners.length, y: acc.y + y / corners.length, z: acc.z + z / corners.length }), { x: 0, y: 0, z: 0 });
    }

    function render3d() {
      const { ctx, width, height } = fitCanvas($("view3dCanvas"));
      ctx.fillStyle = "#101216"; ctx.fillRect(0, 0, width, height);
      if (!state.frame) return;
      const camera = camera3d(width, height);
      draw3dGrid(ctx, camera);
      if (state.layers.atx) draw3dPoints(ctx, camera, state.frame.sensors.lidar_front_2.points, colors.atx, 10000);
      if (state.layers.em4) draw3dPoints(ctx, camera, state.frame.sensors.lidar_front.points, colors.em4, 10000);
      if (state.layers.radar) draw3dRadarPoints(ctx, camera, 14000);
      if (state.layers.gt) draw3dGtBoxes(ctx, camera);
      if (state.layers.pred) draw3dPredBoxes(ctx, camera);
      draw3dEgo(ctx, camera);
    }

    function camera3d(width, height) {
      const v = state.view3d;
      const target = { x: v.targetX, y: v.targetY, z: scaledZ(v.targetZ) };
      const cp = Math.cos(v.pitch), sp = Math.sin(v.pitch), cy = Math.cos(v.yaw), sy = Math.sin(v.yaw);
      const position = { x: target.x - v.distance * cp * cy, y: target.y - v.distance * cp * sy, z: target.z + v.distance * sp };
      const forward = normalize({ x: target.x - position.x, y: target.y - position.y, z: target.z - position.z });
      let right = normalize(cross(forward, { x: 0, y: 0, z: 1 }));
      if (!Number.isFinite(right.x)) right = { x: 0, y: -1, z: 0 };
      const up = normalize(cross(right, forward));
      return { position, forward, right, up, focal: Math.min(width, height) * 0.9, width, height };
    }

    function pan3d(dx, dy) {
      const camera = camera3d(1, 1);
      const scale = state.view3d.distance / 560;
      state.view3d.targetX += (-dx * camera.right.x + dy * camera.up.x) * scale;
      state.view3d.targetY += (-dx * camera.right.y + dy * camera.up.y) * scale;
      state.view3d.targetZ += (-dx * camera.right.z + dy * camera.up.z) * scale / Math.max(state.view3d.zScale, 0.01);
    }

    function project3d(camera, point) {
      const rel = { x: point.x - camera.position.x, y: point.y - camera.position.y, z: scaledZ(point.z) - camera.position.z };
      const depth = dot(rel, camera.forward);
      if (depth <= 0.2) return null;
      const sx = dot(rel, camera.right), sy = dot(rel, camera.up);
      return { x: camera.width / 2 + sx * camera.focal / depth, y: camera.height / 2 - sy * camera.focal / depth, depth };
    }

    function draw3dGrid(ctx, camera) {
      ctx.lineWidth = 1;
      ctx.strokeStyle = "#29313d";
      for (let x = 0; x <= state.view.range; x += 20) line3d(ctx, camera, { x, y: -70, z: 0 }, { x, y: 70, z: 0 });
      for (let y = -60; y <= 60; y += 20) line3d(ctx, camera, { x: -10, y, z: 0 }, { x: state.view.range, y, z: 0 });
      ctx.strokeStyle = "#687386";
      line3d(ctx, camera, { x: -10, y: 0, z: 0 }, { x: state.view.range, y: 0, z: 0 });
      ctx.strokeStyle = "#d9dee7";
      line3d(ctx, camera, { x: 0, y: 0, z: 0 }, { x: 20, y: 0, z: 0 });
      ctx.strokeStyle = colors.fp;
      line3d(ctx, camera, { x: 0, y: 0, z: 0 }, { x: 0, y: 12, z: 0 });
      ctx.strokeStyle = colors.tp;
      line3d(ctx, camera, { x: 0, y: 0, z: 0 }, { x: 0, y: 0, z: 8 });
    }

    function draw3dPoints(ctx, camera, points, color, maxPoints) {
      const xs = points.x || [], ys = points.y || [], zs = points.z || [];
      const stride = Math.max(1, Math.ceil(xs.length / maxPoints));
      ctx.fillStyle = color;
      for (let i = 0; i < xs.length; i += stride) {
        const x = xs[i], y = ys[i], z = zs[i] || 0;
        if (x < -10 || x > state.view.range || y < -75 || y > 75 || z < -6 || z > 18) continue;
        const p = project3d(camera, { x, y, z });
        if (!p || p.x < -4 || p.x > camera.width + 4 || p.y < -4 || p.y > camera.height + 4) continue;
        const size = clamp(220 / p.depth, 1, 2.8);
        ctx.fillRect(p.x, p.y, size, size);
      }
    }

    function draw3dRadarPoints(ctx, camera, maxPoints) {
      const sensor = state.frame.sensors.radar_front;
      const points = sensor.points || {}, features = sensor.features || {}, stats = sensor.feature_stats || {};
      const xs = points.x || [], ys = points.y || [], zs = points.z || [];
      const stride = Math.max(1, Math.ceil(xs.length / maxPoints));
      for (let i = 0; i < xs.length; i += stride) {
        const x = xs[i], y = ys[i], z = zs[i] || 0;
        if (x < -10 || x > state.view.range || y < -75 || y > 75 || z < -6 || z > 18) continue;
        const p = project3d(camera, { x, y, z });
        if (!p || p.x < -4 || p.x > camera.width + 4 || p.y < -4 || p.y > camera.height + 4) continue;
        const size = clamp(220 / p.depth, 1, 3.0);
        ctx.fillStyle = radarPointColor(features, stats, i);
        ctx.fillRect(p.x, p.y, size, size);
      }
    }

    function draw3dGtBoxes(ctx, camera) {
      state.frame.annotations.forEach((gt, i) => {
        const style = styleForGt(i);
        draw3dBox(ctx, camera, gt, style.color, selected(style.evalId) ? 4 : style.width);
      });
    }

    function draw3dPredBoxes(ctx, camera) {
      state.frame.overlays.forEach(pred => draw3dBox(ctx, camera, pred, colorForPred(pred), selected(pred.eval_id) ? 4 : 2));
    }

    function draw3dBox(ctx, camera, item, color, width) {
      const pts = boxCorners3d(item).map(([x, y, z]) => project3d(camera, { x, y, z }));
      const edges = [[0,1], [1,2], [2,3], [3,0], [4,5], [5,6], [6,7], [7,4], [0,4], [1,5], [2,6], [3,7]];
      ctx.strokeStyle = color; ctx.lineWidth = width;
      edges.forEach(([a, b]) => {
        if (!pts[a] || !pts[b]) return;
        ctx.beginPath(); ctx.moveTo(pts[a].x, pts[a].y); ctx.lineTo(pts[b].x, pts[b].y); ctx.stroke();
      });
    }

    function boxCorners3d(item) {
      if (Array.isArray(item.corners) && item.corners.length >= 8) return item.corners.slice(0, 8);
      const [x, y, z, l, w, h, yaw] = item.box, c = Math.cos(yaw), s = Math.sin(yaw);
      const zs = [z - h / 2, z + h / 2];
      const base = [[l/2,w/2], [l/2,-w/2], [-l/2,-w/2], [-l/2,w/2]].map(([lx, ly]) => [x + lx*c - ly*s, y + lx*s + ly*c]);
      return zs.flatMap(zz => base.map(([cx, cy]) => [cx, cy, zz]));
    }

    function draw3dEgo(ctx, camera) {
      const body = [{ x: 3, y: 0, z: 0.4 }, { x: -2.5, y: 2.1, z: 0.1 }, { x: -2.5, y: -2.1, z: 0.1 }].map(p => project3d(camera, p));
      if (body.some(p => !p)) return;
      ctx.fillStyle = "#eef2f7";
      ctx.beginPath(); ctx.moveTo(body[0].x, body[0].y); ctx.lineTo(body[1].x, body[1].y); ctx.lineTo(body[2].x, body[2].y); ctx.closePath(); ctx.fill();
    }

    function line3d(ctx, camera, a, b) {
      const pa = project3d(camera, a), pb = project3d(camera, b);
      if (!pa || !pb) return;
      ctx.beginPath(); ctx.moveTo(pa.x, pa.y); ctx.lineTo(pb.x, pb.y); ctx.stroke();
    }

    function renderBev() {
      const { ctx, width, height } = fitCanvas($("bevCanvas"));
      ctx.fillStyle = "#101216"; ctx.fillRect(0, 0, width, height);
      const geom = bevGeom(width, height);
      drawGrid(ctx, geom);
      if (state.layers.atx) drawPoints(ctx, geom, state.frame.sensors.lidar_front_2.points, colors.atx, 13000);
      if (state.layers.em4) drawPoints(ctx, geom, state.frame.sensors.lidar_front.points, colors.em4, 13000);
      if (state.layers.radar) drawRadarPoints(ctx, geom, 16000);
      if (state.layers.gt) drawGtBoxes(ctx, geom);
      if (state.layers.pred) drawPredBoxes(ctx, geom);
      drawEgo(ctx, geom);
    }

    function bevGeom(width, height) {
      const range = state.view.range;
      const xMin = -10, xMax = range, yMin = -70, yMax = 70;
      const margin = { l: 46, r: 16, t: 18, b: 28 };
      const plotW = width - margin.l - margin.r, plotH = height - margin.t - margin.b;
      const baseScale = Math.min(plotW / (yMax - yMin), plotH / (xMax - xMin));
      const scale = baseScale * state.view.zoom;
      const originX = margin.l + (plotW - (yMax - yMin) * baseScale) / 2 + state.view.panX;
      const originY = margin.t + (plotH - (xMax - xMin) * baseScale) / 2 + state.view.panY;
      return { xMin, xMax, yMin, yMax, scale, originX, originY };
    }
    function worldToCanvas(geom, x, y) { return [geom.originX + (geom.yMax - y) * geom.scale, geom.originY + (geom.xMax - x) * geom.scale]; }
    function drawGrid(ctx, g) {
      ctx.strokeStyle = "#303642"; ctx.lineWidth = 1; ctx.fillStyle = "#808899"; ctx.font = "11px sans-serif";
      for (let x = 0; x <= g.xMax; x += 20) { const a = worldToCanvas(g, x, g.yMin), b = worldToCanvas(g, x, g.yMax); line(ctx, a, b); ctx.fillText(`${x}m`, b[0] + 2, b[1] + 10); }
      for (let y = -60; y <= 60; y += 20) { line(ctx, worldToCanvas(g, g.xMin, y), worldToCanvas(g, g.xMax, y)); }
      ctx.strokeStyle = "#596273"; line(ctx, worldToCanvas(g, g.xMin, 0), worldToCanvas(g, g.xMax, 0));
    }
    function drawPoints(ctx, g, points, color, maxPoints) {
      const xs = points.x || [], ys = points.y || [];
      const stride = Math.max(1, Math.ceil(xs.length / maxPoints));
      ctx.fillStyle = color;
      for (let i = 0; i < xs.length; i += stride) {
        const x = xs[i], y = ys[i];
        if (x < g.xMin || x > g.xMax || y < g.yMin || y > g.yMax) continue;
        const p = worldToCanvas(g, x, y); ctx.fillRect(p[0], p[1], 1.5, 1.5);
      }
    }
    function drawRadarPoints(ctx, g, maxPoints) {
      const sensor = state.frame.sensors.radar_front;
      const points = sensor.points || {}, features = sensor.features || {}, stats = sensor.feature_stats || {};
      const xs = points.x || [], ys = points.y || [];
      const stride = Math.max(1, Math.ceil(xs.length / maxPoints));
      for (let i = 0; i < xs.length; i += stride) {
        const x = xs[i], y = ys[i];
        if (x < g.xMin || x > g.xMax || y < g.yMin || y > g.yMax) continue;
        const p = worldToCanvas(g, x, y);
        ctx.fillStyle = radarPointColor(features, stats, i);
        ctx.fillRect(p[0], p[1], 1.6, 1.6);
      }
    }
    function drawGtBoxes(ctx, g) {
      state.frame.annotations.forEach((gt, i) => {
        const style = styleForGt(i);
        drawBox(ctx, g, gt.box, style.color, style.width, `${style.label} ${gt.name}`);
      });
    }
    function drawPredBoxes(ctx, g) {
      state.frame.overlays.forEach(pred => drawBox(ctx, g, pred.box, colorForPred(pred), selected(pred.eval_id) ? 4 : 2, `${pred.match_status || "pred"} ${pred.name} ${Number(pred.score).toFixed(2)}`));
    }
    function drawBox(ctx, g, box, color, width, label) {
      const pts = corners(box).map(p => worldToCanvas(g, p[0], p[1]));
      ctx.strokeStyle = color; ctx.lineWidth = width; ctx.beginPath(); ctx.moveTo(pts[0][0], pts[0][1]); pts.slice(1).forEach(p => ctx.lineTo(p[0], p[1])); ctx.closePath(); ctx.stroke();
      ctx.fillStyle = color; ctx.font = "12px sans-serif"; ctx.fillText(label, pts[0][0] + 4, pts[0][1] - 4);
    }
    function corners(box) {
      const [x,y,,l,w,,yaw] = box, c = Math.cos(yaw), s = Math.sin(yaw);
      return [[l/2,w/2], [l/2,-w/2], [-l/2,-w/2], [-l/2,w/2]].map(([lx, ly]) => [x + lx*c - ly*s, y + lx*s + ly*c]);
    }
    function drawEgo(ctx, g) {
      const a = worldToCanvas(g, 3, 0), b = worldToCanvas(g, -2, 2.2), c = worldToCanvas(g, -2, -2.2);
      ctx.fillStyle = "#eee"; ctx.beginPath(); ctx.moveTo(a[0], a[1]); ctx.lineTo(b[0], b[1]); ctx.lineTo(c[0], c[1]); ctx.closePath(); ctx.fill();
    }
    function line(ctx, a, b) { ctx.beginPath(); ctx.moveTo(a[0], a[1]); ctx.lineTo(b[0], b[1]); ctx.stroke(); }
    function evalGt(index) { return (state.frame.eval.gt || []).find(item => item.gt_index === index); }
    function styleForGt(index) {
      const item = evalGt(index);
      if (!item || item.in_region !== true) return { color: colors.ignore, width: 1, label: "IGNORE GT" };
      if (item.match_status === "fn") return { color: colors.fn, width: 3, label: "FN GT", evalId: item.eval_id };
      return { color: colors.gt, width: 2, label: "GT", evalId: item.eval_id };
    }
    function radarColorField() {
      return { rcs: "RCS", doppler: "doppler", speed: "AbsV" }[state.radarColorMode] || "";
    }
    function radarColorLabel() {
      return { fixed: "Fixed", rcs: "RCS", doppler: "Doppler", speed: "AbsV" }[state.radarColorMode] || "Fixed";
    }
    function radarColorStatsLabel() {
      const field = radarColorField();
      if (!field) return "";
      const stat = state.frame?.sensors?.radar_front?.feature_stats?.[field];
      if (!stat) return "";
      const format = value => Number.isFinite(Number(value)) ? Number(value).toFixed(2) : "n/a";
      return `min ${format(stat.min)} / med ${format(stat.median)} / max ${format(stat.max)}`;
    }
    function radarPointColor(features, stats, index) {
      const field = radarColorField();
      if (!field || !features || !features[field]) return colors.radar;
      const value = Number(features[field][index]);
      if (!Number.isFinite(value)) return colors.radar;
      const stat = stats?.[field] || {};
      if (state.radarColorMode === "doppler") {
        const limit = Math.max(Math.abs(Number(stat.min) || 0), Math.abs(Number(stat.max) || 0), 0.5);
        const normalized = clamp((value / limit + 1) / 2, 0, 1);
        return divergingColor(normalized);
      }
      const min = Number.isFinite(Number(stat.min)) ? Number(stat.min) : value - 1;
      const max = Number.isFinite(Number(stat.max)) ? Number(stat.max) : value + 1;
      const normalized = clamp((value - min) / Math.max(max - min, 1e-6), 0, 1);
      return state.radarColorMode === "rcs" ? sequentialColor(normalized) : speedColor(normalized);
    }
    function sequentialColor(t) {
      return interpolateRgb([37, 99, 235], [250, 204, 21], t);
    }
    function speedColor(t) {
      return interpolateRgb([34, 211, 238], [244, 63, 94], t);
    }
    function divergingColor(t) {
      return t < 0.5
        ? interpolateRgb([37, 99, 235], [238, 242, 247], t * 2)
        : interpolateRgb([238, 242, 247], [244, 63, 94], (t - 0.5) * 2);
    }
    function interpolateRgb(a, b, t) {
      const c = a.map((value, i) => Math.round(value + (b[i] - value) * clamp(t, 0, 1)));
      return `rgb(${c[0]},${c[1]},${c[2]})`;
    }
    function clamp(value, min, max) { return Math.max(min, Math.min(max, value)); }
    function degToRad(value) { return value * Math.PI / 180; }
    function radToDeg(value) { return value * 180 / Math.PI; }
    function scaledZ(value) { return value * state.view3d.zScale; }
    function dot(a, b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
    function cross(a, b) { return { x: a.y * b.z - a.z * b.y, y: a.z * b.x - a.x * b.z, z: a.x * b.y - a.y * b.x }; }
    function normalize(v) { const n = Math.hypot(v.x, v.y, v.z); return { x: v.x / n, y: v.y / n, z: v.z / n }; }
    function colorForPred(pred) { return pred.match_status === "tp" ? colors.tp : colors.fp; }
    function selected(evalId) { return state.selectedCase && evalId && state.selectedCase.eval_id === evalId; }
    init().catch(err => { document.body.innerHTML = `<pre>${err.stack || err}</pre>`; });
  </script>
</body>
</html>
"""


def main() -> None:
    args = parse_args()
    index = build_bundle(args)
    print(json.dumps({"summary": index["summary"], "review": index["review_summary"]}, ensure_ascii=False))
    print(f"wrote_html={args.output_dir / 'index.html'}")
    print(f"wrote_index={args.output_dir / 'assets' / 'index.json'}")
    print(f"wrote_diff={args.output_dir / 'eval_diff.json'}")


if __name__ == "__main__":
    main()
