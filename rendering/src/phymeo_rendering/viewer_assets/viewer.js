"use strict";

(() => {
  const $ = (id) => document.getElementById(id);
  const state = {
    scene: null, wanted: null, shown: null, frontier: 0, serial: 0, busy: false,
    playing: false, timer: 0, scrubTimer: 0, failed: false, cameras: [], cameraKeys: [],
  };
  const ui = {
    play: $("playback"), previous: $("prev-frame"), next: $("next-frame"), seek: $("seek"),
    number: $("frame-number"), status: $("play-state"), picture: $("picture"), message: $("stage-message"),
  };
  const channelIds = new Set(["rgb", "depth", "id"]);
  const chineseChannels = {rgb: "RGB", depth: "深度", id: "实例 ID"};
  const defaultNotes = {
    rgb: "相机彩色画面",
    depth: "前向深度，近处为白色，远处为深灰，无命中为黑色。",
    id: "实例使用固定伪彩，同一场景中相同 ID 跨机位同色，背景为黑色。",
  };
  const iconNamespace = "http://www.w3.org/2000/svg";

  function icon(name) {
    const svg = document.createElementNS(iconNamespace, "svg");
    svg.setAttribute("class", "icon");
    svg.setAttribute("aria-hidden", "true");
    const use = document.createElementNS(iconNamespace, "use");
    use.setAttribute("href", "#" + name);
    svg.append(use);
    return svg;
  }

  function keyedList(items, name) {
    if (!Array.isArray(items) || !items.length) throw new Error("场景没有提供" + name + "。");
    const seen = new Set();
    for (const item of items) {
      if (!item || typeof item.id !== "string" || !item.id || seen.has(item.id) ||
          typeof item.label !== "string" || !item.label.trim()) throw new Error(name + "信息不完整或 ID 重复。");
      seen.add(item.id);
    }
    return items;
  }

  function validateScene(data) {
    if (!data || !["replay", "live"].includes(data.mode) || typeof data.title !== "string" ||
        !data.title.trim() || !Number.isFinite(data.fps) || data.fps <= 0) {
      throw new Error("服务返回的场景信息不完整。");
    }
    if (!(data.frame_count === null && data.mode === "live") &&
        (!Number.isSafeInteger(data.frame_count) || data.frame_count < 1)) {
      throw new Error("服务返回的帧数无效。");
    }
    if (data.current_frame != null && (!Number.isSafeInteger(data.current_frame) ||
        data.current_frame < 0 || (data.frame_count !== null && data.current_frame >= data.frame_count))) {
      throw new Error("服务返回的当前帧无效。");
    }
    keyedList(data.cameras, "机位");
    if (!keyedList(data.channels, "通道").some((item) => channelIds.has(item.id))) {
      throw new Error("场景没有可显示的图像通道。");
    }
    return data;
  }

  const isLive = () => state.scene?.mode === "live" || state.scene?.seekable === false;
  const lastFrame = () => state.scene?.frame_count == null ? Infinity : state.scene.frame_count - 1;
  const isOverview = (c) => /overview|global|总览|全景/i.test(c.id + " " + c.label);
  const isWrist = (c) => /wrist|腕部|腕相机/i.test(c.id + " " + c.label);
  const cameraById = (id) => state.cameras.find((c) => c.id === id);
  const sameView = (a, b) => a && b && a.camera === b.camera && a.channel === b.channel && a.frame === b.frame;
  const cameraType = (c) => isOverview(c) ? "overview" : isWrist(c) ? "wrist" : "camera";

  function buildChoices() {
    const cameras = state.scene.cameras;
    const overview = cameras.find(isOverview);
    state.cameras = overview ? [overview, ...cameras.filter((c) => c !== overview)] : [...cameras];
    state.cameraKeys = [...state.cameras.filter((c) => !isWrist(c)), ...state.cameras.filter(isWrist)].slice(0, 9);
    $("camera-count").textContent = state.cameras.length;
    const cameraButtons = state.cameras.map((camera) => {
      const button = document.createElement("button");
      button.type = "button";
      button.className = "camera-option";
      button.dataset.camera = camera.id;
      const label = document.createElement("span");
      label.textContent = camera.label;
      button.append(icon(cameraType(camera)), label);
      const shortcut = state.cameraKeys.indexOf(camera);
      if (shortcut >= 0) {
        const key = document.createElement("kbd");
        key.textContent = String(shortcut);
        key.setAttribute("aria-hidden", "true");
        button.setAttribute("aria-keyshortcuts", String(shortcut));
        button.title = camera.label + " (" + shortcut + ")";
        button.append(key);
      }
      button.addEventListener("click", () => chooseCamera(camera.id));
      return button;
    });
    $("camera-list").replaceChildren(...cameraButtons);
    const channels = state.scene.channels.filter((c) => channelIds.has(c.id));
    $("channels").replaceChildren(...channels.map((ch) => {
      const button = document.createElement("button");
      button.type = "button";
      button.className = "channel";
      button.dataset.channel = ch.id;
      button.textContent = chineseChannels[ch.id];
      button.title = ch.description || ch.label;
      button.addEventListener("click", () => choose({channel: ch.id}));
      return button;
    }));
    const count = state.scene.frame_count;
    const ticks = count === null ? [] : [...new Set([0, Math.round((count-1)/4), Math.round((count-1)/2),
      Math.round(3*(count-1)/4), count-1])];
    $("ruler").replaceChildren(...ticks.map((value) => {
      const span = document.createElement("span");
      span.textContent = String(value);
      return span;
    }));
    $("track").hidden = isLive();
    ui.number.readOnly = isLive();
    ui.number.title = isLive() ? "实时状态按步前进" : "输入从 0 开始的帧号，然后按 Enter";
    $("frame-limit").textContent = count === null ? "" : "/ " + (count-1);
    if (count !== null) ui.number.max = String(count-1); else ui.number.removeAttribute("max");
    ui.seek.max = String(count === null ? 0 : count-1);
  }

  function transportState() {
    const want = state.wanted;
    if (!want) return;
    const ready = !!state.shown && !state.failed;
    const atEnd = want.frame >= lastFrame();
    const liveAdvance = isLive() && (state.busy || want.frame > state.frontier);
    ui.previous.disabled = !ready || isLive() || want.frame <= 0;
    ui.next.disabled = !ready || atEnd || liveAdvance;
    ui.play.disabled = !ready || (atEnd && (isLive() || lastFrame() === 0) && !state.playing);
    ui.seek.disabled = isLive() || lastFrame() === 0;
    ui.seek.value = want.frame;
    ui.seek.setAttribute("aria-valuetext", "第 " + want.frame + " 帧");
    ui.number.disabled = !ready;
    if (document.activeElement !== ui.number) ui.number.value = String(want.frame);
    const playText = state.playing ? "暂停" : "播放";
    ui.play.querySelector("span").textContent = playText;
    ui.play.querySelector("use").setAttribute("href", state.playing ? "#pause" : "#play");
    ui.play.title = playText + " (空格)";
    ui.play.setAttribute("aria-label", playText);
    ui.status.hidden = false;
    ui.status.classList.toggle("playing", state.playing);
    ui.status.classList.toggle("error", state.failed);
    ui.status.textContent = state.failed ? "读取失败" : state.playing ? "播放中" :
      !sameView(want, state.shown) ? "正在取帧" : "已暂停";
    for (const button of $("camera-list").children) {
      button.setAttribute("aria-pressed", String(button.dataset.camera === want.camera));
    }
    for (const button of $("channels").children) {
      button.setAttribute("aria-pressed", String(button.dataset.channel === want.channel));
    }
    const camera = cameraById(want.camera);
    $("camera-title").textContent = camera.label;
    $("mobile-camera").textContent = camera.label;
  }

  function stage(title, detail = "", failed = false) {
    ui.message.hidden = false;
    ui.message.classList.toggle("failed", failed);
    $("stage-title").textContent = title;
    $("stage-detail").textContent = detail;
    $("retry").hidden = !failed;
  }

  function progress() {
    const view = state.shown;
    const want = state.wanted;
    if (!view || view.camera !== want.camera || view.channel !== want.channel) {
      ui.picture.hidden = true;
      $("captured-frame").textContent = "等待画面";
      $("sim-time").textContent = "—";
      $("render-ms").textContent = "—";
      $("channel-note").textContent = "";
      stage("正在读取画面", cameraById(want.camera).label + "，帧 " + want.frame);
    } else {
      // Hold the last complete image during a seek, with its own frame and timing.
      ui.picture.hidden = false;
      ui.message.hidden = true;
      $("captured-frame").textContent = sameView(want, view) ? "当前帧 " + view.frame :
        "显示帧 " + view.frame + "，正在读取 " + want.frame;
      $("sim-time").textContent = view.time.toFixed(3) + " s";
      $("render-ms").textContent = view.ms.toFixed(1) + " ms";
      const spec = state.scene.channels.find((ch) => ch.id === view.channel);
      const camera = cameraById(view.camera);
      const range = camera.depth_range_m;
      $("channel-note").textContent = view.channel === "depth" && Array.isArray(range) &&
        range.length === 2 && range.every(Number.isFinite) ?
        "前向深度 " + range[0] + "–" + range[1] + " m，近白远灰，未命中为黑" :
        spec.description || defaultNotes[view.channel];
    }
    transportState();
  }

  function stop() {
    state.playing = false;
    clearTimeout(state.timer);
  }

  function fail(error) {
    state.failed = true;
    stop();
    ui.picture.hidden = true;
    $("captured-frame").textContent = "画面未发布";
    $("sim-time").textContent = "—";
    $("render-ms").textContent = "—";
    $("channel-note").textContent = "";
    stage("无法读取画面", error.message || "请检查渲染服务后重试。", true);
    transportState();
  }

  async function responseError(response) {
    if (response.status === 429) return new Error("渲染器正在处理另一项请求，请稍后重试。");
    let detail = "";
    try {
      const data = await response.json();
      const reason = data.error || data.detail || data.message;
      if (typeof reason === "string") detail = reason.slice(0, 220);
    } catch { /* Empty or non-JSON error reply. */ }
    return new Error("服务返回 HTTP " + response.status + (detail ? "，" + detail : "，请检查服务后重试。"));
  }

  function readNumber(headers, name) {
    const raw = headers.get(name);
    const value = raw === null || raw.trim() === "" ? NaN : Number(raw);
    if (!Number.isFinite(value) || value < 0) throw new Error("响应缺少有效的 " + name + "。");
    return value;
  }

  function schedulePlay(elapsed) {
    clearTimeout(state.timer);
    if (!state.playing) return;
    if (state.wanted.frame >= lastFrame()) {
      stop();
      transportState();
      return;
    }
    state.timer = setTimeout(() => {
      if (state.playing && sameView(state.wanted, state.shown) && !state.busy) {
        choose({frame: state.wanted.frame + 1});
      }
    }, Math.max(0, 1000 / state.scene.fps - elapsed));
  }

  async function pump() {
    if (state.busy || !state.wanted || state.failed || state.scrubTimer) return;
    if (sameView(state.wanted, state.shown)) {
      progress();
      schedulePlay(0);
      return;
    }
    const request = {...state.wanted};
    const revision = state.serial;
    const start = performance.now();
    let url;
    state.busy = true;
    transportState();
    try {
      const query = new URLSearchParams(request);
      const response = await fetch("/api/frame?" + query, {cache: "no-store"});
      if (!response.ok) throw await responseError(response);
      if (response.headers.get("Content-Type")?.split(";")[0].trim().toLowerCase() !== "image/png") {
        throw new Error("服务返回的画面格式有误。");
      }
      const index = readNumber(response.headers, "X-Frame-Index");
      const time = readNumber(response.headers, "X-Sim-Time-S");
      const ms = readNumber(response.headers, "X-Render-Ms");
      if (index !== request.frame || response.headers.get("X-Camera-Id") !== request.camera ||
          response.headers.get("X-Channel-Id") !== request.channel) {
        throw new Error("返回画面的帧号、机位或通道与请求不符，请重试。");
      }
      if (isLive()) state.frontier = Math.max(state.frontier, index);
      // Drain even a stale response before requesting the most recent view.
      const blob = await response.blob();
      if (revision !== state.serial) return;
      url = URL.createObjectURL(blob);
      const decoded = new Image();
      decoded.src = url;
      await decoded.decode();
      if (revision !== state.serial) return;
      const previous = state.shown;
      state.shown = {...request, time, ms, url};
      decoded.id = "picture";
      decoded.alt = cameraById(request.camera).label + "，" + chineseChannels[request.channel] + "，帧 " + index;
      ui.picture.replaceWith(decoded);
      ui.picture = decoded;
      url = null;
      if (previous?.url) URL.revokeObjectURL(previous.url);
      progress();
    } catch (error) {
      if (revision === state.serial) fail(error);
    } finally {
      if (url) URL.revokeObjectURL(url);
      state.busy = false;
      transportState();
      if (!state.failed) {
        if (revision !== state.serial) void pump();
        else schedulePlay(performance.now() - start);
      }
    }
  }

  function choose(change, delay = 0) {
    if (!state.scene) return;
    const desired = {...state.wanted, ...change};
    if (isLive()) desired.frame = Math.max(state.frontier, Math.min(state.frontier+1, desired.frame));
    desired.frame = Math.max(0, Math.min(lastFrame(), desired.frame));
    if (!Number.isSafeInteger(desired.frame)) return;
    if (!sameView(desired, state.wanted) || state.failed) state.serial++;
    state.wanted = desired;
    state.failed = false;
    clearTimeout(state.timer);
    clearTimeout(state.scrubTimer);
    state.scrubTimer = 0;
    progress();
    if (delay) state.scrubTimer = setTimeout(() => { state.scrubTimer = 0; void pump(); }, delay);
    else void pump();
  }

  function chooseCamera(id) {
    $("camera-toggle").setAttribute("aria-expanded", "false");
    document.querySelector(".camera-rack").classList.remove("expanded");
    choose({camera: id});
  }

  function step(delta) {
    if (!state.wanted || (delta < 0 ? ui.previous.disabled : ui.next.disabled)) return;
    stop();
    choose({frame: state.wanted.frame+delta});
  }

  function playPause() {
    if (ui.play.disabled) return;
    if (state.playing) { stop(); transportState(); return; }
    state.playing = true;
    if (state.wanted.frame >= lastFrame() && !isLive()) choose({frame: 0});
    else if (sameView(state.wanted, state.shown)) choose({frame: state.wanted.frame+1});
    else { transportState(); void pump(); }
  }

  let loading = false;
  async function loadScene() {
    if (loading || state.busy) return;
    loading = true;
    stage("正在连接场景");
    try {
      const response = await fetch("/api/scene", {cache: "no-store"});
      if (!response.ok) throw await responseError(response);
      const data = validateScene(await response.json());
      const prior = state.wanted;
      state.scene = data;
      state.failed = false;
      const oldUrl = state.shown?.url;
      state.shown = null;
      ui.picture.hidden = true;
      ui.picture.removeAttribute("src");
      if (oldUrl) URL.revokeObjectURL(oldUrl);
      state.frontier = data.current_frame ?? 0;
      $("title").textContent = data.title;
      document.title = data.title + " | Phymeo 渲染工作台";
      $("mode").textContent = data.mode === "live" ? "实时仿真" : "轨迹回放";
      $("mode").classList.toggle("live", data.mode === "live");
      $("mode").hidden = false;
      buildChoices();
      const channel = data.channels.some((c) => c.id === prior?.channel) ? prior.channel :
        (data.channels.find((c) => c.id === "rgb") || data.channels.find((c) => channelIds.has(c.id))).id;
      const camera = cameraById(prior?.camera)?.id ?? state.cameras[0].id;
      state.wanted = {camera, frame: isLive() ? state.frontier : prior?.frame ?? state.frontier, channel};
      choose({});
    } catch (error) { fail(error); } finally { loading = false; }
  }

  ui.play.addEventListener("click", playPause);
  ui.previous.addEventListener("click", () => step(-1));
  ui.next.addEventListener("click", () => step(1));
  ui.seek.addEventListener("input", () => {
    stop();
    choose({frame: Number(ui.seek.value)}, 90);
  });
  ui.seek.addEventListener("change", () => { stop(); choose({frame: Number(ui.seek.value)}); });
  function enterFrame() {
    if (!state.wanted || isLive()) return;
    const value = ui.number.valueAsNumber;
    if (Number.isSafeInteger(value)) { stop(); choose({frame: value}); }
    ui.number.value = state.wanted.frame;
  }
  ui.number.addEventListener("change", enterFrame);
  ui.number.addEventListener("keydown", (e) => { if (e.key === "Enter") { enterFrame(); ui.number.blur(); } });
  $("retry").addEventListener("click", () => { void loadScene(); });
  $("camera-toggle").addEventListener("click", () => {
    const open = document.querySelector(".camera-rack").classList.toggle("expanded");
    $("camera-toggle").setAttribute("aria-expanded", String(open));
  });
  const dialog = $("shortcuts");
  $("help").addEventListener("click", () => dialog.showModal());
  $("close-dialog").addEventListener("click", () => dialog.close());
  dialog.addEventListener("click", (e) => {
    if (e.target !== dialog) return;
    const r = dialog.getBoundingClientRect();
    if (e.clientX < r.left || e.clientX > r.right || e.clientY < r.top || e.clientY > r.bottom) dialog.close();
  });
  $("fullscreen").hidden = !document.fullscreenEnabled;
  $("fullscreen").addEventListener("click", async () => {
    try {
      if (document.fullscreenElement) await document.exitFullscreen();
      else await $("monitor").requestFullscreen();
    } catch { $("fullscreen").hidden = true; }
  });
  document.addEventListener("fullscreenchange", () => {
    const on = !!document.fullscreenElement;
    const label = on ? "退出全屏" : "全屏画面";
    $("fullscreen").title = label;
    $("fullscreen").setAttribute("aria-label", label);
    $("fullscreen").querySelector("use").setAttribute("href", on ? "#collapse" : "#expand");
  });
  document.addEventListener("keydown", (e) => {
    if (!state.scene || e.defaultPrevented || e.isComposing || e.altKey || e.ctrlKey || e.metaKey ||
        dialog.open || e.target.closest("input, textarea, select, [contenteditable=true]")) return;
    if (/^[0-8]$/.test(e.key) && state.cameraKeys[Number(e.key)]) {
      e.preventDefault(); chooseCamera(state.cameraKeys[Number(e.key)].id);
    } else if (e.key === "ArrowLeft" || e.key === "ArrowRight") {
      e.preventDefault(); step(e.key === "ArrowLeft" ? -1 : 1);
    } else if (e.code === "Space" && !e.repeat &&
               (!e.target.closest("button") || e.target.closest("#camera-list, #channels, .transport-buttons"))) {
      e.preventDefault(); playPause();
    }
  });
  document.addEventListener("visibilitychange", () => {
    if (document.hidden && state.playing) { stop(); transportState(); }
  });
  void loadScene();
})();
