"use strict";

(() => {
  const $ = (id) => document.getElementById(id);
  const state = {
    scene: null, wanted: null, shown: null, frontier: 0, serial: 0, busy: false,
    playing: false, timer: 0, scrubTimer: 0, failed: false, cameras: [], cameraKeys: [],
    after: null, notice: "", failures: 0, metadataFailures: 0, transfer: null,
  };
  const ui = {
    play: $("playback"), previous: $("prev-frame"), next: $("next-frame"), seek: $("seek"),
    number: $("frame-number"), status: $("play-state"), picture: $("picture"), message: $("stage-message"),
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
    if (!data || !["replay", "live", "observe"].includes(data.mode) || typeof data.title !== "string" ||
        !data.title.trim() || !Number.isFinite(data.fps) || data.fps <= 0) {
      throw new Error("服务返回的场景信息不完整。");
    }
    if (!(data.frame_count === null && data.mode !== "replay") &&
        (!Number.isSafeInteger(data.frame_count) || data.frame_count < 1)) {
      throw new Error("服务返回的帧数无效。");
    }
    if (data.current_frame != null && (!Number.isSafeInteger(data.current_frame) ||
        data.current_frame < 0 || (data.frame_count !== null && data.current_frame >= data.frame_count))) {
      throw new Error("服务返回的当前帧无效。");
    }
    keyedList(data.cameras, "机位");
    keyedList(data.channels, "通道");
    return data;
  }

  const observing = () => state.scene?.mode === "observe";
  const isLive = () => state.scene?.mode === "live";
  const lastFrame = () => state.scene?.frame_count == null ? Infinity : state.scene.frame_count - 1;
  const cameraById = (id) => state.cameras.find((c) => c.id === id);
  const channelById = (id) => state.scene.channels.find((ch) => ch.id === id);
  const sameView = (a, b) => a && b && a.camera === b.camera && a.channel === b.channel && a.frame === b.frame;

  function filterCameras() {
    const query = $("camera-search").value.trim().toLocaleLowerCase();
    let visible = 0;
    for (const button of $("camera-list").children) {
      const camera = cameraById(button.dataset.camera);
      button.hidden = !(camera.label + " " + camera.id).toLocaleLowerCase().includes(query);
      if (!button.hidden) visible++;
    }
    $("no-cameras").hidden = visible !== 0;
  }

  function buildChoices() {
    state.cameras = state.scene.cameras;
    state.cameraKeys = state.cameras.slice(0, 9);
    $("camera-count").textContent = state.cameras.length;
    $("search-box").hidden = state.cameras.length < 8;
    $("camera-search").value = "";
    $("no-cameras").hidden = true;
    const cameraButtons = state.cameras.map((camera) => {
      const button = document.createElement("button");
      button.type = "button";
      button.className = "camera-option";
      button.dataset.camera = camera.id;
      const label = document.createElement("span");
      label.textContent = camera.label;
      button.append(icon("camera"), label);
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
    const channels = state.scene.channels;
    $("channels").replaceChildren(...channels.map((ch) => {
      const button = document.createElement("button");
      button.type = "button";
      button.className = "channel";
      button.dataset.channel = ch.id;
      button.textContent = ch.label;
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
    const observe = observing();
    document.body.classList.toggle("observer", observe);
    $("transport-title").textContent = observe ? "实时监看" : "时间轴";
    $("observe-note").hidden = !observe;
    $("frame-entry").hidden = observe;
    $("shortcut-step").hidden = observe;
    $("shortcut-play").textContent = observe ? "冻结或跟随画面" : "播放或暂停";
    $("track").hidden = isLive() || observe;
    ui.previous.hidden = ui.next.hidden = observe;
    ui.number.readOnly = isLive();
    ui.number.title = isLive() ? "实时状态按步前进" : "输入从 0 开始的帧号，然后按 Enter";
    $("frame-limit").textContent = count === null ? "" : "/ " + (count-1);
    if (count !== null) ui.number.max = String(count-1); else ui.number.removeAttribute("max");
    ui.seek.max = String(count === null ? 0 : count-1);
  }

  function transportState() {
    const want = state.wanted;
    if (!want) return;
    const observe = observing();
    const frozen = observe && !state.playing;
    document.body.classList.toggle("frozen", frozen);
    const ready = !!state.shown && !state.failed;
    const atEnd = want.frame >= lastFrame();
    const liveAdvance = isLive() && (state.busy || want.frame > state.frontier);
    ui.previous.disabled = observe || !ready || isLive() || want.frame <= 0;
    ui.next.disabled = observe || !ready || atEnd || liveAdvance;
    ui.play.disabled = observe ? false : !ready || (atEnd && (isLive() || lastFrame() === 0) && !state.playing);
    ui.seek.disabled = observe || isLive() || lastFrame() === 0;
    if (!observe) {
      ui.seek.value = want.frame;
      ui.seek.setAttribute("aria-valuetext", "第 " + want.frame + " 帧");
    }
    ui.number.disabled = !ready;
    if (!observe && document.activeElement !== ui.number) ui.number.value = String(want.frame);
    const playText = observe ? (frozen ? "恢复跟随" : "冻结画面") : (state.playing ? "暂停" : "播放");
    ui.play.querySelector("span").textContent = playText;
    ui.play.querySelector("use").setAttribute("href", state.playing ? "#pause" : observe ? "#signal" : "#play");
    ui.play.title = playText + " (空格)";
    ui.play.setAttribute("aria-label", playText);
    ui.status.hidden = false;
    ui.status.classList.toggle("playing", state.playing && !state.failed && !state.failures);
    ui.status.classList.toggle("error", state.failed || !!state.failures);
    ui.status.textContent = state.failed ? "读取失败" : observe ?
      (frozen ? "画面已冻结" : state.failures ? "正在重连" : "实时跟随") : state.playing ? "播放中" :
      !sameView(want, state.shown) ? "正在取帧" : "已暂停";
    if (observe) $("observe-note").textContent = frozen ? "当前画面已冻结，恢复跟随后可切换视图。" :
      state.notice || "跟随仿真发布的最新画面";
    for (const button of $("camera-list").children) {
      button.setAttribute("aria-pressed", String(button.dataset.camera === want.camera));
      button.disabled = frozen;
    }
    for (const button of $("channels").children) {
      button.setAttribute("aria-pressed", String(button.dataset.channel === want.channel));
      button.disabled = frozen;
    }
    const camera = cameraById(want.camera);
    $("camera-title").textContent = camera.label;
    $("mobile-camera").textContent = camera.label;
  }

  function stage(title, detail = "", failed = false) {
    ui.message.hidden = false;
    ui.message.classList.toggle("failed", failed);
    if ($("stage-title").textContent !== title) $("stage-title").textContent = title;
    if ($("stage-detail").textContent !== detail) $("stage-detail").textContent = detail;
    $("retry").hidden = !failed;
  }

  function progress() {
    const view = state.shown;
    const want = state.wanted;
    const observe = observing();
    if (!view || view.camera !== want.camera || view.channel !== want.channel) {
      ui.picture.hidden = true;
      $("captured-frame").textContent = "等待画面";
      $("sim-time").textContent = "—";
      $("render-ms").textContent = "—";
      $("channel-note").textContent = "";
      $("source-size").hidden = true;
      if (observe) {
        stage(!state.playing ? "监看已冻结" : state.failures ? "等待连接恢复" : "等待仿真画面",
          !state.playing ? "恢复跟随后将显示最新帧。" : state.notice || cameraById(want.camera).label);
      } else stage("正在读取画面", cameraById(want.camera).label + "，帧 " + want.frame);
    } else {
      // Hold the last complete image during a seek, with its own frame and timing.
      ui.picture.hidden = false;
      ui.message.hidden = true;
      $("captured-frame").textContent = observe || sameView(want, view) ? "当前帧 " + view.frame :
        "显示帧 " + view.frame + "，正在读取 " + want.frame;
      $("source-size").hidden = false;
      $("source-size").textContent = ui.picture.naturalWidth + " × " + ui.picture.naturalHeight;
      $("sim-time").textContent = view.time.toFixed(3) + " s";
      $("render-ms").textContent = view.ms.toFixed(1) + " ms";
      const spec = state.scene.channels.find((ch) => ch.id === view.channel);
      const camera = cameraById(view.camera);
      const range = camera.depth_range_m;
      const calibrated = view.channel === "depth" && Array.isArray(range) && range.length === 2 && range.every(Number.isFinite);
      $("channel-note").textContent = (calibrated ? range[0] + "–" + range[1] + " m　" : "") +
        (spec.description || spec.label);
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
    $("source-size").hidden = true;
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

  function pollAfter(delay) {
    clearTimeout(state.timer);
    if (state.playing && !document.hidden) state.timer = setTimeout(() => { void pump(); }, delay);
  }

  async function pump() {
    if (state.busy || !state.wanted || state.failed || state.scrubTimer) return;
    const observe = observing();
    if (observe && (!state.playing || document.hidden)) return;
    if (!observe && sameView(state.wanted, state.shown)) {
      progress();
      schedulePlay(0);
      return;
    }
    const request = {...state.wanted};
    const revision = state.serial;
    const start = performance.now();
    let url;
    let delay = 0;
    let retryable = false;
    const controller = new AbortController();
    state.transfer = controller;
    const deadline = observe ? setTimeout(() => controller.abort(), 30000) : 0;
    state.busy = true;
    transportState();
    try {
      const query = new URLSearchParams(request);
      if (observe && state.after !== null) query.set("after", String(state.after));
      retryable = true;
      const response = await fetch("/api/frame?" + query, {cache: "no-store", signal: controller.signal});
      if (observe && response.status === 204) {
        if (revision === state.serial) {
          state.notice = state.shown && state.after !== null ? "等待仿真发布下一帧" : "该视图尚未发布画面。";
          state.failures = 0;
          progress();
        }
        delay = state.after === null ? 650 : 350;
        return;
      }
      if (!response.ok) {
        retryable = response.status === 429 || response.status === 408 || response.status >= 500;
        throw await responseError(response);
      }
      retryable = false;
      if (response.headers.get("Content-Type")?.split(";")[0].trim().toLowerCase() !== "image/png") {
        throw new Error("服务返回的画面格式有误。");
      }
      const index = readNumber(response.headers, "X-Frame-Index");
      const time = readNumber(response.headers, "X-Sim-Time-S");
      const ms = readNumber(response.headers, "X-Render-Ms");
      if (!Number.isSafeInteger(index) || (!observe && index !== request.frame) ||
          (observe && query.has("after") && index <= Number(query.get("after"))) ||
          response.headers.get("X-Camera-Id") !== request.camera ||
          response.headers.get("X-Channel-Id") !== request.channel) {
        throw new Error("返回画面的帧号、机位或通道与请求不符，请重试。");
      }
      if (isLive()) state.frontier = Math.max(state.frontier, index);
      // Drain even a stale response before requesting the most recent view.
      retryable = true;
      const blob = await response.blob();
      retryable = false;
      if (revision !== state.serial) return;
      url = URL.createObjectURL(blob);
      const decoded = new Image();
      decoded.src = url;
      await decoded.decode();
      if (revision !== state.serial) return;
      const previous = state.shown;
      state.shown = {...request, frame: index, time, ms, url};
      if (observe) {
        state.after = index;
        state.failures = 0;
        state.notice = "";
      }
      decoded.id = "picture";
      decoded.alt = cameraById(request.camera).label + "，" + channelById(request.channel).label + "，帧 " + index;
      ui.picture.replaceWith(decoded);
      ui.picture = decoded;
      url = null;
      if (previous?.url) URL.revokeObjectURL(previous.url);
      progress();
    } catch (error) {
      if (revision === state.serial) {
        if (observe && retryable) {
          state.failures++;
          delay = Math.min(5000, 350 * 2 ** Math.min(state.failures-1, 4));
          state.notice = "暂时无法接收画面，" + (delay/1000).toFixed(1) + " 秒后重试。";
          state.after = null;
          progress();
        } else fail(error);
      }
    } finally {
      clearTimeout(deadline);
      state.transfer = null;
      if (url) URL.revokeObjectURL(url);
      state.busy = false;
      transportState();
      if (!state.failed) {
        if (revision !== state.serial) void pump();
        else if (observe) pollAfter(delay || Math.max(15, 1000/state.scene.fps - (performance.now()-start)));
        else schedulePlay(performance.now() - start);
      }
    }
  }

  function choose(change, delay = 0) {
    if (!state.scene) return;
    if (observing()) {
      if (!state.playing) return;
      const next = {...state.wanted, ...change, frame: "latest"};
      if (!sameView(next, state.wanted) || state.failed) {
        state.serial++;
        state.after = null;
        state.notice = "";
        state.failures = 0;
      }
      state.wanted = next;
      state.failed = false;
      clearTimeout(state.timer);
      progress();
      void pump();
      return;
    }
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
    if (observing() && !state.playing) return;
    if ($("camera-search").value && [...$("camera-list").children].some((row) => row.dataset.camera === id && row.hidden)) {
      $("camera-search").value = "";
      filterCameras();
    }
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
    if (observing()) {
      state.serial++;
      state.playing = !state.playing;
      state.failed = false;
      state.failures = 0;
      state.notice = "";
      state.after = null;
      clearTimeout(state.timer);
      if (!state.playing && state.shown) {
        state.wanted = {camera: state.shown.camera, channel: state.shown.channel, frame: "latest"};
      }
      progress();
      if (state.playing) void pump();
      return;
    }
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
    clearTimeout(state.timer);
    stage("正在连接场景");
    let retryable = true;
    const controller = new AbortController();
    const deadline = setTimeout(() => controller.abort(), 20000);
    try {
      const response = await fetch("/api/scene", {cache: "no-store", signal: controller.signal});
      if (!response.ok) {
        retryable = response.status === 429 || response.status >= 500;
        throw await responseError(response);
      }
      retryable = false;
      const data = validateScene(await response.json());
      const prior = state.wanted;
      state.scene = data;
      state.failed = false;
      state.playing = data.mode === "observe";
      state.after = null;
      state.failures = state.metadataFailures = 0;
      state.notice = "";
      const oldUrl = state.shown?.url;
      state.shown = null;
      ui.picture.hidden = true;
      ui.picture.removeAttribute("src");
      if (oldUrl) URL.revokeObjectURL(oldUrl);
      state.frontier = data.current_frame ?? 0;
      $("title").textContent = data.title;
      document.title = data.title + " | Phymeo 仿真监看";
      $("mode").textContent = {live: "实时步进", observe: "仿真观察", replay: "轨迹回放"}[data.mode];
      $("mode").classList.toggle("live", data.mode !== "replay");
      $("mode").hidden = false;
      buildChoices();
      const channel = data.channels.some((c) => c.id === prior?.channel) ? prior.channel : data.channels[0].id;
      const camera = cameraById(prior?.camera)?.id ?? state.cameras[0].id;
      state.wanted = {camera, frame: observing() ? "latest" : isLive() ? state.frontier :
        Number.isSafeInteger(prior?.frame) ? prior.frame : state.frontier, channel};
      choose({});
    } catch (error) {
      fail(error);
      if (retryable) {
        const delay = Math.min(8000, 1000 * 2 ** Math.min(state.metadataFailures++, 3));
        if (!document.hidden) state.timer = setTimeout(() => { void loadScene(); }, delay);
        stage("正在重连场景", "等待渲染服务响应。", true);
      }
    } finally { clearTimeout(deadline); loading = false; }
  }

  $("camera-search").addEventListener("input", filterCameras);
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
    if (observing()) {
      state.serial++;
      state.after = null;
      clearTimeout(state.timer);
      if (document.hidden) state.transfer?.abort();
      else if (state.playing) void pump();
    } else if (document.hidden && state.playing) {
      stop(); transportState();
    }
    if (!state.scene && state.metadataFailures) {
      clearTimeout(state.timer);
      if (!document.hidden) void loadScene();
    }
  });
  void loadScene();
})();
