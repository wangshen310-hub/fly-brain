"use strict";

(() => {
  const $ = (id) => document.getElementById(id);
  const canvas = $("board");
  const context = canvas.getContext("2d");
  const actionNames = { left: "向左转", forward: "向前走", right: "向右转", straight: "向前走", hold: "保持", none: "暂无选择" };
  const statusNames = { ready: "等待开始", idle: "等待开始", running: "实验运行中", paused: "已暂停", loading: "加载神经模型", preparing: "准备中", saving: "保存记忆中", feedback: "交付奖惩中", reward: "交付奖励中", punishment: "交付惩罚中", error: "需要处理", stopped: "已停止", finished: "本局结束" };
  const eventNames = { reward: "食物奖励", food: "吃到食物", aversive: "死亡惩罚", punishment: "死亡惩罚", death: "死亡", no_action: "无有效输出", decision: "神经决策", move: "移动", save: "保存记忆", saved: "已保存", reset: "新棋局", episode: "棋局", run: "开始", start: "开始", pause: "暂停", paused: "暂停", info: "记录", error: "错误", warning: "提示", load: "加载", loaded: "已加载", feedback: "奖惩" };
  let state = null;
  statusNames.warming = "中性画面预热中";
  statusNames.thinking = "神经计算中";
  statusNames.no_action = "无有效输出，惩罚后继续尝试";
  let connected = false;
  let polling = false;
  let pending = false;
  let lastBoardSignature = "";
  let lastEventSignature = "";
  let verifiedImage = null;
  let resetPending = null;
  let toastTimer = null;

  const number = (value) => typeof value === "number" && Number.isFinite(value) ? value : null;
  const count = (value) => number(value) === null ? "—" : value.toLocaleString("zh-CN");
  const firstNumber = (...values) => values.map(number).find((value) => value !== null) ?? null;
  const raw = (value) => value === undefined || value === null ? "—" : String(value);
  const text = (id, value) => { const node = $(id); if (node.textContent !== String(value)) node.textContent = value; };
  const lower = (value) => String(value ?? "").toLowerCase();

  function duration(milliseconds) {
    if (number(milliseconds) === null) return "—";
    return milliseconds < 1000 ? `${milliseconds.toLocaleString("zh-CN", { maximumFractionDigits: 1 })} ms` : `${(milliseconds / 1000).toLocaleString("zh-CN", { maximumFractionDigits: 2 })} s`;
  }

  function timestamp(value) {
    if (value === undefined || value === null || value === "") return "—";
    const date = new Date(typeof value === "number" && value < 1e12 ? value * 1000 : value);
    return Number.isNaN(date.getTime()) ? String(value) : date.toLocaleTimeString("zh-CN", { hour12: false });
  }

  function notify(message, error = false) {
    clearTimeout(toastTimer);
    const toast = $("toast");
    toast.textContent = message;
    toast.classList.toggle("error", error);
    toast.hidden = false;
    toastTimer = setTimeout(() => { toast.hidden = true; }, error ? 6500 : 4000);
  }

  function setConnection(ok, explanation = "") {
    connected = ok;
    $("autosave-status").classList.toggle("stale", !ok);
    $("connection-dot").className = `status-dot ${ok ? "ready" : "error"}`;
    text("connection-label", ok ? "本地实验已连接" : "连接中断 · 等待重连");
    if (!ok) {
      text("run-status", "连接中断");
      $("run-status").className = "pill error";
      text("state-message", state ? `无法获取新状态。当前保留最后一次真实快照；不能据此判断后端是否仍在运行。${explanation}` : `暂未连接到本地实验服务。${explanation}`);
      $("phase-dot").className = "status-dot error";
    }
    updateControls();
  }

  function updateControls() {
    const unavailable = !connected || pending;
    $("run-button").disabled = unavailable || state?.running === true;
    $("pause-button").disabled = unavailable || state?.running !== true;
    $("save-button").disabled = unavailable || state?.brain?.loaded !== true;
    $("reset-button").disabled = unavailable || !state?.game || resetPending !== null;
    $("run-button").querySelector("span").textContent = state?.running ? "实验运行中" : state?.game && number(state.game.step) > 0 ? "继续实验" : "开始实验";
    document.querySelectorAll(".control-strip button").forEach((button) => button.setAttribute("aria-busy", String(pending)));
  }

  function roundedRect(x, y, width, height, radius) {
    context.beginPath();
    if (context.roundRect) context.roundRect(x, y, width, height, radius);
    else context.rect(x, y, width, height);
    context.fill();
  }

  function directionVector(game) {
    const directions = { up: [0, -1], north: [0, -1], n: [0, -1], right: [1, 0], east: [1, 0], e: [1, 0], down: [0, 1], south: [0, 1], s: [0, 1], left: [-1, 0], west: [-1, 0], w: [-1, 0] };
    if (Number.isInteger(game.direction) && game.direction >= 0 && game.direction <= 3) return [[0, -1], [1, 0], [0, 1], [-1, 0]][game.direction];
    if (Array.isArray(game.direction) && game.direction.length === 2) return game.direction;
    const named = directions[lower(game.direction)];
    if (named) return named;
    if (game.snake.length > 1) {
      const dx = game.snake[0][0] - game.snake[1][0];
      const dy = game.snake[0][1] - game.snake[1][1];
      if (Math.abs(dx) + Math.abs(dy) === 1) return [dx, dy];
    }
    return null;
  }

  function drawBoard(force = false) {
    if (!context) return;
    const game = state?.game;
    const size = number(game?.size);
    canvas.dataset.episode = raw(game?.episode_id ?? game?.episode);
    canvas.dataset.step = raw(game?.step_id ?? game?.step);
    canvas.dataset.observationHash = typeof game?.observation_hash === "string" ? game.observation_hash : "";
    const valid = size !== null && Number.isInteger(size) && size > 1 && size <= 128 && Array.isArray(game.snake) && game.snake.every((cell) => Array.isArray(cell) && cell.length >= 2 && cell.slice(0, 2).every((n) => number(n) !== null));
    $("board-placeholder").hidden = valid;
    const signature = JSON.stringify(valid ? { size, snake: game.snake, food: game.food, direction: game.direction, alive: game.alive } : null);
    if (!force && signature === lastBoardSignature) return;
    lastBoardSignature = signature;
    const displayWidth = canvas.getBoundingClientRect().width || 640;
    const pixelRatio = Math.min(window.devicePixelRatio || 1, 2);
    canvas.width = Math.round(displayWidth * pixelRatio);
    canvas.height = canvas.width;
    context.setTransform(pixelRatio, 0, 0, pixelRatio, 0, 0);
    context.fillStyle = "#0c1510";
    context.fillRect(0, 0, displayWidth, displayWidth);
    if (!valid) return;
    text("board-size", `${size} × ${size}`);
    const unit = displayWidth / size;
    context.strokeStyle = "#203326";
    context.lineWidth = 0.65;
    context.beginPath();
    for (let i = 1; i < size; i++) {
      const edge = i * unit;
      context.moveTo(edge, 0); context.lineTo(edge, displayWidth);
      context.moveTo(0, edge); context.lineTo(displayWidth, edge);
    }
    context.stroke();
    const food = game.food;
    if (Array.isArray(food) && food.length >= 2 && food.slice(0, 2).every((n) => number(n) !== null)) {
      const x = (food[0] + 0.5) * unit;
      const y = (food[1] + 0.5) * unit;
      context.shadowColor = "#efbc7155";
      context.shadowBlur = unit * 0.5;
      context.fillStyle = "#efbc75";
      context.beginPath(); context.arc(x, y, unit * 0.21, 0, Math.PI * 2); context.fill();
      context.shadowBlur = 0;
      context.strokeStyle = "#efbc7538";
      context.lineWidth = 1;
      context.beginPath(); context.arc(x, y, unit * 0.34, 0, Math.PI * 2); context.stroke();
    }
    const inset = Math.max(1.3, unit * 0.08);
    for (let i = game.snake.length - 1; i >= 0; i--) {
      const [x, y] = game.snake[i];
      context.fillStyle = i === 0 ? game.alive === false ? "#d99686" : "#b8f4cb" : "#46a06e";
      roundedRect(x * unit + inset, y * unit + inset, unit - 2 * inset, unit - 2 * inset, Math.max(2, unit * 0.16));
    }
    const direction = directionVector(game);
    if (game.snake.length && direction) {
      const [hx, hy] = game.snake[0];
      const [dx, dy] = direction;
      context.fillStyle = "#1a4027";
      for (const side of [-1, 1]) {
        const x = (hx + 0.5 + dx * 0.15 - dy * side * 0.17) * unit;
        const y = (hy + 0.5 + dy * 0.15 + dx * side * 0.17) * unit;
        context.beginPath(); context.arc(x, y, Math.max(1, unit * 0.045), 0, Math.PI * 2); context.fill();
      }
    }
  }

  function renderChannels(decision) {
    const values = ["left", "forward", "right"].map((action) => number(decision?.scores?.[action]));
    const scale = Math.max(...values.map((value) => value === null ? 0 : Math.abs(value)), 0);
    const selected = lower(decision?.action);
    text("last-action", decision?.action === undefined ? "—" : decision.action === null ? "无有效输出" : actionNames[selected] ?? String(decision.action));
    ["left", "forward", "right"].forEach((action, index) => {
      const value = values[index];
      text(`${action}-value`, value === null ? "—" : value.toLocaleString("zh-CN", { maximumFractionDigits: 3 }));
      const bar = $(`${action}-bar`);
      bar.style.width = `${value === null || scale === 0 ? 0 : Math.abs(value) / scale * 100}%`;
      const row = bar.closest(".channel");
      row.classList.toggle("selected", selected === action || action === "forward" && selected === "straight");
      row.classList.toggle("negative", value !== null && value < 0);
    });
  }

  function renderFeedback(feedback) {
    const kind = lower(feedback?.kind);
    const reward = ["reward", "positive", "food"].includes(kind);
    const aversive = ["aversive", "negative", "punishment", "death"].includes(kind);
    const noActionPenalty = aversive && lower(feedback?.reason) === "no_action";
    $("feedback-kind").className = `feedback-value${reward ? " reward" : aversive ? " aversive" : ""}`;
    text("feedback-kind", reward ? "食物奖励 · +1" : noActionPenalty ? "无有效输出惩罚" : aversive ? "死亡惩罚" : kind === "none" ? "本次无奖惩" : kind ? String(feedback.kind) : "等待事件");
    const details = [];
    if (noActionPenalty) details.push("棋局与步数保持不变，完成惩罚后在运行状态下继续尝试");
    if (feedback?.message) details.push(String(feedback.message));
    const pulse = firstNumber(feedback?.stimulus_ms, feedback?.pulse_ms, feedback?.duration_ms);
    if (pulse !== null) details.push(`交付时长 ${duration(pulse)}`);
    const stimulusCurrent = number(feedback?.stimulus_current_mV_equivalent);
    if (stimulusCurrent !== null) details.push(`刺激强度 ${stimulusCurrent}（模型等效值）`);
    const spikes = firstNumber(feedback?.spikes, reward ? feedback?.reward_spikes : feedback?.aversive_spikes);
    if (spikes !== null) details.push(`记录到 ${count(spikes)} 个脉冲`);
    text("feedback-detail", details.length ? details.join(" · ") : kind === "none" ? "是否改变了记忆，需通过后端记录和对照实验判断。" : kind ? "具体刺激与记忆更新以原始状态记录为准。" : "吃食交付奖励，死亡或无有效输出交付惩罚。");
  }

  function sensorySyncError(message) {
    const error = new Error(message); error.name = "SensorySyncError"; return error;
  }

  function rgbFromRgba(rgba) {
    const rgb = new Uint8Array(rgba.length / 4 * 3);
    for (let input = 0, output = 0; input < rgba.length; input += 4) {
      rgb[output++] = rgba[input]; rgb[output++] = rgba[input + 1]; rgb[output++] = rgba[input + 2];
    }
    return rgb;
  }

  function sensoryMetadata(next) {
    const game = next.game ?? {};
    const episode = game.episode_id ?? game.episode;
    const step = game.step_id ?? game.step;
    const native = next.sensory;
    if (native && typeof native === "object" && !Array.isArray(native)) {
      return { ...native, native: true, expectedHash: native.rgb_sha256, source: native.data_url };
    }
    if (resetPending && !(number(episode) !== null && episode > resetPending.episode)) return { hidden: true, kind: "neutral_warmup", view: "neutral", label: "正在切换新棋局，等待中性预热完成", source_episode_id: null, source_step_id: null };
    const feedback = game.pending_feedback;
    if ((game.phase === "terminal" || game.terminal === true) && !feedback) return { hidden: true, kind: "neutral_warmup", view: "neutral", label: "终止后的过渡状态，暂不展示棋局感官画面", source_episode_id: episode, source_step_id: null };
    const inFeedback = game.phase === "feedback" && feedback;
    return {
      native: false, kind: inFeedback ? "feedback" : "snapshot", view: "egocentric_head_up",
      expectedHash: game.observation_hash, source: next.observation_url,
      source_episode_id: inFeedback ? feedback.episode_id ?? episode : episode,
      source_step_id: inFeedback ? feedback.step_id : step,
      source_heading: inFeedback ? null : game.direction,
      neural_input_active: null,
      label: inFeedback ? `奖惩画面来自动作前第 ${raw(feedback.step_id)} 步；棋盘已结算至第 ${raw(step)} 步` : "与该棋盘状态匹配的感官快照；是否正被输入以服务端阶段为准",
    };
  }

  async function decodeVerifiedSensory(blob, expectedHash) {
    let bitmap;
    try { bitmap = await createImageBitmap(blob, { colorSpaceConversion: "none", premultiplyAlpha: "none" }); }
    catch { throw sensorySyncError("感官图像尚未成功解码"); }
    try {
      if (bitmap.width !== 320 || bitmap.height !== 180) throw sensorySyncError("感官图像尺寸与 320 × 180 协议不一致");
      const preparedCanvas = document.createElement("canvas"); preparedCanvas.width = 320; preparedCanvas.height = 180;
      const preparedContext = preparedCanvas.getContext("2d", { willReadFrequently: true, colorSpace: "srgb" });
      preparedContext.drawImage(bitmap, 0, 0);
      const rgba = preparedContext.getImageData(0, 0, 320, 180).data;
      const digest = await crypto.subtle.digest("SHA-256", rgbFromRgba(rgba));
      const actualHash = Array.from(new Uint8Array(digest), (byte) => byte.toString(16).padStart(2, "0")).join("");
      if (actualHash !== expectedHash) throw sensorySyncError("棋盘与感官快照正在切换，哈希暂未匹配");
      return { canvas: preparedCanvas, hash: actualHash };
    } finally { bitmap.close(); }
  }

  async function prepareSensory(next, signal) {
    const metadata = sensoryMetadata(next);
    if (metadata.hidden) return { metadata, image: null };
    const expectedHash = typeof metadata.expectedHash === "string" ? metadata.expectedHash.toLowerCase() : "";
    if (!/^[a-f0-9]{64}$/.test(expectedHash)) throw sensorySyncError("等待服务端提供可核查的感官哈希");
    if (metadata.native && (metadata.width !== 320 || metadata.height !== 180)) throw sensorySyncError("感官快照尺寸元数据不符合协议");
    let source;
    if (metadata.native) {
      if (typeof metadata.source !== "string" || !metadata.source.startsWith("data:image/png;base64,") || metadata.source.length > 2 * 1024 * 1024) throw sensorySyncError("服务端内嵌快照不是有效 PNG 数据");
      source = metadata.source;
    } else {
      let resolved;
      try { resolved = new URL(metadata.source, window.location.href); } catch { throw sensorySyncError("感官快照地址不可用"); }
      if (resolved.origin !== window.location.origin || !["http:", "https:"].includes(resolved.protocol)) throw sensorySyncError("感官快照必须来自同一本地实验服务");
      source = resolved.href;
    }
    if (verifiedImage?.hash === expectedHash) return { metadata, image: verifiedImage };
    let response;
    try { response = await fetch(source, { cache: "no-store", signal, credentials: "same-origin" }); }
    catch (error) { if (error.name === "AbortError") throw error; throw sensorySyncError("暂时无法读取感官快照"); }
    if (!response.ok) throw sensorySyncError(`感官快照尚不可用（HTTP ${response.status}）`);
    const image = await decodeVerifiedSensory(await response.blob(), expectedHash);
    verifiedImage = image;
    return { metadata, image };
  }

  function renderPreparedSensory(prepared) {
    const imageCanvas = $("observation");
    const metadata = prepared.metadata;
    const visible = Boolean(prepared.image) && !metadata.hidden;
    imageCanvas.hidden = !visible;
    $("observation-placeholder").hidden = visible;
    $("sensory-source").classList.remove("syncing");
    imageCanvas.dataset.rgbSha256 = visible ? prepared.image.hash : "";
    imageCanvas.dataset.sourceEpisode = metadata.source_episode_id === null || metadata.source_episode_id === undefined ? "" : String(metadata.source_episode_id);
    imageCanvas.dataset.sourceStep = metadata.source_step_id === null || metadata.source_step_id === undefined ? "" : String(metadata.source_step_id);
    imageCanvas.dataset.kind = String(metadata.kind ?? "snapshot");
    imageCanvas.dataset.view = String(metadata.view ?? "unknown");
    if (visible) imageCanvas.getContext("2d").drawImage(prepared.image.canvas, 0, 0);
    else text("observation-placeholder", metadata.label || "等待匹配的感官快照");
    const source = metadata.source_step_id === null || metadata.source_step_id === undefined ? "来源步骤：中性 / 过渡" : `来源：第 ${raw(metadata.source_episode_id)} 局 · ${metadata.kind === "feedback" ? "动作前" : ""}第 ${raw(metadata.source_step_id)} 步`;
    text("sensory-source", `${source}${metadata.label ? `。${metadata.label}` : ""}`);
    text("sensory-badge", metadata.kind === "feedback" ? "动作前奖惩快照" : metadata.view === "neutral" ? "中性 / 过渡" : "该步感官快照");
    $("sensory-legend").hidden = !visible || !metadata.native || metadata.view === "neutral";
    text("sensory-legend", state?.game?.sensory_version === "fixed-world-distinct-body-v4" ? "珊瑚红：蛇头 · 绿色：身体 · 蓝色：食物 · 黑色：墙" : "珊瑚红：蛇头 · 蓝色：食物 · 黑色：身体 / 墙");
    text("sensory-view", metadata.view === "neutral" ? "中性或过渡阶段不代表新的棋局视野；仅展示服务端明确提供的画面。" : metadata.view === "fixed_world_north_up" ? "整个棋盘的位置和朝向固定，蛇头在棋盘内移动；食物吃到前保持原位置。" : "旧版跟随蛇头的视角。请打开当前更新版以使用固定棋盘画面。");
  }

  function commitPreparedSnapshot(next, prepared) {
    return new Promise((resolve) => requestAnimationFrame(() => {
      if (resetPending) {
        const episode = next.game?.episode_id ?? next.game?.episode;
        if (number(episode) !== null && episode > resetPending.episode) resetPending = null;
        else if (!next.sensory) prepared = { metadata: sensoryMetadata(next), image: null };
      }
      renderState(next, prepared); resolve();
    }));
  }

  function showSyncing(message) {
    setConnection(true);
    text("run-status", "快照同步中"); $("run-status").className = "pill waiting";
    text("sensory-badge", "快照同步中"); $("sensory-source").classList.add("syncing");
    text("state-message", `${message}。${state ? "暂时保留上一组成对校验的棋盘与感官快照。" : "等待首组成对校验的棋盘与感官快照。"}`);
    text("snapshot-time", "等待匹配快照 · 当前显示上次已校验状态");
  }

  function renderEvents(events) {
    const list = Array.isArray(events) ? events : [];
    const signature = JSON.stringify(list);
    if (signature === lastEventSignature) return;
    lastEventSignature = signature;
    text("event-count", `${list.length} 条`);
    const root = $("event-list");
    root.replaceChildren();
    if (!list.length) {
      const empty = document.createElement("p"); empty.className = "empty-log"; empty.textContent = "暂无后端事件记录。"; root.append(empty); return;
    }
    const fragment = document.createDocumentFragment();
    list.slice(-80).reverse().forEach((event) => {
      if (!event || typeof event !== "object") return;
      const row = document.createElement("div"); row.className = "event-row";
      const time = document.createElement("time"); time.className = "event-time"; time.textContent = timestamp(event.time);
      const kind = lower(event.kind);
      const noActionPenalty = ["aversive", "punishment", "feedback"].includes(kind) && lower(event.reason ?? event.feedback?.reason ?? event.outcome?.reason) === "no_action";
      const type = document.createElement("span"); type.className = "event-type";
      if (["reward", "food"].includes(kind)) type.classList.add("reward");
      if (["aversive", "death", "punishment"].includes(kind) || noActionPenalty) type.classList.add("aversive");
      if (kind === "error") type.classList.add("error");
      type.textContent = noActionPenalty ? "无有效输出惩罚" : eventNames[kind] ?? raw(event.kind);
      const message = document.createElement("span"); message.className = "event-message"; message.textContent = raw(event.message);
      row.append(time, type, message); fragment.append(row);
    });
    root.append(fragment);
  }

  function renderAutosave(autosave) {
    const unavailable = autosave?.mode === "compatibility" && (autosave.backend_connected !== true || Boolean(autosave.last_error));
    const enabled = autosave?.enabled === true && !unavailable;
    const saving = autosave?.saving === true && !unavailable;
    const interval = number(autosave?.interval_seconds);
    const label = !autosave ? "等待自动保存状态" : enabled ? `自动保存${interval !== null ? ` · ${count(interval)} 秒 / 暂停 / 奖惩后` : "已开启"}${saving ? " · 保存中" : ""}` : saving ? "正在保存状态" : "自动保存未开启";
    text("autosave-label", unavailable ? "自动保存暂不可用 · 请检查实验记录" : label);
    $("autosave-label").title = enabled ? `模型有新活动且距上次成功保存满${interval === null ? "设定间隔" : `${count(interval)}秒`}时，在当前神经计算或完整奖惩结束后保存；暂停时也会保存。不重复保存未变化的状态。` : "自动保存状态来自后端";
    $("autosave-dot").className = `status-dot${saving ? " waiting" : enabled ? " ready" : ""}`;
    const savedAt = autosave?.last_saved_at;
    const savedDate = savedAt ? new Date(savedAt) : null;
    const validDate = savedDate && !Number.isNaN(savedDate.getTime());
    const today = validDate && savedDate.toDateString() === new Date().toDateString();
    const saveTime = validDate ? (today ? timestamp(savedAt) : savedDate.toLocaleString("zh-CN", { hour12: false })) : null;
    text("autosave-time", saveTime ? `上次成功保存 ${saveTime}${autosave.last_reason === "restored" ? " · 已恢复" : ""}` : "尚未记录成功保存");
    const reasons = { periodic: "定时保存", paused: "暂停时保存", manual: "手动保存", feedback: "奖惩后保存", new_episode: "新棋局保存", restored: "从已保存状态恢复" };
    $("autosave-time").title = validDate ? `${savedDate.toLocaleString("zh-CN", { hour12: false })}${reasons[autosave.last_reason] ? ` · ${reasons[autosave.last_reason]}` : ""}` : "仅在后端确认保存成功后显示时间";
  }

  function renderState(next, prepared) {
    state = next;
    setConnection(true);
    const game = state.game ?? {};
    const brain = state.brain ?? {};
    const decision = state.last_decision ?? {};
    const status = lower(state.status);
    const retrying = status === "no_action" && state.running === true;
    const busy = ["loading", "preparing", "saving", "feedback", "reward", "punishment", "warming"].includes(status) || retrying;
    text("score", count(game.score)); text("episode", count(game.episode)); text("step", count(game.step)); text("total-food", count(state.stats?.total_food));
    text("run-status", status === "no_action" && state.running === false ? "已暂停 · 最近无有效输出" : statusNames[status] ?? (state.running ? "实验运行中" : state.status ? String(state.status) : "等待开始"));
    $("run-status").className = `pill${status === "error" ? " error" : busy ? " waiting" : state.running ? " running" : ""}`;
    $("phase-dot").className = `status-dot ${status === "error" ? "error" : busy ? "waiting" : brain.loaded ? "ready" : ""}`;
    text("state-message", retrying ? "无唯一有效神经输出；棋局保持不动，交付 200 ms 惩罚后继续尝试。" : typeof state.message === "string" && state.message ? state.message : state.running ? "服务端正在推进神经计算。每一步动作均由神经输出决定。" : "状态已连接。开始实验后，由服务端推进神经计算与棋局。");
    text("brain-status", brain.loaded ? "真实模型已加载" : status === "loading" ? "加载中" : "等待加载");
    text("neuron-count", count(brain.neurons)); text("edge-count", count(brain.edges));
    text("neural-time", duration(number(brain.neural_ms)));
    const seconds = firstNumber(decision.compute_seconds, decision.wall_seconds, brain.compute_seconds, brain.last_compute_seconds);
    const milliseconds = firstNumber(decision.compute_ms, brain.compute_ms, brain.last_compute_ms, seconds === null ? null : seconds * 1000);
    text("compute-time", duration(milliseconds));
    renderChannels(decision); renderFeedback(state.last_feedback); drawBoard(); renderPreparedSensory(prepared); renderEvents(state.events); renderAutosave(state.autosave);
    text("brain-details", JSON.stringify(brain, null, 2));
    text("decision-details", JSON.stringify({ last_decision: state.last_decision ?? null, last_feedback: state.last_feedback ?? null, stats: state.stats ?? null, autosave: state.autosave ?? null }, null, 2));
    text("snapshot-time", `状态更新 ${new Date().toLocaleTimeString("zh-CN", { hour12: false })}`);
    updateControls();
  }

  function validateCompanionAutosave(value) {
    const reasons = [null, "periodic", "paused", "manual", "feedback", "new_episode", "restored"];
    if (!value || typeof value !== "object" || Array.isArray(value)
      || value.mode !== "compatibility"
      || typeof value.enabled !== "boolean" || typeof value.saving !== "boolean"
      || typeof value.backend_connected !== "boolean"
      || !(value.last_error === null || typeof value.last_error === "string")
      || number(value.interval_seconds) === null || value.interval_seconds <= 0
      || !reasons.includes(value.last_reason)
      || !(value.last_saved_at === null || typeof value.last_saved_at === "string" && value.last_saved_at.length > 0 && !Number.isNaN(new Date(value.last_saved_at).getTime()))) return null;
    return {
      enabled: value.enabled,
      interval_seconds: value.interval_seconds,
      last_saved_at: value.last_saved_at,
      last_reason: value.last_reason,
      saving: value.saving,
      mode: "compatibility",
      backend_connected: value.backend_connected,
      last_error: value.last_error,
    };
  }

  async function readCompanionAutosave() {
    if (window.location.hostname !== "127.0.0.1" || window.location.port !== "8765" || window.location.protocol !== "http:") return null;
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 900);
    try {
      const response = await fetch("http://127.0.0.1:8766/status", { cache: "no-store", credentials: "omit", signal: controller.signal });
      if (!response.ok) return null;
      return validateCompanionAutosave(await response.json());
    } catch { return null; }
    finally { clearTimeout(timeout); }
  }

  async function poll() {
    if (polling) return;
    polling = true;
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 8000);
    try {
      for (let attempt = 0; attempt < 3; attempt++) {
        const response = await fetch("/api/state", { cache: "no-store", signal: controller.signal });
        if (!response.ok) throw new Error(`状态接口返回 HTTP ${response.status}`);
        const next = await response.json();
        if (!next || typeof next !== "object" || Array.isArray(next)) throw new Error("状态接口未返回有效对象");
        let prepared;
        try { prepared = await prepareSensory(next, controller.signal); }
        catch (error) { if (error.name !== "SensorySyncError") throw error; if (attempt < 2) continue; showSyncing(error.message); return; }
        if (next.autosave === undefined || next.autosave === null) {
          const autosave = await readCompanionAutosave();
          if (autosave) next.autosave = autosave;
        }
        await commitPreparedSnapshot(next, prepared);
        return;
      }
    } catch (error) {
      setConnection(false, error.name === "AbortError" ? " 状态请求超时。" : ` ${error.message}`);
    } finally { clearTimeout(timeout); polling = false; }
  }

  async function command(action) {
    if (pending || !connected) return;
    if (action === "reset") {
      resetPending = { episode: firstNumber(state?.game?.episode_id, state?.game?.episode) ?? -1 };
      renderPreparedSensory({ metadata: { hidden: true, kind: "neutral_warmup", view: "neutral", label: "新棋局命令已发送，等待服务端完成切换", source_episode_id: null, source_step_id: null }, image: null });
    }
    pending = true; updateControls();
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 30000);
    try {
      const response = await fetch("/api/command", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ action }), signal: controller.signal });
      let result = {};
      try { result = await response.json(); } catch { /* HTTP status remains authoritative when no JSON is returned. */ }
      if (!result || typeof result !== "object") result = {};
      if (!response.ok || result.ok === false || result.error) { if (action === "reset") resetPending = null; throw new Error(result.error || result.message || `HTTP ${response.status}`); }
      const names = { run: "开始命令已提交", pause: "暂停命令已提交", save: "保存记忆命令已提交", reset: "新棋局命令已提交，保留已保存的可塑状态" };
      notify(typeof result.message === "string" ? result.message : names[action]);
      await poll();
    } catch (error) { notify(error.name === "AbortError" ? "命令响应超时，结果尚未确认。请查看后端状态，避免重复提交。" : `命令未确认：${error.message}`, true); }
    finally { clearTimeout(timeout); pending = false; updateControls(); }
  }

  $("run-button").addEventListener("click", () => command("run"));
  $("pause-button").addEventListener("click", () => command("pause"));
  $("save-button").addEventListener("click", () => command("save"));
  $("reset-button").addEventListener("click", () => { $("reset-dialog").returnValue = "cancel"; $("reset-dialog").showModal(); });
  $("reset-dialog").addEventListener("close", () => { if ($("reset-dialog").returnValue === "confirm") command("reset"); });
  if (window.ResizeObserver) new ResizeObserver(() => drawBoard(true)).observe(canvas.parentElement);
  else window.addEventListener("resize", () => drawBoard(true));
  drawBoard(true);
  poll();
  setInterval(poll, 1000);
})();
