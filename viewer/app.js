(() => {
  "use strict";

  const state = {
    data: null,
    events: [],
    eventIndex: 0,
    playing: false,
    timer: null,
    viewMode: "memory",
    artifactById: new Map(),
    callByArtifact: new Map(),
  };

  const el = (id) => document.getElementById(id);
  const escapeHtml = (value) => String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
  const formatCell = (cell) => Array.isArray(cell) ? `[${cell.join(", ")}]` : "—";
  const formatNumber = (value) => Number.isFinite(value) ? value.toLocaleString() : "—";
  const formatLatency = (value) => {
    if (!Number.isFinite(value)) return "—";
    if (value >= 1000) return `${(value / 1000).toFixed(2)} s`;
    return `${value.toFixed(value < 10 ? 2 : 1)} ms`;
  };
  const cellKey = (cell) => Array.isArray(cell) ? cell.join(",") : "";

  function configuredAgentCell(agent, key) {
    const configured = state.data.map.agents?.[agent]?.[key];
    if (Array.isArray(configured)) return configured;
    return findCellSymbol(key === "start" ? "S" : "G");
  }

  function emptyKnowledge(starts) {
    return {
      P1: { publicCells: new Set([cellKey(starts.P1)]), hazards: new Map() },
      P2: { publicCells: new Set([cellKey(starts.P2)]), hazards: new Map() },
    };
  }

  function addPublicFootprint(knowledge, position) {
    if (!position) return;
    const { width, height } = state.data.map.dimensions;
    const radius = state.data.map.observation_radius;
    for (let y = position[1] - radius; y <= position[1] + radius; y += 1) {
      for (let x = position[0] - radius; x <= position[0] + radius; x += 1) {
        if (x >= 0 && y >= 0 && x < width && y < height) knowledge.publicCells.add(`${x},${y}`);
      }
    }
  }

  function truthHazardAt(cell) {
    for (const [hazard, cells] of Object.entries(state.data.map.hazards)) {
      if (cells.some((candidate) => cellKey(candidate) === cellKey(cell))) return hazard;
    }
    return null;
  }

  function inSensorFootprint(cell, position) {
    if (!position) return false;
    const radius = state.data.map.observation_radius;
    return Math.max(Math.abs(cell[0] - position[0]), Math.abs(cell[1] - position[1])) <= radius;
  }

  function eventAttemptLabels(events) {
    let attemptNumber = 0;
    return events.map((event, globalIndex) => {
      if (event.event_type === "attempt_started") attemptNumber += 1;
      return { ...event, globalIndex, attemptNumber: Math.max(attemptNumber, 1) };
    });
  }

  async function fetchJson(url) {
    const response = await fetch(url, { cache: "no-store" });
    const body = await response.json();
    if (!response.ok || body.error) throw new Error(body.error || `HTTP ${response.status}`);
    return body;
  }

  async function initialize() {
    bindControls();
    try {
      const listing = await fetchJson("/api/runs");
      const select = el("run-select");
      select.innerHTML = listing.runs.map((run) => {
        const outcome = run.success ? "✓" : "✕";
        const label = `${outcome} ${run.name} · ${run.planning_granularity || "unknown"}`;
        return `<option value="${escapeHtml(run.name)}">${escapeHtml(label)}</option>`;
      }).join("");
      const requested = new URLSearchParams(location.search).get("run");
      const selected = requested || listing.default_run || listing.runs[0]?.name;
      if (!selected) throw new Error("No completed runs found under results/runs.");
      select.value = selected;
      await loadRun(selected);
    } catch (error) {
      showError(error.message);
    }
  }

  async function loadRun(name) {
    stopPlayback();
    hideError();
    const data = await fetchJson(`/api/run?name=${encodeURIComponent(name)}`);
    state.data = data;
    state.events = eventAttemptLabels(data.events);
    state.eventIndex = 0;
    state.artifactById = new Map(data.artifacts.map((artifact) => [artifact.artifact_id, artifact]));
    associateCalls();
    history.replaceState(null, "", `?run=${encodeURIComponent(name)}`);
    renderSummary();
    renderTimeline();
    updatePlayback();
  }

  function associateCalls() {
    state.callByArtifact = new Map();
    const plans = state.data.artifacts.filter((item) => item.artifact_type === "plan");
    const planCalls = state.data.calls.filter((call) => ["route_plan", "chunk_plan", "step_plan", "plan"].includes(call.kind));
    let cursor = 0;
    plans.forEach((plan) => {
      let candidate = null;
      for (let i = cursor; i < planCalls.length; i += 1) {
        const call = planCalls[i];
        if (call.agent !== plan.agent || call.phase !== plan.phase) continue;
        candidate = call;
        cursor = i + 1;
        if (!call.semantic_validation_error) break;
      }
      if (candidate) state.callByArtifact.set(plan.artifact_id, candidate);
    });
  }

  function renderSummary() {
    const { summary } = state.data;
    el("metric-mode").textContent = summary.mode || "—";
    el("metric-planner").textContent = summary.planning_granularity === "chunk"
      ? `chunk · ${summary.chunk_size || 3}` : summary.planning_granularity || "—";
    const outcome = el("metric-outcome");
    outcome.textContent = summary.success ? "Success" : "Failure";
    outcome.className = summary.success ? "success" : "failure";
    el("metric-calls").textContent = formatNumber(summary.llm_calls);
    el("metric-tokens").textContent = formatNumber(summary.usage?.total_tokens);
    el("metric-latency").textContent = formatLatency(summary.latency_ms?.end_to_end);
    el("metric-recovery-latency").textContent = formatLatency(summary.latency_ms?.recovery);
    el("metric-fault").textContent = summary.fault_id || "None";
    el("event-slider").max = Math.max(0, state.events.length - 1);
  }

  function replayStateAt(index) {
    const starts = {
      P1: configuredAgentCell("P1", "start"),
      P2: configuredAgentCell("P2", "start"),
    };
    const replay = {
      positions: { P1: starts.P1, P2: starts.P2 },
      trails: { P1: [starts.P1], P2: [starts.P2] },
      triggered: { P1: null, P2: null },
      attemptNumber: 1,
      activePhase: null,
      messages: [],
      plans: [],
      knowledge: emptyKnowledge(starts),
    };
    for (let i = 0; i <= index && i < state.events.length; i += 1) {
      const event = state.events[i];
      if (event.event_type === "attempt_started") {
        replay.positions = { P1: starts.P1, P2: starts.P2 };
        replay.trails = { P1: [starts.P1], P2: [starts.P2] };
        replay.triggered = { P1: null, P2: null };
        replay.messages = [];
        replay.plans = [];
        replay.knowledge = emptyKnowledge(starts);
        replay.attemptNumber = event.attemptNumber;
        replay.activePhase = null;
      }
      if (event.phase) replay.activePhase = event.phase;
      if (event.event_type === "action_executed" && event.agent) {
        replay.positions[event.agent] = event.cell;
        replay.trails[event.agent].push(event.cell);
        replay.knowledge[event.agent].publicCells.add(cellKey(event.cell));
      }
      if (event.event_type === "action_noop" && event.agent) {
        replay.trails[event.agent].push(event.cell);
      }
      if (event.event_type === "observation_created" && event.agent) {
        const observation = state.artifactById.get(event.observation_artifact_id);
        (observation?.public_cells || []).forEach((item) => {
          replay.knowledge[event.agent].publicCells.add(cellKey(item.cell));
        });
        (observation?.sensor_fact_ids || event.sensor_fact_ids || []).forEach((factId) => {
          const fact = state.artifactById.get(factId);
          if (!fact?.cell || !fact?.hazard) return;
          replay.knowledge[event.agent].hazards.set(cellKey(fact.cell), {
            hazard: fact.hazard,
            source: "sensor",
            sourceAgent: event.agent,
            faulty: false,
          });
        });
      }
      if (event.event_type === "trap_triggered") {
        replay.triggered[event.agent] = event.cell;
        const hazard = truthHazardAt(event.cell);
        if (hazard) replay.knowledge[event.agent].hazards.set(cellKey(event.cell), {
          hazard,
          source: "execution",
          sourceAgent: event.agent,
          faulty: false,
        });
      }
      if (event.event_type === "message_delivered") {
        replay.messages.push(event.message_artifact_id);
        const message = state.artifactById.get(event.message_artifact_id);
        (message?.content?.message_facts || []).forEach((fact) => {
          if (!fact.cell || !fact.hazard || !event.recipient) return;
          replay.knowledge[event.recipient].hazards.set(cellKey(fact.cell), {
            hazard: fact.hazard,
            source: "message",
            sourceAgent: event.sender || message.agent,
            faulty: message.lifecycle_status === "faulty",
          });
        });
      }
      if (event.plan_artifact_id && ["coordinate_path_plan_created", "step_decision_created", "junction_decision_created", "action_chunk_plan_created"].includes(event.event_type)) {
        replay.plans.push(event.plan_artifact_id);
        const plan = state.artifactById.get(event.plan_artifact_id);
        if (event.agent && plan?.current_cell) addPublicFootprint(replay.knowledge[event.agent], plan.current_cell);
      }
    }
    return replay;
  }

  function findCellSymbol(symbol) {
    const rows = state.data.map.public_rows;
    for (let y = 0; y < rows.length; y += 1) {
      const x = rows[y].indexOf(symbol);
      if (x >= 0) return [x, y];
    }
    return [0, 0];
  }

  function renderMap(agent, replay) {
    const map = state.data.map;
    const container = el(`map-${agent.toLowerCase()}`);
    container.style.gridTemplateColumns = `repeat(${map.dimensions.width}, 1fr)`;
    container.style.gridTemplateRows = `repeat(${map.dimensions.height}, 1fr)`;
    const trail = replay.trails[agent];
    const trailLookups = {
      P1: new Map(replay.trails.P1.map((cell, i) => [cell.join(","), i])),
      P2: new Map(replay.trails.P2.map((cell, i) => [cell.join(","), i])),
    };
    const current = replay.positions[agent];
    const triggered = replay.triggered[agent];
    const knowledge = replay.knowledge[agent];
    const sensor = map.agents[agent].sensor;
    const starts = {
      P1: configuredAgentCell("P1", "start"),
      P2: configuredAgentCell("P2", "start"),
    };
    const goals = {
      P1: configuredAgentCell("P1", "goal"),
      P2: configuredAgentCell("P2", "goal"),
    };
    const visibleAgents = state.viewMode === "truth" ? ["P1", "P2"] : [agent];
    const cells = [];
    map.public_rows.forEach((row, y) => {
      [...row].forEach((tile, x) => {
        const key = `${x},${y}`;
        const cell = [x, y];
        const currentlyVisible = inSensorFootprint(cell, current);
        const terrainKnown = state.viewMode === "truth"
          || (state.viewMode === "current" && currentlyVisible)
          || (state.viewMode === "memory" && knowledge.publicCells.has(key));
        const classes = ["cell"];
        if (!terrainKnown) classes.push("fog");
        if (state.viewMode !== "truth" && currentlyVisible) classes.push("current-visible");
        if (terrainKnown && tile === "#") classes.push("wall");
        if (visibleAgents.some((item) => cellKey(starts[item]) === key)) classes.push("start");
        if (visibleAgents.some((item) => cellKey(goals[item]) === key)) classes.push("goal");
        visibleAgents.forEach((item) => {
          if (trailLookups[item].has(key)) classes.push(`visited-${item.toLowerCase()}`);
        });
        if (triggered && triggered[0] === x && triggered[1] === y) classes.push("trap-triggered");
        let hazardFact = null;
        if (state.viewMode === "truth") {
          const hazard = truthHazardAt(cell);
          if (hazard) hazardFact = { hazard, source: "truth", faulty: false };
        } else if (state.viewMode === "current" && currentlyVisible) {
          const hazard = truthHazardAt(cell);
          if (hazard === sensor) hazardFact = { hazard, source: "sensor", faulty: false };
        } else if (state.viewMode === "memory") {
          hazardFact = knowledge.hazards.get(key) || null;
        }
        if (hazardFact?.source === "message") classes.push("reported-belief");
        if (hazardFact?.faulty) classes.push("faulty-belief");
        const markers = visibleAgents
          .filter((item) => cellKey(replay.positions[item]) === key)
          .map((item) => `<span class="agent-marker ${item.toLowerCase()}">${item}</span>`)
          .join("");
        const marker = markers ? `<span class="agent-stack">${markers}</span>` : "";
        const hazardMark = hazardFact
          ? `<span class="hazard ${hazardFact.hazard} ${hazardFact.source === "message" ? "reported" : ""}">${hazardFact.hazard === "fire" ? "F" : "W"}${hazardFact.source === "message" ? "!" : ""}</span>`
          : "";
        const visit = state.viewMode !== "truth" && trailLookups[agent].has(key) && !marker
          ? `<span class="trail-index">${trailLookups[agent].get(key)}</span>` : "";
        const endpoints = visibleAgents.flatMap((item) => {
          const labels = [];
          if (cellKey(starts[item]) === key) labels.push(`S${item.slice(1)}`);
          if (cellKey(goals[item]) === key) labels.push(`G${item.slice(1)}`);
          return labels;
        });
        const base = endpoints.length
          ? `<span class="endpoint-label">${endpoints.join("·")}</span>`
          : terrainKnown && (tile === "S" || tile === "G") ? tile : "";
        const fogMark = terrainKnown ? "" : '<span class="fog-mark">?</span>';
        const hazardLabel = hazardFact ? `, ${hazardFact.hazard} from ${hazardFact.source}${hazardFact.faulty ? ", faulty" : ""}` : "";
        cells.push(`<div class="${classes.join(" ")}" aria-label="cell ${x},${y}, ${terrainKnown ? "known" : "unknown"}${hazardLabel}">${fogMark}${base}${hazardMark}${visit}${marker}</div>`);
      });
    });
    container.innerHTML = cells.join("");
    const totalCells = map.dimensions.width * map.dimensions.height;
    const viewNote = state.viewMode === "memory"
      ? `${knowledge.publicCells.size}/${totalCells} cells remembered`
      : state.viewMode === "current"
        ? `radius-${map.observation_radius} ${sensor} sensor`
        : "omniscient analysis view";
    const positionText = state.viewMode === "truth" && map.world_model === "shared"
      ? `P1 ${formatCell(replay.positions.P1)} → ${formatCell(goals.P1)} · P2 ${formatCell(replay.positions.P2)} → ${formatCell(goals.P2)}`
      : `Position ${formatCell(current)} → goal ${formatCell(goals[agent])} · ${trail.length - 1} moves · ${viewNote}`;
    el(`position-${agent.toLowerCase()}`).textContent = positionText;
  }

  function eventTypeLabel(type) {
    return ({
      attempt_started: "Attempt started",
      attempt_finished: "Attempt finished",
      round_started: "Tick started",
      round_finished: "Tick finished",
      observation_created: "Observation",
      sensor_fact_discovered: "New sensor fact",
      sensor_fact_reobserved: "Sensor fact re-observed",
      message_delivered: "Message delivered",
      fault_injected: "Fault injected",
      fault_consumed: "Fault consumed",
      action_chunk_plan_created: "Chunk plan",
      coordinate_path_plan_created: "Route plan",
      step_decision_created: "Step plan",
      junction_decision_created: "Junction decision",
      action_executed: "Action executed",
      action_noop: "Invalid action / STAY",
      trap_triggered: "Trap triggered",
      navigation_loop_detected: "Navigation loop",
      plan_interrupted: "Plan interrupted",
      plan_queue_discarded_at_target: "Plan completed",
    })[type] || type.replaceAll("_", " ");
  }

  function renderEventPanel(event) {
    const artifactId = event.plan_artifact_id || event.message_artifact_id || event.observation_artifact_id || event.artifact_id || event.action_artifact_id;
    const artifact = artifactId ? state.artifactById.get(artifactId) : null;
    const details = Object.entries(event)
      .filter(([key]) => !["globalIndex", "attemptNumber", "event_type", "failure"].includes(key))
      .map(([key, value]) => `<dt>${escapeHtml(key)}</dt><dd>${escapeHtml(typeof value === "object" ? JSON.stringify(value) : value)}</dd>`)
      .join("");
    const reason = artifact?.reason ? `<p class="reason">${escapeHtml(artifact.reason)}</p>` : "";
    const failure = event.failure ? `<p class="reason">${escapeHtml(event.failure.error || JSON.stringify(event.failure))}</p>` : "";
    el("event-panel").innerHTML = `
      <div class="event-title">
        <div><p class="eyebrow">Event ${event.globalIndex}</p><h3>${escapeHtml(eventTypeLabel(event.event_type))}</h3></div>
        <code>attempt ${event.attemptNumber}</code>
      </div>
      <dl class="detail-list">${details || "<dt>State</dt><dd>No additional fields</dd>"}</dl>
      ${reason}${failure}`;
  }

  function messageHtml(id, currentEvent) {
    const message = state.artifactById.get(id);
    if (!message) return "";
    const facts = message.content?.message_facts || [];
    const factText = facts.length ? facts.map((fact) => `${fact.hazard} ${formatCell(fact.cell)}`).join(" · ") : "No facts";
    const isCurrent = currentEvent.message_artifact_id === id;
    return `<article class="feed-item ${message.lifecycle_status === "faulty" ? "faulty" : ""} ${isCurrent ? "current" : ""}">
      <div class="feed-head"><strong>${escapeHtml(message.agent)} → ${escapeHtml(message.recipient)}</strong><span>${escapeHtml(message.phase)}</span></div>
      <p>${escapeHtml(factText)}</p>
      <p>${escapeHtml(message.content?.note || "")}</p>
      ${message.controlled_fault_id ? `<code>fault: ${escapeHtml(message.controlled_fault_id)}</code>` : ""}
    </article>`;
  }

  function planHtml(id, currentEvent) {
    const plan = state.artifactById.get(id);
    if (!plan) return "";
    const call = state.callByArtifact.get(id);
    const actions = plan.direct_actions || plan.compiled_actions || (plan.next_action ? [plan.next_action] : []);
    const path = plan.proposed_path_cells || plan.predicted_cells || (plan.next_cell ? [plan.next_cell] : []);
    const isCurrent = currentEvent.plan_artifact_id === id;
    const dependencyCount = (plan.declared_dependency_ids || []).length;
    const rawTrace = call ? `<details>
      <summary>Prompt and raw LLM response</summary>
      <p><code>Prompt</code></p>
      <pre>${escapeHtml(call.prompt?.at(-1)?.content || "Unavailable")}</pre>
      <p><code>Response</code></p>
      <pre>${escapeHtml(call.response_text || "Unavailable")}</pre>
    </details>` : "";
    return `<article class="feed-item ${isCurrent ? "current" : ""}">
      <div class="feed-head"><strong>${escapeHtml(plan.agent)} · ${escapeHtml(plan.planning_granularity)}</strong><span>${escapeHtml(plan.phase)}</span></div>
      ${actions.length ? `<div class="action-sequence">${actions.map((action) => `<span>${escapeHtml(action)}</span>`).join("")}</div>` : ""}
      <p>${escapeHtml(plan.reason || "No model reason recorded.")}</p>
      <p><code>waypoint ${escapeHtml(formatCell(plan.waypoint))} · deps ${dependencyCount}${path.length ? ` · cells ${escapeHtml(JSON.stringify(path))}` : ""}</code></p>
      ${rawTrace}
    </article>`;
  }

  function renderInspector(replay, event) {
    renderEventPanel(event);
    el("messages-panel").innerHTML = replay.messages.length
      ? `<div class="feed">${replay.messages.map((id) => messageHtml(id, event)).join("")}</div>`
      : '<p class="empty">No messages delivered yet in this attempt.</p>';
    el("plans-panel").innerHTML = replay.plans.length
      ? `<div class="feed">${replay.plans.map((id) => planHtml(id, event)).join("")}</div>`
      : '<p class="empty">No LLM plan created yet in this attempt.</p>';
  }

  function renderTimeline() {
    el("timeline").innerHTML = state.events.map((event, index) => {
      let typeClass = "";
      if (["action_executed", "action_noop"].includes(event.event_type)) typeClass = `action ${event.agent === "P2" ? "p2" : ""}`;
      else if (event.event_type.includes("message") || event.event_type === "fault_injected") typeClass = "message";
      else if (event.event_type.includes("plan") || event.event_type.includes("decision")) typeClass = "plan";
      else if (["trap_triggered", "navigation_loop_detected"].includes(event.event_type)) typeClass = "failure";
      else if (event.event_type.includes("attempt")) typeClass = "attempt";
      return `<button type="button" class="timeline-event ${typeClass}" data-event-index="${index}" aria-label="${escapeHtml(`${index}: ${eventTypeLabel(event.event_type)}`)}"></button>`;
    }).join("");
  }

  function updatePlayback() {
    if (!state.data || !state.events.length) return;
    const event = state.events[state.eventIndex];
    const replay = replayStateAt(state.eventIndex);
    el("event-slider").value = state.eventIndex;
    el("event-counter").textContent = `${state.eventIndex + 1} / ${state.events.length}`;
    el("attempt-label").textContent = `Attempt ${replay.attemptNumber}`;
    el("phase-heading").textContent = replay.activePhase || eventTypeLabel(event.event_type);
    el("prev-button").disabled = state.eventIndex === 0;
    el("next-button").disabled = state.eventIndex === state.events.length - 1;
    el("play-button").textContent = state.playing ? "Pause" : "Play";
    const captions = {
      memory: "Two belief views of the same physical world: accumulated observations, private sensor facts, and received messages.",
      current: "Two live views of the same physical world. Each robot sees only its local terrain and assigned hazard type.",
      truth: "One omniscient shared-world view with both robots, configured starts/goals, paths, and all true hazards.",
    };
    el("view-caption").textContent = captions[state.viewMode];
    const sharedWorld = state.data.map.world_model === "shared";
    document.querySelector(".lane-grid").classList.toggle(
      "truth-mode",
      sharedWorld && state.viewMode === "truth",
    );
    el("map-title-p1").textContent = sharedWorld && state.viewMode === "truth"
      ? "Shared world · Ground truth"
      : sharedWorld ? "P1 · Belief view" : "P1 · Legacy lane A";
    el("map-title-p2").textContent = sharedWorld ? "P2 · Belief view" : "P2 · Legacy lane B";
    el("map-subtitle-p1").textContent = sharedWorld && state.viewMode === "truth"
      ? "Both agents · all hazards"
      : "Fire sensor";
    el("map-subtitle-p2").textContent = "Water sensor";
    renderMap("P1", replay);
    renderMap("P2", replay);
    renderInspector(replay, event);
    let currentTimelineButton = null;
    document.querySelectorAll(".timeline-event").forEach((button, index) => {
      button.classList.toggle("current", index === state.eventIndex);
      if (index === state.eventIndex) currentTimelineButton = button;
    });
    if (currentTimelineButton) {
      const timeline = el("timeline");
      timeline.scrollLeft = Math.max(
        0,
        currentTimelineButton.offsetLeft - timeline.clientWidth / 2,
      );
    }
  }

  function setEventIndex(index) {
    state.eventIndex = Math.max(0, Math.min(index, state.events.length - 1));
    updatePlayback();
  }

  function togglePlayback() {
    if (state.playing) {
      stopPlayback();
      return;
    }
    if (state.eventIndex >= state.events.length - 1) state.eventIndex = 0;
    state.playing = true;
    updatePlayback();
    scheduleNext();
  }

  function scheduleNext() {
    clearTimeout(state.timer);
    if (!state.playing) return;
    state.timer = setTimeout(() => {
      if (state.eventIndex >= state.events.length - 1) {
        stopPlayback();
        return;
      }
      state.eventIndex += 1;
      updatePlayback();
      scheduleNext();
    }, Number(el("speed-select").value));
  }

  function stopPlayback() {
    state.playing = false;
    clearTimeout(state.timer);
    state.timer = null;
    if (el("play-button")) el("play-button").textContent = "Play";
  }

  function bindControls() {
    el("run-select").addEventListener("change", (event) => loadRun(event.target.value).catch((error) => showError(error.message)));
    el("prev-button").addEventListener("click", () => { stopPlayback(); setEventIndex(state.eventIndex - 1); });
    el("next-button").addEventListener("click", () => { stopPlayback(); setEventIndex(state.eventIndex + 1); });
    el("play-button").addEventListener("click", togglePlayback);
    el("event-slider").addEventListener("input", (event) => { stopPlayback(); setEventIndex(Number(event.target.value)); });
    el("speed-select").addEventListener("change", () => { if (state.playing) scheduleNext(); });
    el("view-mode").addEventListener("change", (event) => { state.viewMode = event.target.value; updatePlayback(); });
    el("timeline").addEventListener("click", (event) => {
      const button = event.target.closest("[data-event-index]");
      if (!button) return;
      stopPlayback();
      setEventIndex(Number(button.dataset.eventIndex));
    });
    document.querySelector(".tabs").addEventListener("click", (event) => {
      const tab = event.target.closest("[data-tab]");
      if (!tab) return;
      document.querySelectorAll(".tab").forEach((button) => {
        const active = button === tab;
        button.classList.toggle("active", active);
        button.setAttribute("aria-selected", String(active));
      });
      document.querySelectorAll(".tab-panel").forEach((panel) => { panel.hidden = panel.id !== `${tab.dataset.tab}-panel`; });
    });
    window.addEventListener("keydown", (event) => {
      if (event.target.matches("select, input, button")) return;
      if (event.key === "ArrowLeft") { stopPlayback(); setEventIndex(state.eventIndex - 1); }
      if (event.key === "ArrowRight") { stopPlayback(); setEventIndex(state.eventIndex + 1); }
      if (event.key === " ") { event.preventDefault(); togglePlayback(); }
    });
  }

  function showError(message) {
    const banner = el("error-banner");
    banner.textContent = message;
    banner.hidden = false;
  }

  function hideError() { el("error-banner").hidden = true; }

  initialize();
})();
