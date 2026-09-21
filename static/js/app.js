(() => {
  const request = async (url, options = {}) => {
    const response = await fetch(url, options);
    const text = await response.text();
    let body;
    try { body = text ? JSON.parse(text) : {}; }
    catch (_) { body = {error: "The server returned an unexpected response (HTTP " + response.status + ")."}; }
    if (!response.ok) throw new Error(body.error || "Request failed.");
    return body;
  };
  const esc = value => String(value ?? "").replace(/[&<>'"]/g, char => ({"&":"&amp;","<":"&lt;",">":"&gt;","'":"&#39;","\"":"&quot;"}[char]));
  const displayState = (key, value) => key === "valve_open" ? (value ? "OPEN" : "CLOSED") : key === "pump_running" ? (value ? "RUNNING" : "STOPPED") : ["tank_level", "tank_setpoint", "pump_speed", "valve_position"].includes(key) ? value + "%" : ["flow_rate", "inlet_flow_rate", "outlet_flow_rate"].includes(key) ? value + " L/s" : key === "net_level_rate" ? value + " %/s" : value;
  const updateState = state => {
    Object.entries(state).forEach(([key, value]) => document.querySelectorAll('[data-state="' + key + '"]').forEach(node => { node.textContent = displayState(key, value); }));
    const controls = {"pump-speed": state.pump_speed, "valve-position": state.valve_position, "tank-setpoint": state.tank_setpoint};
    Object.entries(controls).forEach(([id, value]) => { const input = document.getElementById(id); if (input && value !== undefined && document.activeElement !== input) input.value = value; });
  };
  const setTankFill = state => {
    const tank = document.querySelector(".tank");
    if (!tank) return;
    let water = tank.querySelector(".tank-water");
    if (!water) {
      water = document.createElement("div");
      water.className = "tank-water";
      water.setAttribute("aria-hidden", "true");
      tank.prepend(water);
      Object.assign(tank.style, {position: "relative", overflow: "hidden", background: "#0d1210"});
      Object.assign(water.style, {position: "absolute", zIndex: "0", left: "0", right: "0", bottom: "0", background: "linear-gradient(to top, rgba(17,78,108,.88), rgba(55,136,151,.68))", borderTop: "1px solid rgba(151,210,211,.8)", transition: "height .65s ease"});
      tank.querySelectorAll("span, strong").forEach(node => Object.assign(node.style, {position: "relative", zIndex: "1"}));
    }
    water.style.height = Math.max(0, Math.min(100, Number(state.tank_level))) + "%";
  };
  const evidence = event => {
    const rule = Boolean(event.rule_triggered);
    const lightgbm = Boolean(event.lightgbm_unsafe);
    const isolation = Boolean(event.anomaly_flag);
    const cell = (name, value, active) => '<div style="border:1px solid ' + (active ? "var(--risk)" : "rgba(62,207,142,.45)") + ';padding:8px;background:var(--panel-2);font:500 11px var(--mono);line-height:1.4"><span>' + esc(name) + ': ' + esc(value) + '</span></div>';
    const ruleStatus = rule ? "Triggered" : "Clear";
    const lightgbmStatus = rule ? "Not evaluated" : lightgbm ? "Triggered" : event.decision_source === "safety_rules" ? "Model unavailable" : "Clear";
    const isolationStatus = rule || lightgbm ? "Not evaluated" : isolation ? "Triggered" : event.decision_source === "safety_rules" ? "Model unavailable" : "Clear";
    const pipeline = '<div style="display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:6px;margin:12px 0">' + cell("Safety rule", ruleStatus, rule) + cell("LightGBM", lightgbmStatus, lightgbm) + cell("Isolation Forest", isolationStatus, isolation) + '</div>';
    const shapLabel = event.shap_scope === "anomaly_context" ? "Why this looked unusual" : "Why the model made this decision";
    const shap = event.shap_factors?.length ? '<div style="border-left:2px solid var(--accent);padding:9px;background:rgba(79,179,169,.06)"><small>' + esc(shapLabel) + '</small>' + event.shap_factors.map(factor => '<div style="padding:4px 0;color:' + (factor.direction === "raises risk" ? "var(--risk)" : "var(--ok)") + '">' + esc(factor.plain || factor.feature) + '</div>').join("") + '</div>' : "";
    return pipeline + shap;
  };
  const openEventIds = new Set();
  const logFilters = { status: "all", severity: "all", date: "", timeFrom: "", timeTo: "", search: "" };
  const parseEventTime = event => {
    const timestamp = event?.timestamp;
    if (!timestamp) return null;
    const date = new Date(timestamp);
    return Number.isNaN(date.getTime()) ? null : date;
  };
  const getFilteredEvents = events => events.filter(event => {
    const eventDate = parseEventTime(event);
    const statusMatch = logFilters.status === "all" || (logFilters.status === "alert" ? Boolean(event.unsafe || event.review_flag) : logFilters.status === "normal" ? !event.unsafe && !event.review_flag : logFilters.status === "review" ? Boolean(event.review_flag) : true);
    const severityMatch = logFilters.severity === "all" || (logFilters.severity === "high" ? Boolean(event.unsafe) : logFilters.severity === "review" ? Boolean(event.review_flag) : logFilters.severity === "normal" ? !event.unsafe && !event.review_flag : true);
    const dateMatch = !logFilters.date || !eventDate || eventDate.toISOString().slice(0, 10) === logFilters.date;
    const timeParts = eventDate ? { hours: eventDate.getHours(), minutes: eventDate.getMinutes() } : null;
    const fromMinutes = logFilters.timeFrom ? (Number(logFilters.timeFrom.slice(0, 2)) * 60) + Number(logFilters.timeFrom.slice(3, 5)) : null;
    const toMinutes = logFilters.timeTo ? (Number(logFilters.timeTo.slice(0, 2)) * 60) + Number(logFilters.timeTo.slice(3, 5)) : null;
    const timeMatch = (!logFilters.timeFrom && !logFilters.timeTo) || !timeParts || (() => {
      const currentMinutes = (timeParts.hours * 60) + timeParts.minutes;
      const afterFrom = !logFilters.timeFrom || currentMinutes >= fromMinutes;
      const beforeTo = !logFilters.timeTo || currentMinutes <= toMinutes;
      return afterFrom && beforeTo;
    })();
    const searchText = [event.classification, event.command_name, event.device, event.operator_summary, event.recommended_action, event.command_value].join(" ").toLowerCase();
    const searchMatch = !logFilters.search || searchText.includes(logFilters.search.toLowerCase());
    return statusMatch && severityMatch && dateMatch && timeMatch && searchMatch;
  });
  const card = event => { const reviewOnly = !event.unsafe && Boolean(event.review_flag); const sevClass = event.unsafe ? "high" : reviewOnly ? "med" : "low"; const sevLabel = event.unsafe ? "HIGH RISK" : reviewOnly ? "REVIEW" : "NORMAL"; return '<article class="alert ' + esc(event.severity || "normal") + (reviewOnly ? " review" : "") + (openEventIds.has(event.event_id) ? " open" : "") + '" data-event-id="' + esc(event.event_id) + '"><div class="alert-top"><span class="alert-sev sev-' + sevClass + '">' + sevLabel + '</span><span class="alert-time">' + esc(event.timestamp) + '</span></div><div class="alert-title">' + esc(event.classification || (event.unsafe ? "Unsafe command pattern" : "Normal operating command")) + '</div><div class="alert-sub">' + esc(event.command_name) + ' on ' + esc(event.device) + ' | requested value ' + esc(event.command_value) + '</div><div class="alert-detail"><p>' + esc(event.operator_summary || "Command analysed.") + '</p>' + evidence(event) + '<p>' + (event.engineer_explanation || event.explanation || []).map(esc).join(" ") + '</p><p>' + esc(event.recommended_action) + '</p></div></article>'; };
  const render = events => { const feed = document.getElementById("event-feed"); if (!feed) return; const visibleIds = new Set(events.map(event => event.event_id)); [...openEventIds].forEach(id => { if (!visibleIds.has(id)) openEventIds.delete(id); }); const filteredEvents = getFilteredEvents(events); feed.innerHTML = filteredEvents.length ? filteredEvents.map(card).join("") : '<p class="empty-alerts">No commands match the selected filter.</p>'; };
  const bindLogFilters = () => {
    const controls = [
      document.getElementById("log-status"),
      document.getElementById("log-severity"),
      document.getElementById("log-date"),
      document.getElementById("log-time-from"),
      document.getElementById("log-time-to"),
      document.getElementById("log-search")
    ].filter(Boolean);
    controls.forEach(control => control.addEventListener("input", () => {
      logFilters.status = document.getElementById("log-status")?.value || "all";
      logFilters.severity = document.getElementById("log-severity")?.value || "all";
      logFilters.date = document.getElementById("log-date")?.value || "";
      logFilters.timeFrom = document.getElementById("log-time-from")?.value || "";
      logFilters.timeTo = document.getElementById("log-time-to")?.value || "";
      logFilters.search = document.getElementById("log-search")?.value || "";
      const feed = document.getElementById("event-feed");
      if (feed && typeof window.__icsGuardEvents !== "undefined") render(window.__icsGuardEvents);
    }));
  };
  const updateControlAvailability = state => {
    document.querySelectorAll("[data-control]").forEach(button => {
      button.disabled = state.maintenance_activity === "shutdown" ? button.dataset.control !== "startup" : button.dataset.control === "startup";
    });
  };
  const removeConsoleCopy = () => {
    document.getElementById("reset-process")?.remove();
    document.querySelector('[data-control="normal"]')?.remove();
    document.querySelector("#tab-control .panel-head .meta")?.remove();
    const drainButton = document.querySelector('[data-control="maintenance_drain"]');
    const maintenancePanel = drainButton?.closest(".panel");
    if (!maintenancePanel) return;
    const heading = maintenancePanel.querySelector(".panel-head h2");
    if (heading) heading.textContent = "Maintenance";
    maintenancePanel.querySelector(".panel-body")?.classList.add("maintenance-actions");
    maintenancePanel.querySelectorAll("[data-control]").forEach(button => Object.assign(button.style, {marginRight: "10px", marginBottom: "8px"}));
    const maintenanceNote = maintenancePanel.querySelector("p.limits");
    if (maintenanceNote) maintenanceNote.textContent = "Planned Drain Pauses Automatic Control, Stops P-01, And Opens The Outlet.";
  };
  const refresh = async () => { try { const result = await request("/api/status"); updateState(result.state); updateControlAvailability(result.state); setTankFill(result.state); window.__icsGuardEvents = result.events || []; render(window.__icsGuardEvents); } catch (_) {} };
  removeConsoleCopy();
  bindLogFilters();
  document.querySelectorAll("[data-control]").forEach(button => button.addEventListener("click", async () => {
    const action = button.dataset.control;
    const value = action === "set_pump_speed" ? document.getElementById("pump-speed").value : action === "set_valve_position" ? document.getElementById("valve-position").value : action === "set_tank_setpoint" ? document.getElementById("tank-setpoint").value : undefined;
    const status = document.getElementById("control-status"); button.disabled = true; if (status) status.textContent = "Sending Modbus command…";
    try { const result = await request("/api/control", {method:"POST", headers:{"Content-Type":"application/json"}, body:JSON.stringify({action, value})}); updateState(result.state); updateControlAvailability(result.state); setTankFill(result.state); render(result.events); await refresh(); if (status) status.textContent = "Command sent through Modbus TCP."; } catch (error) { if (status) status.textContent = error.message; } finally { await refresh(); }
  }));
  document.getElementById("refresh-events")?.addEventListener("click", refresh);
  document.getElementById("event-feed")?.addEventListener("click", event => { const cardNode = event.target.closest(".alert"); if (!cardNode) return; const id = cardNode.dataset.eventId; if (openEventIds.has(id)) { openEventIds.delete(id); cardNode.classList.remove("open"); } else { openEventIds.add(id); cardNode.classList.add("open"); } });
  const updateReportDownload = () => {
    const link = document.getElementById("download-filtered-log");
    if (!link) return;
    const query = new URLSearchParams();
    const status = document.getElementById("report-status")?.value || "all";
    const dateFrom = document.getElementById("report-date-from")?.value || "";
    const dateTo = document.getElementById("report-date-to")?.value || "";
    const search = document.getElementById("report-search")?.value || "";
    if (status !== "all") query.set("status", status);
    if (dateFrom) query.set("date_from", dateFrom);
    if (dateTo) query.set("date_to", dateTo);
    if (search) query.set("search", search);
    const endpoint = document.getElementById("report-log-type")?.value || "/api/download/detections.csv";
    link.href = endpoint + (query.toString() ? "?" + query.toString() : "");
  };
  document.querySelectorAll("#report-log-type, #report-status, #report-date-from, #report-date-to, #report-search").forEach(control => control.addEventListener("input", updateReportDownload));
  updateReportDownload();
  document.getElementById("train-live-model")?.addEventListener("click", async event => { const button = event.currentTarget; button.disabled = true; const status = document.getElementById("live-training-status"); status.textContent = "Building live simulator detector…"; try { const result = await request("/api/train/live", {method:"POST", headers:{"Content-Type":"application/json"}}); status.textContent = "Build completed: " + result.records.toLocaleString() + " records processed."; setTimeout(() => location.reload(), 900); } catch (error) { status.textContent = error.message; button.disabled = false; } });
  document.querySelectorAll(".tab-btn").forEach(button => button.addEventListener("click", () => { document.querySelectorAll(".tab-btn").forEach(item => item.classList.remove("active")); document.querySelectorAll(".tabview").forEach(item => item.classList.remove("active")); button.classList.add("active"); document.getElementById("tab-" + button.dataset.tab)?.classList.add("active"); }));
  refresh(); setInterval(refresh, 2500);
})();
