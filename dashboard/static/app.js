// Metacog live dashboard — pure JS, talks to /api/stream (SSE).
(() => {
  const $ = (id) => document.getElementById(id);

  let lastSnapshot = null;

  function fmt(n, digits = 3) {
    if (n === null || n === undefined || Number.isNaN(n)) return "—";
    if (typeof n === "number") return n.toFixed(digits);
    return n;
  }
  function fmtPct(n) {
    if (n === null || n === undefined || Number.isNaN(n)) return "—";
    return n.toFixed(1);
  }
  function fmtBytes(n) {
    if (!n) return "0 B";
    const u = ["B", "KB", "MB", "GB"]; let i = 0;
    while (n >= 1024 && i < u.length - 1) { n /= 1024; i++; }
    return `${n.toFixed(i ? 1 : 0)} ${u[i]}`;
  }
  function fmtTime(ts) {
    if (!ts) return "—";
    const d = new Date(ts * 1000);
    return d.toLocaleTimeString();
  }

  // -- render --------------------------------------------------------------
  function renderHeader(s) {
    const cfg = s.config || {};
    const baseModel = (cfg.base_model || "?").split("/").pop();
    $("headerMeta").textContent =
      `${baseModel} · lora_rank=${cfg.lora_rank || "?"} · group_size=${cfg.group_size || "?"} · lr=${cfg.lr || "?"}`;
  }

  function renderKPIs(s) {
    const step = s.current_step ?? (s.latest_step ? s.latest_step.step_index : null);
    const total = s.total_steps || 20;
    if (step) {
      $("kpiStep").textContent = step;
      $("kpiTotal").textContent = total;
      $("stepBar").style.width = `${(step / total) * 100}%`;
    } else {
      $("kpiStep").textContent = "—";
    }
    const ls = s.latest_step;
    if (ls) {
      $("kpiReward").textContent = fmt(ls.reward_mean);
      $("kpiRewardSub").textContent =
        `std ${fmt(ls.reward_std)} · step_time ${fmt(ls.step_time_s, 1)}s · loss ${fmt(ls.loss)}`;
      $("kpiAcc").textContent = fmtPct(ls.accuracy_pct);
      $("kpiAccSub").textContent =
        `${ls.outcomes?.correct || 0} correct · ${ls.outcomes?.humble_wrong || 0} humble · ${ls.outcomes?.overconfident_wrong || 0} over`;
      $("kpiOver").textContent = fmtPct(ls.overconfident_pct);
    } else {
      $("kpiReward").textContent = "—";
      $("kpiRewardSub").textContent = "";
      $("kpiAcc").textContent = "—";
      $("kpiAccSub").textContent = "";
      $("kpiOver").textContent = "—";
    }
  }

  function renderInit(s) {
    const init = s.init;
    if (!init) {
      $("initAcc").textContent = $("initEce").textContent = $("initConf").textContent = $("initAbst").textContent = "—";
      return;
    }
    $("initAcc").textContent = `${(init.accuracy * 100).toFixed(1)}%`;
    $("initEce").textContent = init.ece.toFixed(4);
    $("initConf").textContent = init.avg_confidence.toFixed(3);
    $("initAbst").textContent = (init.abstention * 100).toFixed(1) + "%";
  }

  function renderCfg(s) {
    const c = s.config || {};
    const rows = [
      ["base model", c.base_model],
      ["lora rank", c.lora_rank],
      ["group size", c.group_size],
      ["learning rate", c.lr],
      ["n steps", c.n_steps],
      ["save every", c.save_every],
      ["eval every", c.eval_every],
      ["loss fn", c.loss_fn],
      ["temperature", c.temperature],
    ];
    $("cfgTable").innerHTML = rows
      .filter(([_, v]) => v !== undefined)
      .map(([k, v]) => `<tr><td>${k}</td><td>${v}</td></tr>`).join("");
  }

  function renderCkpts(s) {
    const ck = s.checkpoints || [];
    if (!ck.length) {
      $("ckptsList").innerHTML = `<li class="muted">none yet</li>`;
      return;
    }
    $("ckptsList").innerHTML = ck.map((c) =>
      `<li><span class="name">${c.name}</span><span class="meta">${c.size_human} · ${c.modified_iso}</span></li>`
    ).join("");
  }

  function renderLog(s) {
    const tail = (s.log_tail || []).join("\n");
    $("logTail").textContent = tail || "(empty)";
    $("footLog").textContent = s.path || "—";
  }

  // -- charts (inline SVG, no libs) ----------------------------------------
  function drawRewardChart(s) {
    const svg = $("rewardChart");
    const W = 600, H = 200, P = 28;
    svg.innerHTML = "";
    const steps = (s.steps || []).slice();
    if (!steps.length) {
      svg.innerHTML = `<text x="${W / 2}" y="${H / 2}" text-anchor="middle" fill="#8a93a6" font-size="12">no data yet</text>`;
      return;
    }
    // x = step index, y = reward_mean
    const xs = steps.map((st, i) => i + 1);
    const ys = steps.map((st) => st.reward_mean ?? 0);
    const xmin = 1, xmax = Math.max(2, xs.length);
    const ymin = Math.min(0, ...ys) - 0.05;
    const ymax = Math.max(0.3, ...ys) + 0.05;

    const xScale = (x) => P + ((x - xmin) / (xmax - xmin)) * (W - 2 * P);
    const yScale = (y) => H - P - ((y - ymin) / (ymax - ymin)) * (H - 2 * P);

    // axes
    const ax = `M${P},${P} L${P},${H - P} L${W - P},${H - P}`;
    svg.innerHTML += `<path d="${ax}" stroke="#232a3a" fill="none" stroke-width="1"/>`;

    // y gridlines (4)
    for (let i = 0; i <= 4; i++) {
      const yv = ymin + (i * (ymax - ymin)) / 4;
      const y = yScale(yv);
      svg.innerHTML += `<line x1="${P}" y1="${y}" x2="${W - P}" y2="${y}" stroke="#1a2030" stroke-width="1"/>`;
      svg.innerHTML += `<text x="${P - 4}" y="${y + 3}" text-anchor="end" fill="#8a93a6" font-size="10">${yv.toFixed(2)}</text>`;
    }
    // x labels
    xs.forEach((x) => {
      svg.innerHTML += `<text x="${xScale(x)}" y="${H - P + 14}" text-anchor="middle" fill="#8a93a6" font-size="10">${x}</text>`;
    });

    // line
    const pts = xs.map((x, i) => `${xScale(x)},${yScale(ys[i])}`).join(" ");
    svg.innerHTML += `<polyline points="${pts}" fill="none" stroke="#6ee7b7" stroke-width="2"/>`;
    // points
    xs.forEach((x, i) => {
      svg.innerHTML += `<circle cx="${xScale(x)}" cy="${yScale(ys[i])}" r="3.5" fill="#6ee7b7"/>`;
      svg.innerHTML += `<text x="${xScale(x)}" y="${yScale(ys[i]) - 8}" text-anchor="middle" fill="#e6ebf2" font-size="10">${ys[i].toFixed(2)}</text>`;
    });
  }

  function drawOutcomes(s) {
    const svg = $("outcomesChart");
    svg.innerHTML = "";
    const ls = s.latest_step;
    if (!ls || !ls.outcomes) {
      svg.innerHTML = `<text x="100" y="100" text-anchor="middle" fill="#8a93a6" font-size="12">no data</text>`;
      return;
    }
    const o = ls.outcomes;
    const total = (o.correct || 0) + (o.humble_wrong || 0) + (o.overconfident_wrong || 0) || 1;
    const slices = [
      { label: "correct", v: o.correct || 0, color: "#22c55e" },
      { label: "humble_wrong", v: o.humble_wrong || 0, color: "#60a5fa" },
      { label: "overconfident_wrong", v: o.overconfident_wrong || 0, color: "#ef4444" },
    ];
    const cx = 100, cy = 100, r = 80, ir = 50;
    let a0 = -Math.PI / 2;
    slices.forEach((sl) => {
      if (sl.v === 0) return;
      const a1 = a0 + (sl.v / total) * Math.PI * 2;
      const x0o = cx + r * Math.cos(a0), y0o = cy + r * Math.sin(a0);
      const x1o = cx + r * Math.cos(a1), y1o = cy + r * Math.sin(a1);
      const x0i = cx + ir * Math.cos(a0), y0i = cy + ir * Math.sin(a0);
      const x1i = cx + ir * Math.cos(a1), y1i = cy + ir * Math.sin(a1);
      const large = (a1 - a0) > Math.PI ? 1 : 0;
      const d = `M${x0o},${y0o} A${r},${r} 0 ${large} 1 ${x1o},${y1o} L${x1i},${y1i} A${ir},${ir} 0 ${large} 0 ${x0i},${y0i} Z`;
      svg.innerHTML += `<path d="${d}" fill="${sl.color}" opacity="0.85"/>`;
      // label
      const am = (a0 + a1) / 2;
      const lx = cx + ((r + ir) / 2) * Math.cos(am);
      const ly = cy + ((r + ir) / 2) * Math.sin(am);
      if (sl.v / total > 0.04) {
        svg.innerHTML += `<text x="${lx}" y="${ly + 4}" text-anchor="middle" fill="#0b0d12" font-size="11" font-weight="600">${((sl.v / total) * 100).toFixed(0)}%</text>`;
      }
      a0 = a1;
    });
  }

  function setDot(state) {
    const d = $("connDot");
    d.classList.remove("ok", "bad");
    if (state === "ok") d.classList.add("ok");
    else if (state === "bad") d.classList.add("bad");
  }

  function applySnapshot(s) {
    lastSnapshot = s;
    renderHeader(s);
    renderKPIs(s);
    renderInit(s);
    renderCfg(s);
    renderCkpts(s);
    renderLog(s);
    drawRewardChart(s);
    drawOutcomes(s);
    $("footTime").textContent = `last update ${new Date().toLocaleTimeString()}`;
  }

  // -- SSE -----------------------------------------------------------------
  function connect() {
    const es = new EventSource("/api/stream");
    es.addEventListener("open", () => setDot("ok"));
    es.addEventListener("snapshot", (e) => {
      try {
        const s = JSON.parse(e.data);
        applySnapshot(s);
      } catch (err) {
        console.error("bad snapshot", err);
      }
    });
    es.addEventListener("error", () => {
      setDot("bad");
      // EventSource auto-reconnects; we just reflect state
    });
  }

  // -- initial fetch in case SSE is slow -----------------------------------
  fetch("/api/snapshot")
    .then((r) => r.json())
    .then((s) => applySnapshot(s))
    .catch(() => {});

  connect();
  setInterval(() => {
    if (lastSnapshot) {
      $("footTime").textContent = `last update ${new Date().toLocaleTimeString()}`;
    }
  }, 1000);
})();
