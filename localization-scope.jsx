import React, { useRef, useState, useEffect, useCallback } from "react";

/* ============================================================
   Anchor-free localization core  (verified against Python/SciPy)
   ============================================================ */
function mulberry32(a) {
  return function () {
    a |= 0; a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
function makeNetwork(n, radius, seed) {
  const rnd = mulberry32(seed);
  const pos = Array.from({ length: n }, () => [rnd(), rnd()]);
  const adj = Array.from({ length: n }, () => []);
  const r2 = radius * radius;
  for (let i = 0; i < n; i++)
    for (let j = i + 1; j < n; j++) {
      const dx = pos[i][0] - pos[j][0], dy = pos[i][1] - pos[j][1];
      if (dx * dx + dy * dy <= r2) { adj[i].push(j); adj[j].push(i); }
    }
  return { pos, adj };
}
function bfs(adj, s) {
  const n = adj.length, d = new Array(n).fill(-1);
  d[s] = 0; const q = [s];
  for (let h = 0; h < q.length; h++) {
    const u = q[h];
    for (const v of adj[u]) if (d[v] < 0) { d[v] = d[u] + 1; q.push(v); }
  }
  return d;
}
function hopMatrix(adj) { return adj.map((_, i) => bfs(adj, i)); }
function diameter(adj) {
  let mx = 0;
  for (let i = 0; i < adj.length; i++) {
    const d = bfs(adj, i);
    for (const x of d) { if (x < 0) return -1; if (x > mx) mx = x; }
  }
  return mx;
}
function matVec(M, v) {
  const n = v.length, out = new Array(n).fill(0);
  for (let i = 0; i < n; i++) { let s = 0; const Mi = M[i]; for (let j = 0; j < n; j++) s += Mi[j] * v[j]; out[i] = s; }
  return out;
}
function norm(v) { let s = 0; for (const x of v) s += x * x; return Math.sqrt(s); }
function topEig(M, iters, rnd) {
  const n = M.length;
  let v = Array.from({ length: n }, () => rnd() - 0.5);
  let nv = norm(v); v = v.map((x) => x / nv);
  for (let k = 0; k < iters; k++) {
    const w = matVec(M, v); nv = norm(w);
    if (nv < 1e-12) break;
    v = w.map((x) => x / nv);
  }
  const Mv = matVec(M, v); let lam = 0;
  for (let i = 0; i < n; i++) lam += v[i] * Mv[i];
  return { lam, v };
}
function classicalMDS(D, seed) {
  const n = D.length;
  const D2 = D.map((row) => row.map((x) => x * x));
  const rowMean = D2.map((r) => r.reduce((a, b) => a + b, 0) / n);
  let grand = 0; for (const m of rowMean) grand += m; grand /= n;
  const B = Array.from({ length: n }, (_, i) =>
    Array.from({ length: n }, (_, j) => -0.5 * (D2[i][j] - rowMean[i] - rowMean[j] + grand)));
  const rnd = mulberry32(seed ^ 0x9e3779b9);
  const e1 = topEig(B, 80, rnd);
  for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) B[i][j] -= e1.lam * e1.v[i] * e1.v[j];
  const e2 = topEig(B, 80, rnd);
  const s1 = Math.sqrt(Math.max(e1.lam, 0)), s2 = Math.sqrt(Math.max(e2.lam, 0));
  return e1.v.map((_, i) => [e1.v[i] * s1, e2.v[i] * s2]);
}
function smacofStep(X, D) {
  const n = X.length;
  const out = Array.from({ length: n }, () => [0, 0]);
  for (let i = 0; i < n; i++) {
    let sx = 0, sy = 0, cnt = 0;
    for (let j = 0; j < n; j++) {
      if (i === j || D[i][j] <= 0) continue;
      const dx = X[i][0] - X[j][0], dy = X[i][1] - X[j][1];
      let dist = Math.sqrt(dx * dx + dy * dy); if (dist < 1e-9) dist = 1e-9;
      const ratio = D[i][j] / dist;
      sx += ratio * dx; sy += ratio * dy; cnt++;
    }
    out[i][0] = cnt ? sx / cnt : X[i][0];
    out[i][1] = cnt ? sy / cnt : X[i][1];
  }
  return out;
}
function stress(X, D) {
  let s = 0;
  for (let i = 0; i < X.length; i++)
    for (let j = i + 1; j < X.length; j++) {
      if (D[i][j] <= 0) continue;
      const dx = X[i][0] - X[j][0], dy = X[i][1] - X[j][1];
      s += (Math.sqrt(dx * dx + dy * dy) - D[i][j]) ** 2;
    }
  return s;
}
function align(X, ref) {
  const n = X.length;
  const mean = (A) => { let mx = 0, my = 0; for (const p of A) { mx += p[0]; my += p[1]; } return [mx / n, my / n]; };
  const [rx, ry] = mean(ref);
  let best = null;
  for (const refl of [1, -1]) {
    const [mx, my] = mean(X);
    const S = X.map((p) => [(p[0] - mx) * refl, p[1] - my]);
    const T = ref.map((p) => [p[0] - rx, p[1] - ry]);
    let C = 0, Dd = 0, SS = 0;
    for (let i = 0; i < n; i++) {
      C += S[i][0] * T[i][0] + S[i][1] * T[i][1];
      Dd += S[i][0] * T[i][1] - S[i][1] * T[i][0];
      SS += S[i][0] ** 2 + S[i][1] ** 2;
    }
    const th = Math.atan2(Dd, C), sc = Math.sqrt(C * C + Dd * Dd) / Math.max(SS, 1e-12);
    const ct = Math.cos(th) * sc, st = Math.sin(th) * sc;
    const A = S.map((p) => [ct * p[0] - st * p[1] + rx, st * p[0] + ct * p[1] + ry]);
    let err = 0;
    for (let i = 0; i < n; i++) err += (A[i][0] - ref[i][0]) ** 2 + (A[i][1] - ref[i][1]) ** 2;
    if (!best || err < best.err) best = { err, A };
  }
  return best;
}
function rmsError(X, pos, radius) { return Math.sqrt(align(X, pos).err / X.length) / radius; }

/* hierarchical: farthest seeds -> patch MDS -> stitch */
function farthestSeeds(adj, k, seed) {
  const rnd = mulberry32(seed ^ 0x1234), n = adj.length;
  const seeds = [Math.floor(rnd() * n)];
  let dmin = bfs(adj, seeds[0]).map((x) => (x < 0 ? 1e9 : x));
  while (seeds.length < k) {
    let best = 0, bv = -1;
    for (let i = 0; i < n; i++) if (dmin[i] > bv && dmin[i] < 1e9) { bv = dmin[i]; best = i; }
    seeds.push(best);
    const d = bfs(adj, best);
    for (let i = 0; i < n; i++) dmin[i] = Math.min(dmin[i], d[i] < 0 ? 1e9 : d[i]);
  }
  return seeds;
}
function subHop(adj, members) {
  const idx = new Map(members.map((g, l) => [g, l]));
  const sub = members.map(() => []);
  for (const g of members) for (const v of adj[g]) if (idx.has(v)) sub[idx.get(g)].push(idx.get(v));
  return hopMatrix(sub);
}
function simFromPairs(src, dst) {
  const n = src.length;
  let scx = 0, scy = 0, dcx = 0, dcy = 0;
  for (let i = 0; i < n; i++) { scx += src[i][0]; scy += src[i][1]; dcx += dst[i][0]; dcy += dst[i][1]; }
  scx /= n; scy /= n; dcx /= n; dcy /= n;
  let best = null;
  for (const refl of [1, -1]) {
    let C = 0, Dd = 0, SS = 0;
    for (let i = 0; i < n; i++) {
      const sx = (src[i][0] - scx) * refl, sy = src[i][1] - scy;
      const tx = dst[i][0] - dcx, ty = dst[i][1] - dcy;
      C += sx * tx + sy * ty; Dd += sx * ty - sy * tx; SS += sx * sx + sy * sy;
    }
    const th = Math.atan2(Dd, C), sc = Math.sqrt(C * C + Dd * Dd) / Math.max(SS, 1e-12);
    const ct = Math.cos(th) * sc, st = Math.sin(th) * sc;
    const f = (p) => { const x = (p[0] - scx) * refl, y = p[1] - scy; return [ct * x - st * y + dcx, st * x + ct * y + dcy]; };
    let res = 0;
    for (let i = 0; i < n; i++) { const a = f(src[i]); res += (a[0] - dst[i][0]) ** 2 + (a[1] - dst[i][1]) ** 2; }
    if (!best || res < best.res) best = { res, f };
  }
  return best.f;
}
function localizeHier(adj, k, seed) {
  const n = adj.length;
  const seeds = farthestSeeds(adj, k, seed);
  const Hs = seeds.map((s) => bfs(adj, s));
  const core = new Array(n);
  for (let i = 0; i < n; i++) {
    let bc = 0, bv = Infinity;
    for (let c = 0; c < k; c++) { const h = Hs[c][i]; if (h >= 0 && h < bv) { bv = h; bc = c; } }
    core[i] = bc;
  }
  const patches = [];
  for (let c = 0; c < k; c++) {
    const m = new Set();
    for (let i = 0; i < n; i++) if (core[i] === c) { m.add(i); for (const v of adj[i]) m.add(v); }
    patches.push([...m].sort((a, b) => a - b));
  }
  const localMaps = [], patchDiam = [];
  for (const mem of patches) {
    const Hsub = subHop(adj, mem);
    if (Hsub.some((row) => row.some((x) => x < 0))) return null;
    let md = 0; for (const row of Hsub) for (const x of row) md = Math.max(md, x);
    patchDiam.push(md);
    let Xs = classicalMDS(Hsub, seed);
    for (let q = 0; q < 40; q++) Xs = smacofStep(Xs, Hsub);
    localMaps.push(new Map(mem.map((g, l) => [g, Xs[l]])));
  }
  const sets = patches.map((p) => new Set(p));
  const padj = patches.map(() => []);
  for (let i = 0; i < patches.length; i++)
    for (let j = i + 1; j < patches.length; j++) {
      let sh = 0; for (const x of patches[j]) if (sets[i].has(x)) sh++;
      if (sh >= 3) { padj[i].push(j); padj[j].push(i); }
    }
  const global = new Map(localMaps[0]);
  const placed = new Set([0]); const order = [0]; const depth = new Map([[0, 0]]); let dSeen = 0;
  for (let h = 0; h < order.length; h++) {
    const a = order[h];
    for (const b of padj[a]) {
      if (placed.has(b)) continue;
      const shared = [...localMaps[b].keys()].filter((kk) => global.has(kk));
      if (shared.length < 3) continue;
      const T = simFromPairs(shared.map((kk) => localMaps[b].get(kk)), shared.map((kk) => global.get(kk)));
      for (const [kk, xy] of localMaps[b]) if (!global.has(kk)) global.set(kk, T(xy));
      placed.add(b); order.push(b); depth.set(b, depth.get(a) + 1); dSeen = Math.max(dSeen, depth.get(b));
    }
  }
  if (global.size < n) return null;
  const X = Array.from({ length: n }, (_, i) => global.get(i));
  return { X, core, latency: { maxPatchDiameter: Math.max(...patchDiam), stitchDepth: dSeen, nClusters: k } };
}

/* ============================================================
   Palette + helpers
   ============================================================ */
const C = {
  bg: "#0e1322", panel: "#161d31", raised: "#1d2740", line: "#27314e",
  edge: "rgba(94,200,224,0.10)", cyan: "#5ec8e0", amber: "#ffae5c",
  gold: "#e6c15a", text: "#cdd8ee", mute: "#6e7b9a", good: "#6ee7a0", warn: "#ff7a6b",
};
const CLUSTER = ["#5ec8e0", "#ffae5c", "#9d8df1", "#6ee7a0", "#ff7a9c", "#e6c15a",
  "#5ad1b4", "#f29bd4", "#8fb6ff", "#d6a86a", "#b6e36b", "#ff9d6b"];

/* ============================================================
   Component
   ============================================================ */
export default function LocalizationScope() {
  const [n, setN] = useState(160);
  const [radius, setRadius] = useState(0.2);
  const [seed, setSeed] = useState(42);
  const [mode, setMode] = useState("flat");        // flat | hier
  const [clusters, setClusters] = useState(4);
  const [showEdges, setShowEdges] = useState(true);
  const [showWhiskers, setShowWhiskers] = useState(true);
  const [running, setRunning] = useState(true);
  const [metrics, setMetrics] = useState({ deg: 0, diam: 0, err: 0, iter: 0, conv: false, fragmented: false, latency: null, stitchFail: false });

  const canvasRef = useRef(null);
  const net = useRef(null);          // {pos, adj, D}
  const X = useRef(null);            // current estimate (flat: live; hier: static)
  const prevAligned = useRef(null);  // for trails
  const stressHist = useRef([]);
  const iter = useRef(0);
  const raf = useRef(0);
  const flat = useRef({ converged: false, plateau: 0 });

  /* rebuild network + initial embedding when structural params change */
  const rebuild = useCallback(() => {
    const { pos, adj } = makeNetwork(n, radius, seed);
    const D = hopMatrix(adj);
    const fragmented = D.some((r) => r.some((x) => x < 0));
    net.current = { pos, adj, D };
    iter.current = 0; stressHist.current = []; prevAligned.current = null;
    flat.current = { converged: false, plateau: 0 };

    if (fragmented) {
      X.current = null;
      setMetrics((m) => ({ ...m, deg: adj.reduce((a, b) => a + b.length, 0) / n, diam: -1, fragmented: true, stitchFail: false }));
      return;
    }
    const deg = adj.reduce((a, b) => a + b.length, 0) / n;
    const diam = diameter(adj);

    if (mode === "flat") {
      X.current = classicalMDS(D, seed);
      setMetrics({ deg, diam, err: rmsError(X.current, pos, radius), iter: 0, conv: false, fragmented: false, latency: null, stitchFail: false });
    } else {
      const out = localizeHier(adj, clusters, seed + 1000);
      if (!out) {
        X.current = null;
        setMetrics({ deg, diam, err: 0, iter: 0, conv: true, fragmented: false, latency: null, stitchFail: true });
      } else {
        X.current = out.X; net.current.core = out.core;
        setMetrics({ deg, diam, err: rmsError(out.X, pos, radius), iter: 0, conv: true, fragmented: false, latency: out.latency, stitchFail: false });
      }
    }
  }, [n, radius, seed, mode, clusters]);

  useEffect(() => { rebuild(); }, [rebuild]);

  /* drawing */
  const draw = useCallback(() => {
    const cv = canvasRef.current; if (!cv) return;
    const dpr = window.devicePixelRatio || 1;
    const W = cv.clientWidth, H = cv.clientHeight;
    if (cv.width !== W * dpr) { cv.width = W * dpr; cv.height = H * dpr; }
    const ctx = cv.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, W, H);

    // ground + vignette + grid
    ctx.fillStyle = C.bg; ctx.fillRect(0, 0, W, H);
    const g = ctx.createRadialGradient(W / 2, H / 2, H * 0.1, W / 2, H / 2, H * 0.75);
    g.addColorStop(0, "rgba(40,52,86,0.35)"); g.addColorStop(1, "rgba(10,14,28,0)");
    ctx.fillStyle = g; ctx.fillRect(0, 0, W, H);
    ctx.strokeStyle = "rgba(39,49,78,0.5)"; ctx.lineWidth = 1;
    for (let i = 1; i < 8; i++) {
      const p = (i / 8);
      ctx.beginPath(); ctx.moveTo(p * W, 0); ctx.lineTo(p * W, H); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(0, p * H); ctx.lineTo(W, p * H); ctx.stroke();
    }

    const data = net.current; if (!data) return;
    const pad = 26;
    const map = (p) => [pad + p[0] * (W - 2 * pad), pad + (1 - p[1]) * (H - 2 * pad)];

    if (metrics.fragmented || metrics.stitchFail || !X.current) {
      ctx.fillStyle = C.mute; ctx.font = "13px 'Space Mono', ui-monospace, monospace";
      ctx.textAlign = "center";
      const msg = metrics.fragmented ? "Network fragmented — raise radio range to connect it."
        : "Patches couldn't all stitch — raise range or use fewer clusters.";
      ctx.fillText(msg, W / 2, H / 2);
      ctx.textAlign = "left";
      return;
    }

    const aligned = align(X.current, data.pos).A;

    // edges
    if (showEdges) {
      ctx.strokeStyle = C.edge; ctx.lineWidth = 1;
      ctx.beginPath();
      for (let i = 0; i < data.adj.length; i++)
        for (const j of data.adj[i]) if (j > i) {
          const a = map(data.pos[i]), b = map(data.pos[j]);
          ctx.moveTo(a[0], a[1]); ctx.lineTo(b[0], b[1]);
        }
      ctx.stroke();
    }
    // whiskers true->estimate
    if (showWhiskers) {
      ctx.strokeStyle = "rgba(255,174,92,0.34)"; ctx.lineWidth = 1;
      ctx.beginPath();
      for (let i = 0; i < n; i++) {
        const t = map(data.pos[i]), e = map(aligned[i]);
        ctx.moveTo(t[0], t[1]); ctx.lineTo(e[0], e[1]);
      }
      ctx.stroke();
    }
    // motion trails (flat, running)
    if (running && mode === "flat" && prevAligned.current) {
      ctx.strokeStyle = "rgba(255,174,92,0.5)"; ctx.lineWidth = 1.5;
      ctx.beginPath();
      for (let i = 0; i < n; i++) {
        const p0 = map(prevAligned.current[i]), p1 = map(aligned[i]);
        ctx.moveTo(p0[0], p0[1]); ctx.lineTo(p1[0], p1[1]);
      }
      ctx.stroke();
    }
    // true nodes
    for (let i = 0; i < n; i++) {
      const t = map(data.pos[i]);
      ctx.fillStyle = mode === "hier" && data.core ? CLUSTER[data.core[i] % CLUSTER.length] : C.cyan;
      ctx.globalAlpha = mode === "hier" ? 0.95 : 0.9;
      ctx.beginPath(); ctx.arc(t[0], t[1], 3.1, 0, 7); ctx.fill();
    }
    ctx.globalAlpha = 1;
    // estimate nodes
    for (let i = 0; i < n; i++) {
      const e = map(aligned[i]);
      ctx.strokeStyle = C.amber; ctx.lineWidth = 1.4;
      ctx.beginPath(); ctx.moveTo(e[0] - 3, e[1]); ctx.lineTo(e[0] + 3, e[1]);
      ctx.moveTo(e[0], e[1] - 3); ctx.lineTo(e[0], e[1] + 3); ctx.stroke();
    }
    prevAligned.current = aligned;

    // stress sparkline
    const hist = stressHist.current;
    if (hist.length > 1) {
      const sw = 92, sh = 30, sx = W - sw - 14, sy = 14;
      ctx.fillStyle = "rgba(13,18,34,0.7)"; ctx.fillRect(sx - 6, sy - 4, sw + 12, sh + 18);
      ctx.strokeStyle = C.line; ctx.strokeRect(sx - 6, sy - 4, sw + 12, sh + 18);
      const mx = Math.max(...hist), mn = Math.min(...hist);
      ctx.strokeStyle = C.gold; ctx.lineWidth = 1.4; ctx.beginPath();
      hist.forEach((v, k) => {
        const px = sx + (k / (hist.length - 1)) * sw;
        const py = sy + (1 - (v - mn) / Math.max(mx - mn, 1e-9)) * sh;
        k ? ctx.lineTo(px, py) : ctx.moveTo(px, py);
      });
      ctx.stroke();
      ctx.fillStyle = C.mute; ctx.font = "9px 'Space Mono', monospace";
      ctx.fillText("stress", sx - 4, sy + sh + 11);
    }
  }, [metrics.fragmented, metrics.stitchFail, showEdges, showWhiskers, running, mode, n]);

  /* animation loop */
  useEffect(() => {
    const tick = () => {
      const data = net.current;
      if (data && X.current && mode === "flat" && running && !flat.current.converged) {
        let s0 = stress(X.current, data.D);
        for (let k = 0; k < 2; k++) X.current = smacofStep(X.current, data.D);
        iter.current += 2;
        const s1 = stress(X.current, data.D);
        const h = stressHist.current; h.push(s1); if (h.length > 90) h.shift();
        if (Math.abs(s0 - s1) < 1e-5 * Math.max(s0, 1e-9)) {
          flat.current.plateau++; if (flat.current.plateau > 6) flat.current.converged = true;
        } else flat.current.plateau = 0;
        setMetrics((m) => ({ ...m, err: rmsError(X.current, data.pos, radius), iter: iter.current, conv: flat.current.converged }));
      }
      draw();
      raf.current = requestAnimationFrame(tick);
    };
    raf.current = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf.current);
  }, [draw, running, mode, radius]);

  const restart = () => {
    const data = net.current; if (!data || metrics.fragmented) return;
    if (mode === "flat") {
      X.current = classicalMDS(data.D, seed);
      iter.current = 0; stressHist.current = []; prevAligned.current = null;
      flat.current = { converged: false, plateau: 0 };
      setRunning(true);
      setMetrics((m) => ({ ...m, err: rmsError(X.current, data.pos, radius), iter: 0, conv: false }));
    } else rebuild();
  };

  /* ---------- UI ---------- */
  const Stat = ({ label, value, accent }) => (
    <div style={{ display: "flex", flexDirection: "column", gap: 2, minWidth: 64 }}>
      <span style={{ fontFamily: "'Space Mono', monospace", fontSize: 10, letterSpacing: ".08em", color: C.mute, textTransform: "uppercase" }}>{label}</span>
      <span style={{ fontFamily: "'Space Mono', monospace", fontSize: 18, color: accent || C.text, fontWeight: 700 }}>{value}</span>
    </div>
  );

  const errColor = metrics.err < 0.15 ? C.good : metrics.err < 0.3 ? C.gold : C.warn;

  return (
    <div style={{ minHeight: "100%", background: C.bg, color: C.text, fontFamily: "system-ui, -apple-system, sans-serif", padding: "18px 16px 28px" }}>
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=Space+Mono:wght@400;700&display=swap');
        .scope-wrap{max-width:1040px;margin:0 auto}
        .scope-grid{display:flex;flex-direction:column;gap:16px}
        @media(min-width:880px){.scope-grid{flex-direction:row}.scope-canvas-col{flex:1}.scope-ctrl-col{width:300px;flex-shrink:0}}
        input[type=range]{-webkit-appearance:none;appearance:none;width:100%;height:3px;background:${C.line};border-radius:3px;outline:none}
        input[type=range]::-webkit-slider-thumb{-webkit-appearance:none;width:15px;height:15px;border-radius:50%;background:${C.gold};cursor:pointer;border:2px solid ${C.bg};box-shadow:0 0 0 1px ${C.gold}}
        input[type=range]::-moz-range-thumb{width:15px;height:15px;border-radius:50%;background:${C.gold};cursor:pointer;border:2px solid ${C.bg}}
        .seg{flex:1;padding:7px 0;text-align:center;font-size:12.5px;font-weight:600;cursor:pointer;border-radius:7px;transition:.15s;letter-spacing:.02em}
        .btn{padding:9px 12px;border-radius:8px;border:1px solid ${C.line};background:${C.raised};color:${C.text};font-size:12.5px;font-weight:600;cursor:pointer;font-family:inherit;transition:.15s}
        .btn:hover{border-color:${C.gold}}
        .btn:focus-visible,.seg:focus-visible{outline:2px solid ${C.gold};outline-offset:2px}
        .tog{display:flex;align-items:center;justify-content:space-between;font-size:12.5px;color:${C.text};cursor:pointer;padding:4px 0}
        @media(prefers-reduced-motion:reduce){*{transition:none!important}}
      `}</style>

      <div className="scope-wrap">
        {/* header */}
        <div style={{ marginBottom: 16 }}>
          <div style={{ fontFamily: "'Space Mono', monospace", fontSize: 11, letterSpacing: ".22em", color: C.cyan, textTransform: "uppercase", marginBottom: 6 }}>
            Anchor-free · GPS-free · distance-vector
          </div>
          <h1 style={{ fontFamily: "'Space Grotesk', system-ui, sans-serif", fontSize: 30, fontWeight: 700, margin: 0, lineHeight: 1.05, letterSpacing: "-.01em" }}>
            Connectivity Scope
          </h1>
          <p style={{ color: C.mute, fontSize: 13.5, margin: "8px 0 0", maxWidth: 560, lineHeight: 1.5 }}>
            A relative map recovered from hop counts alone — no GPS, no anchors. Watch the
            estimate (<span style={{ color: C.amber }}>amber crosses</span>) relax onto the true
            layout (<span style={{ color: C.cyan }}>cyan dots</span>). Whiskers are the residual error.
          </p>
        </div>

        <div className="scope-grid">
          {/* canvas + metrics */}
          <div className="scope-canvas-col">
            <div style={{ display: "flex", gap: 18, flexWrap: "wrap", padding: "12px 14px", background: C.panel, border: `1px solid ${C.line}`, borderRadius: "12px 12px 0 0", borderBottom: "none" }}>
              <Stat label="mean degree" value={metrics.deg.toFixed(1)} />
              <Stat label="hops Ø / latency" value={metrics.diam < 0 ? "—" : metrics.diam} accent={C.cyan} />
              <Stat label="RMS err ×R" value={metrics.fragmented || metrics.stitchFail ? "—" : metrics.err.toFixed(3)} accent={errColor} />
              {mode === "flat"
                ? <Stat label="iteration" value={metrics.iter} />
                : <Stat label="patch Ø + stitch" value={metrics.latency ? `${metrics.latency.maxPatchDiameter}+${metrics.latency.stitchDepth}` : "—"} accent={C.amber} />}
              {mode === "flat" && metrics.conv && <Stat label="state" value="converged" accent={C.good} />}
            </div>
            <div style={{ position: "relative", background: C.bg, border: `1px solid ${C.line}`, borderRadius: "0 0 12px 12px", overflow: "hidden" }}>
              <canvas ref={canvasRef} style={{ width: "100%", height: "min(70vh, 520px)", display: "block" }} />
            </div>
          </div>

          {/* controls */}
          <div className="scope-ctrl-col">
            <div style={{ background: C.panel, border: `1px solid ${C.line}`, borderRadius: 12, padding: 16, display: "flex", flexDirection: "column", gap: 18 }}>
              {/* mode */}
              <div>
                <div style={{ display: "flex", gap: 4, background: C.bg, padding: 4, borderRadius: 9, border: `1px solid ${C.line}` }}>
                  {["flat", "hier"].map((md) => (
                    <div key={md} className="seg" tabIndex={0} role="button"
                      onClick={() => setMode(md)}
                      onKeyDown={(e) => e.key === "Enter" && setMode(md)}
                      style={{ background: mode === md ? C.raised : "transparent", color: mode === md ? C.gold : C.mute, border: mode === md ? `1px solid ${C.line}` : "1px solid transparent" }}>
                      {md === "flat" ? "Flat MDS" : "Hierarchical"}
                    </div>
                  ))}
                </div>
                <p style={{ fontSize: 11, color: C.mute, margin: "8px 2px 0", lineHeight: 1.45 }}>
                  {mode === "flat"
                    ? "Whole-network DV-Hop → classical MDS → live SMACOF relaxation."
                    : "Cluster → per-patch MDS → stitch. Nodes are colored by cluster; latency reads as parallel patch rounds + serial stitch depth."}
                </p>
              </div>

              {/* sliders */}
              <Slider label="Nodes" value={n} min={50} max={280} step={10} onChange={setN} fmt={(v) => v} />
              <Slider label="Radio range (×side)" value={radius} min={0.12} max={0.34} step={0.01} onChange={setRadius} fmt={(v) => v.toFixed(2)} />
              {mode === "hier" && (
                <Slider label="Clusters" value={clusters} min={2} max={16} step={1} onChange={setClusters} fmt={(v) => v} hint="sweet spot ≈ patch radius near ½ network diameter" />
              )}

              {/* transport */}
              {mode === "flat" && (
                <div style={{ display: "flex", gap: 8 }}>
                  <button className="btn" style={{ flex: 1, background: running ? C.raised : C.gold, color: running ? C.text : C.bg, borderColor: running ? C.line : C.gold }} onClick={() => setRunning((r) => !r)}>
                    {running ? "Pause" : "Run"}
                  </button>
                  <button className="btn" onClick={() => { const d = net.current; if (d && X.current) { X.current = smacofStep(X.current, d.D); iter.current++; setMetrics((m) => ({ ...m, err: rmsError(X.current, d.pos, radius), iter: iter.current })); } }}>Step</button>
                  <button className="btn" onClick={restart}>Reset</button>
                </div>
              )}

              <div style={{ display: "flex", gap: 8 }}>
                <button className="btn" style={{ flex: 1 }} onClick={() => setSeed((s) => s + 1)}>New layout ↻</button>
              </div>

              {/* toggles */}
              <div style={{ borderTop: `1px solid ${C.line}`, paddingTop: 12, display: "flex", flexDirection: "column", gap: 2 }}>
                <Toggle label="Connectivity edges" on={showEdges} set={setShowEdges} />
                <Toggle label="Error whiskers" on={showWhiskers} set={setShowWhiskers} />
              </div>

              <p style={{ fontSize: 10.5, color: C.mute, lineHeight: 1.5, margin: 0, fontFamily: "'Space Mono', monospace" }}>
                Map recoverable only up to rotation / reflection / scale; error is scored after
                Procrustes alignment. Range-free, so accuracy floors at hop quantization.
              </p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );

  function Slider({ label, value, min, max, step, onChange, fmt, hint }) {
    return (
      <div>
        <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 7 }}>
          <span style={{ fontSize: 12.5, color: C.text }}>{label}</span>
          <span style={{ fontFamily: "'Space Mono', monospace", fontSize: 12.5, color: C.gold }}>{fmt(value)}</span>
        </div>
        <input type="range" min={min} max={max} step={step} value={value} onChange={(e) => onChange(Number(e.target.value))} />
        {hint && <p style={{ fontSize: 10, color: C.mute, margin: "6px 0 0", lineHeight: 1.4 }}>{hint}</p>}
      </div>
    );
  }
  function Toggle({ label, on, set }) {
    return (
      <div className="tog" tabIndex={0} role="switch" aria-checked={on}
        onClick={() => set((v) => !v)} onKeyDown={(e) => e.key === "Enter" && set((v) => !v)}>
        <span>{label}</span>
        <span style={{ width: 34, height: 19, borderRadius: 19, background: on ? C.gold : C.line, position: "relative", transition: ".15s", flexShrink: 0 }}>
          <span style={{ position: "absolute", top: 2, left: on ? 17 : 2, width: 15, height: 15, borderRadius: "50%", background: on ? C.bg : C.mute, transition: ".15s" }} />
        </span>
      </div>
    );
  }
}
