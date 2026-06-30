import React, { useState, useEffect, useRef, useMemo } from "react";

// ---------------------------------------------------------------------------
// "Apple in space" — an interactive explainer for how LLMs turn words into
// vectors, mix them with attention, stack them through transformer blocks, and
// generate text one token at a time.
//
// This refactor (a) corrects several technical inaccuracies and (b) fleshes out
// the "go deeper" threads with new interactive slides: positional encoding
// (RoPE), attention (QKV + heatmap + multi-head), the transformer block /
// residual stream, the training pipeline (pretrain -> SFT -> RLHF/DPO), and the
// context window / KV cache.
//
// Framing: a modern decoder-only LLM — rotary positions, gated MLPs, RMSNorm,
// causal attention — kept generic, without naming specific models.
// ---------------------------------------------------------------------------

const PALETTE = {
  ink: "#1a1714",
  paper: "#f3ede1",
  panel: "#fbf7ee",
  line: "#cdbfa6",
  apple: "#c0392b",     // ambiguous red
  tech: "#2c6e8f",      // cool slate-blue
  fruit: "#5a8a3c",     // orchard green
  dim: "#8a7d68",
  glow: "#e0b341",
  violet: "#7d5ba6",    // new: used for attention / Q,K,V accents
};

// Anchor positions in abstract 3D space (-1..1 on each axis)
const ANCHORS = {
  apple:    { x: 0.0,  y: 0.05, z: 0.0,  label: "apple",    color: PALETTE.apple },
  computer: { x: 0.82, y: 0.55, z: 0.35, label: "computer", color: PALETTE.tech },
  fruit:    { x: -0.8, y: -0.5, z: -0.3, label: "fruit",    color: PALETTE.fruit },
};

const PROMPTS = [
  {
    id: "chip",
    text: "How much faster is the new Apple chip?",
    triggers: ["chip", "faster"],
    target: "computer",
    pull: 0.72,
    bridge: ["chip", "processor", "silicon"],
    note: "“chip” and “faster” sit deep in the tech region. Attention lets Apple attend to them, pulling its vector toward computer.",
  },
  {
    id: "orchard",
    text: "Is that apple ripe enough to pick from the tree?",
    triggers: ["ripe", "pick", "tree"],
    target: "fruit",
    pull: 0.74,
    bridge: ["ripe", "tree", "orchard"],
    note: "“ripe”, “pick”, and “tree” are food/nature words. Apple attends to them and slides toward fruit.",
  },
  {
    id: "screen",
    text: "Does the Apple laptop screen support that software update?",
    triggers: ["laptop", "screen", "software"],
    target: "computer",
    pull: 0.85,
    bridge: ["laptop", "screen", "software"],
    note: "Three strong tech tokens dominate the blend. Apple lands almost on top of computer.",
  },
  {
    id: "pie",
    text: "Can I bake this apple into a pie for dessert?",
    triggers: ["bake", "pie", "dessert"],
    target: "fruit",
    pull: 0.86,
    bridge: ["bake", "pie", "sweet"],
    note: "Cooking words dominate. Apple resolves firmly to the edible meaning.",
  },
];

// ---------------------------------------------------------------------------
// Tiny 3D -> 2D projection (no libraries). Rotate around Y + tilt around X.
// ---------------------------------------------------------------------------
function project(p, rotY, rotX, w, h) {
  let x = p.x * Math.cos(rotY) - p.z * Math.sin(rotY);
  let z = p.x * Math.sin(rotY) + p.z * Math.cos(rotY);
  let y = p.y;
  let y2 = y * Math.cos(rotX) - z * Math.sin(rotX);
  let z2 = y * Math.sin(rotX) + z * Math.cos(rotX);
  y = y2; z = z2;
  const persp = 2.6 / (2.6 + z);
  const cx = w / 2;
  const cy = h / 2;
  const scale = Math.min(w, h) * 0.36;
  return { sx: cx + x * scale * persp, sy: cy - y * scale * persp, depth: z, persp };
}

function lerp(a, b, t) {
  return { x: a.x + (b.x - a.x) * t, y: a.y + (b.y - a.y) * t, z: a.z + (b.z - a.z) * t };
}

function cosineSim(a, b) {
  const dot = a.x * b.x + a.y * b.y + a.z * b.z;
  const ma = Math.hypot(a.x, a.y, a.z);
  const mb = Math.hypot(b.x, b.y, b.z);
  if (ma === 0 || mb === 0) return 0;
  return dot / (ma * mb);
}

function navBtn(disabled) {
  return {
    background: disabled ? "transparent" : PALETTE.ink,
    color: disabled ? PALETTE.line : PALETTE.paper,
    border: `1px solid ${disabled ? PALETTE.line : PALETTE.ink}`,
    borderRadius: 3, padding: "10px 20px", fontSize: 14, fontWeight: 600,
    cursor: disabled ? "default" : "pointer", fontFamily: "inherit",
  };
}

// small reusable label tag used across slides
function Tag({ children }) {
  return (
    <span style={{
      fontFamily: "monospace", fontSize: 10, color: PALETTE.dim, whiteSpace: "nowrap",
      border: `1px dashed ${PALETTE.line}`, borderRadius: 10, padding: "1px 7px",
    }}>{children}</span>
  );
}

// ---------------------------------------------------------------------------
// GLOSSARY — definitions for the hoverable <Term> component. Keep them short and
// plain; these are the "what does that word mean" cards.
// ---------------------------------------------------------------------------
const GLOSSARY = {
  "token": "A chunk of text the model treats as one unit — often a word, a word-piece, or even a single byte.",
  "embedding": "The vector (list of numbers) a token is turned into — its coordinates in the model's space.",
  "vector": "An ordered list of numbers. Here, it's the coordinates of a point in the model's space.",
  "dot product": "Multiply two vectors element-by-element and add it all up. A bigger result means the two vectors point in more similar directions.",
  "cosine similarity": "The angle between two vectors, ignoring their length. 1 = same direction, 0 = unrelated, −1 = opposite.",
  "softmax": "A function that turns a list of raw scores into percentages that add up to 100%. Bigger scores get exponentially more share.",
  "logit": "A raw, unnormalised score for one token — what comes out before softmax turns the scores into probabilities.",
  "attention": "The step where each token looks at the other tokens and pulls in a blend of their information, weighted by relevance.",
  "query": "What a token is looking for. Compared against every other token's key to decide where to attend.",
  "key": "What a token offers to others — matched against incoming queries.",
  "value": "The information a token actually contributes once it's been attended to.",
  "causal mask": "A rule that stops a token from attending to tokens that come after it — so generation only ever looks backwards.",
  "residual stream": "The running vector that flows through the whole network. Each layer reads it and adds its result back, rather than overwriting.",
  "MLP": "Multi-layer perceptron — a small feed-forward network applied to each token on its own. Holds most of the model's stored knowledge.",
  "RoPE": "Rotary position embedding — encodes a token's position by rotating its query and key vectors by an angle that grows with position.",
  "BPE": "Byte-pair encoding — the algorithm that learns the vocabulary by repeatedly merging the most common adjacent pair of symbols.",
  "autoregressive": "Generating one token at a time, where each new token is fed back in as input for predicting the next.",
  "cross-entropy": "The training loss that measures how surprised the model was by the correct next token. Lower is better.",
  "SFT": "Supervised fine-tuning — further training on curated example (instruction, good-answer) pairs.",
  "RLHF": "Reinforcement learning from human feedback — train a reward model on human rankings, then optimise the model toward higher-rated answers.",
  "DPO": "Direct preference optimisation — a simpler alternative to RLHF that learns straight from preferred-vs-rejected answer pairs, with no separate reward model.",
  "PPO": "Proximal policy optimisation — the reinforcement-learning algorithm commonly used to update the model during RLHF.",
  "KV cache": "Stored keys and values from earlier tokens, kept so each new generation step doesn't have to recompute them.",
  "RMSNorm": "A lightweight normalisation step that keeps each layer's input at a stable scale, which makes deep stacks train reliably.",
  "pre-norm": "Normalising a layer's input before the attention/MLP step (rather than after). Standard in modern deep transformers for stability.",
};

// Hoverable / tappable definition card. Works on desktop (hover) and touch (tap).
function Term({ k, children }) {
  const [open, setOpen] = useState(false);
  const def = GLOSSARY[k || (typeof children === "string" ? children : "")];
  if (!def) return <span>{children}</span>;
  return (
    <span
      style={{ position: "relative", display: "inline-block" }}
      onMouseEnter={() => setOpen(true)}
      onMouseLeave={() => setOpen(false)}
    >
      <span
        role="button" tabIndex={0}
        onClick={(e) => { e.stopPropagation(); setOpen((o) => !o); }}
        onKeyDown={(e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); setOpen((o) => !o); } }}
        style={{
          borderBottom: `1.5px dotted ${PALETTE.violet}`, cursor: "help",
          color: "inherit", fontWeight: "inherit",
        }}
      >{children}</span>
      {open && (
        <span style={{
          position: "absolute", bottom: "calc(100% + 8px)", left: "50%", transform: "translateX(-50%)",
          width: 244, maxWidth: "78vw", zIndex: 50,
          background: PALETTE.ink, color: PALETTE.paper, borderRadius: 6, padding: "9px 11px",
          fontFamily: "'Helvetica Neue', Arial, sans-serif", fontSize: 12.5, lineHeight: 1.45,
          fontWeight: 400, textAlign: "left", boxShadow: "0 6px 20px rgba(0,0,0,0.25)",
          whiteSpace: "normal",
        }}>
          <span style={{ fontFamily: "monospace", fontSize: 10, color: PALETTE.glow, display: "block", marginBottom: 3, textTransform: "uppercase", letterSpacing: 0.5 }}>{k || children}</span>
          {def}
          <span style={{
            position: "absolute", top: "100%", left: "50%", transform: "translateX(-50%)",
            width: 0, height: 0, borderLeft: "6px solid transparent", borderRight: "6px solid transparent",
            borderTop: `6px solid ${PALETTE.ink}`,
          }} />
        </span>
      )}
    </span>
  );
}

// ===========================================================================
// SCENE — the three-word embedding space (apple / computer / fruit)
// ===========================================================================
function Scene({ activePrompt, autoRotate }) {
  const W = 560, H = 460;
  const [rotY, setRotY] = useState(0.5);
  const [t, setT] = useState(0);
  const dragRef = useRef(null);
  const rafRef = useRef(null);

  const targetAnchor = activePrompt ? ANCHORS[activePrompt.target] : null;
  const pull = activePrompt ? activePrompt.pull : 0;

  useEffect(() => {
    let frame;
    const goal = activePrompt ? 1 : 0;
    const step = () => {
      setT((cur) => {
        const next = cur + (goal - cur) * 0.08;
        if (Math.abs(goal - next) > 0.001) frame = requestAnimationFrame(step);
        return next;
      });
    };
    frame = requestAnimationFrame(step);
    return () => cancelAnimationFrame(frame);
  }, [activePrompt]);

  useEffect(() => {
    if (!autoRotate) return;
    const step = () => { setRotY((r) => r + 0.0035); rafRef.current = requestAnimationFrame(step); };
    rafRef.current = requestAnimationFrame(step);
    return () => cancelAnimationFrame(rafRef.current);
  }, [autoRotate]);

  const rotX = -0.32;

  const applePos = useMemo(() => {
    if (!targetAnchor) return ANCHORS.apple;
    return lerp(ANCHORS.apple, targetAnchor, pull * t);
  }, [targetAnchor, pull, t]);

  const onDown = (e) => { const pt = e.touches ? e.touches[0] : e; dragRef.current = { x: pt.clientX, startRot: rotY }; };
  const onMove = (e) => { if (!dragRef.current) return; const pt = e.touches ? e.touches[0] : e; setRotY(dragRef.current.startRot + (pt.clientX - dragRef.current.x) * 0.008); };
  const onUp = () => { dragRef.current = null; };

  const pApple = project(applePos, rotY, rotX, W, H);
  const pComputer = project(ANCHORS.computer, rotY, rotX, W, H);
  const pFruit = project(ANCHORS.fruit, rotY, rotX, W, H);

  const bridgePoints = useMemo(() => {
    if (!activePrompt) return [];
    const tgt = ANCHORS[activePrompt.target];
    return activePrompt.bridge.map((word, i) => {
      const frac = (i + 1) / (activePrompt.bridge.length + 1);
      const base = lerp(ANCHORS.apple, tgt, frac);
      const jitter = (i % 2 === 0 ? 1 : -1) * 0.12;
      const pos = { x: base.x, y: base.y + jitter, z: base.z + jitter * 0.5 };
      return { word, pr: project(pos, rotY, rotX, W, H) };
    });
  }, [activePrompt, rotY]);

  const simTech = cosineSim(applePos, ANCHORS.computer);
  const simFruit = cosineSim(applePos, ANCHORS.fruit);

  const axisPts = {
    o: project({ x: 0, y: 0, z: 0 }, rotY, rotX, W, H),
    x: project({ x: 1.1, y: 0, z: 0 }, rotY, rotX, W, H),
    y: project({ x: 0, y: 1.0, z: 0 }, rotY, rotX, W, H),
    z: project({ x: 0, y: 0, z: 1.1 }, rotY, rotX, W, H),
  };

  const Node = ({ p, color, label, big }) => {
    const r = (big ? 11 : 8) * p.persp;
    return (
      <g>
        <circle cx={p.sx} cy={p.sy} r={r + 7} fill={color} opacity={0.14} />
        <circle cx={p.sx} cy={p.sy} r={r} fill={color} />
        <circle cx={p.sx} cy={p.sy} r={r} fill="none" stroke="#fff" strokeOpacity="0.5" strokeWidth="1.5" />
        <text x={p.sx} y={p.sy - r - 8} textAnchor="middle" fontFamily="'Georgia', serif"
          fontSize={big ? 18 : 14} fontWeight={big ? 700 : 600} fill={PALETTE.ink}>{label}</text>
      </g>
    );
  };

  const nodeList = [
    { key: "computer", p: pComputer, color: PALETTE.tech, label: "computer" },
    { key: "fruit", p: pFruit, color: PALETTE.fruit, label: "fruit" },
    { key: "apple", p: pApple, color: PALETTE.apple, label: "apple", big: true },
  ].sort((a, b) => b.p.depth - a.p.depth);

  return (
    <div style={{ userSelect: "none" }}>
      <svg width="100%" viewBox={`0 0 ${W} ${H}`}
        style={{ touchAction: "none", cursor: dragRef.current ? "grabbing" : "grab", display: "block" }}
        onMouseDown={onDown} onMouseMove={onMove} onMouseUp={onUp} onMouseLeave={onUp}
        onTouchStart={onDown} onTouchMove={onMove} onTouchEnd={onUp}>
        {["x", "y", "z"].map((ax) => (
          <line key={ax} x1={axisPts.o.sx} y1={axisPts.o.sy} x2={axisPts[ax].sx} y2={axisPts[ax].sy}
            stroke={PALETTE.line} strokeWidth="1" strokeDasharray="3 4" opacity="0.6" />
        ))}
        {["x", "y", "z"].map((ax) => (
          <text key={ax + "l"} x={axisPts[ax].sx} y={axisPts[ax].sy} fontFamily="monospace" fontSize="11" fill={PALETTE.dim} opacity="0.7">{ax}</text>
        ))}
        {activePrompt && (
          <line x1={pApple.sx} y1={pApple.sy}
            x2={(activePrompt.target === "computer" ? pComputer : pFruit).sx}
            y2={(activePrompt.target === "computer" ? pComputer : pFruit).sy}
            stroke={ANCHORS[activePrompt.target].color} strokeWidth="2" strokeDasharray="5 5" opacity={0.5 * t} />
        )}
        {bridgePoints.map((b, i) => (
          <g key={i} opacity={t}>
            <circle cx={b.pr.sx} cy={b.pr.sy} r={4} fill={PALETTE.glow} />
            <text x={b.pr.sx + 7} y={b.pr.sy + 4} fontFamily="monospace" fontSize="12" fill={PALETTE.ink} fontWeight="600">{b.word}</text>
          </g>
        ))}
        {nodeList.map((n) => (<Node key={n.key} p={n.p} color={n.color} label={n.label} big={n.big} />))}
      </svg>
      <div style={{ display: "flex", gap: 12, marginTop: 4 }}>
        <SimBar label="cos(apple, computer)" value={simTech} color={PALETTE.tech} />
        <SimBar label="cos(apple, fruit)" value={simFruit} color={PALETTE.fruit} />
      </div>
    </div>
  );
}

function SimBar({ label, value, color }) {
  const pct = Math.max(0, Math.min(1, (value + 1) / 2)) * 100;
  return (
    <div style={{ flex: 1 }}>
      <div style={{ display: "flex", justifyContent: "space-between", fontFamily: "monospace", fontSize: 11, color: PALETTE.dim, marginBottom: 3 }}>
        <span>{label}</span>
        <span style={{ color, fontWeight: 700 }}>{value.toFixed(2)}</span>
      </div>
      <div style={{ height: 6, background: "#e6dcc8", borderRadius: 3, overflow: "hidden" }}>
        <div style={{ width: `${pct}%`, height: "100%", background: color, transition: "width 0.1s linear" }} />
      </div>
    </div>
  );
}

// ===========================================================================
// TOKENISER — subword tokens, with an accurate UTF-8 byte ladder for the emoji.
// Byte-level BPE operates on raw bytes, so a 4-byte emoji starts as 4 byte
// tokens and may merge into fewer. A leading space carries a marker and is a
// different token id from the same word without one.
// ===========================================================================
const TOKEN_EXAMPLES = [
  { word: "apple",        tokens: ["apple"],                  note: "common word, one token" },
  { word: " apple",       tokens: ["▁apple"],                 note: "leading space (▁) makes a different id" },
  { word: "apples",       tokens: ["app", "les"],             note: "rarer plural splits in two" },
  { word: "tokenisation", tokens: ["token", "isation"],       note: "long word breaks into pieces" },
  { word: "🍎",           tokens: ["\\xF0", "\\x9F", "\\x8D\\x8E"], note: "4 UTF-8 bytes become byte tokens", isEmoji: true },
];

function Tokeniser() {
  const [revealed, setRevealed] = useState(false);
  const tokenColors = [PALETTE.tech, PALETTE.fruit, PALETTE.apple, "#8a6d3b"];
  const totalTokens = TOKEN_EXAMPLES.reduce((a, e) => a + e.tokens.length, 0);

  return (
    <div style={{ userSelect: "none" }}>
      <div style={{ display: "flex", flexDirection: "column", gap: 9 }}>
        {TOKEN_EXAMPLES.map((ex, i) => (
          <div key={i} style={{
            display: "flex", alignItems: "center", gap: 10,
            background: "#fff", border: `1px solid ${PALETTE.line}`, borderRadius: 4, padding: "9px 12px",
          }}>
            <span style={{ width: 96, flexShrink: 0, fontFamily: "'Georgia', serif", fontSize: 16, whiteSpace: "pre" }}>{ex.word}</span>
            <span style={{ fontFamily: "monospace", fontSize: 14, color: PALETTE.dim, flexShrink: 0 }}>→</span>
            <span style={{ display: "flex", gap: 5, flexWrap: "wrap", width: 128, flexShrink: 0 }}>
              {revealed
                ? ex.tokens.map((tk, ti) => (
                    <span key={ti} style={{
                      fontFamily: "monospace", fontSize: 12, padding: "3px 7px", borderRadius: 4,
                      background: ex.isEmoji ? PALETTE.dim : tokenColors[ti % tokenColors.length],
                      color: "#fff", fontWeight: 600,
                    }}>{tk}</span>
                  ))
                : <span style={{ fontFamily: "monospace", fontSize: 13, color: PALETTE.line }}>▢ ▢ ▢</span>}
            </span>
            {revealed && (
              <span style={{ flex: 1, minWidth: 0, fontFamily: "monospace", fontSize: 10.5, color: PALETTE.dim, whiteSpace: "normal", wordBreak: "break-word", textAlign: "right", lineHeight: 1.3 }}>
                {ex.note}
              </span>
            )}
          </div>
        ))}
      </div>

      <div style={{ display: "flex", alignItems: "center", gap: 14, marginTop: 14 }}>
        <button onClick={() => setRevealed((r) => !r)} style={{
          background: PALETTE.ink, color: PALETTE.paper, border: "none", borderRadius: 3,
          padding: "10px 18px", fontSize: 14, fontWeight: 700, cursor: "pointer", fontFamily: "inherit",
        }}>
          {revealed ? "↺ Hide tokens" : "✂ Tokenise"}
        </button>
        <span style={{ fontFamily: "monospace", fontSize: 12, color: PALETTE.dim }}>
          {revealed
            ? <>{TOKEN_EXAMPLES.length} inputs → <b style={{ color: PALETTE.apple }}>{totalTokens} tokens</b> · tokens become vectors</>
            : "click to split these into the model's actual units"}
        </span>
      </div>
    </div>
  );
}

// ===========================================================================
// ROPE — positional encoding shown on a real sentence. Move the chosen word to
// a new position; the links to the other words re-weight because RoPE makes the
// query·key score depend on the GAP between positions (and it decays with
// distance). The little dials show the rotation that drives it. Accurate: RoPE
// rotates q and k by an angle ∝ position, so their dot product sees only the
// relative offset, and far-apart tokens attend less.
// ===========================================================================
const ROPE_WORDS = ["the", "ripe", "apple", "fell", "from", "the", "tree"];

function RoPE() {
  const W = 540;
  const [queryIdx, setQueryIdx] = useState(2); // "apple"
  const theta = 0.55; // rotation per position step, one 2D subspace (illustrative)

  const n = ROPE_WORDS.length;
  const chipW = (W - 40) / n;
  const rowY = 70;
  const chipY = 96;

  // RoPE attention strength between query position i and key position j:
  // driven by the relative angle (i-j)*theta; cos() gives the natural distance
  // decay. Clamp to a positive weight for display.
  const strength = (i, j) => {
    if (i === j) return 1;
    const rel = Math.abs(i - j) * theta;
    return Math.max(0.06, (Math.cos(rel) + 1) / 2); // 1 at gap 0, decays with distance
  };

  // dial helper
  const Dial = (cx, cy, ang, color, R = 34) => {
    const x = cx + R * Math.cos(-ang), y = cy + R * Math.sin(-ang);
    return (
      <g>
        <circle cx={cx} cy={cy} r={R} fill="none" stroke={PALETTE.line} strokeWidth="1.2" strokeDasharray="3 4" />
        <line x1={cx} y1={cy} x2={x} y2={y} stroke={color} strokeWidth="2.5" />
        <circle cx={x} cy={y} r={4} fill={color} />
      </g>
    );
  };

  // pick a neighbour to spotlight in the dials (nearest other word)
  const neighbour = queryIdx === n - 1 ? queryIdx - 1 : queryIdx + 1;
  const angQ = queryIdx * theta;
  const angK = neighbour * theta;
  const gapDeg = Math.round(Math.abs(queryIdx - neighbour) * theta * 180 / Math.PI);

  const chipCx = (idx) => 20 + idx * chipW + chipW / 2;

  return (
    <div style={{ userSelect: "none" }}>
      <svg width="100%" viewBox={`0 0 ${W} 150`} style={{ display: "block" }}>
        {/* attention links from the query word to every other word */}
        {ROPE_WORDS.map((_, j) => {
          if (j === queryIdx) return null;
          const s = strength(queryIdx, j);
          const x1 = chipCx(queryIdx), x2 = chipCx(j);
          const midX = (x1 + x2) / 2;
          const lift = 22 + Math.abs(queryIdx - j) * 9;
          return (
            <path key={j} d={`M ${x1} ${rowY} Q ${midX} ${rowY - lift} ${x2} ${rowY}`}
              fill="none" stroke={PALETTE.violet} strokeWidth={0.6 + s * 5} opacity={0.12 + s * 0.6} />
          );
        })}
        {/* word chips */}
        {ROPE_WORDS.map((w, idx) => {
          const isQ = idx === queryIdx;
          const s = idx === queryIdx ? 1 : strength(queryIdx, idx);
          return (
            <g key={idx}>
              <circle cx={chipCx(idx)} cy={rowY} r={5} fill={isQ ? PALETTE.apple : PALETTE.violet} opacity={isQ ? 1 : 0.25 + s * 0.7} />
              <text x={chipCx(idx)} y={chipY} textAnchor="middle" fontFamily="'Georgia',serif"
                fontSize={isQ ? 16 : 14} fontWeight={isQ ? 700 : 400}
                fill={isQ ? PALETTE.apple : PALETTE.ink} opacity={isQ ? 1 : 0.4 + s * 0.6}>{w}</text>
              <text x={chipCx(idx)} y={chipY + 16} textAnchor="middle" fontFamily="monospace" fontSize="9" fill={PALETTE.dim}>pos {idx}</text>
            </g>
          );
        })}
      </svg>

      {/* dials showing the rotation that drives the strength */}
      <div style={{ display: "flex", alignItems: "center", justifyContent: "center", gap: 18, margin: "4px 0 12px" }}>
        <svg width="220" height="92" viewBox="0 0 220 92">
          {Dial(45, 46, angQ, PALETTE.apple)}
          <text x={45} y={88} textAnchor="middle" fontFamily="monospace" fontSize="9" fill={PALETTE.apple} fontWeight={700}>query · pos {queryIdx}</text>
          {Dial(175, 46, angK, PALETTE.tech)}
          <text x={175} y={88} textAnchor="middle" fontFamily="monospace" fontSize="9" fill={PALETTE.tech} fontWeight={700}>key · pos {neighbour}</text>
        </svg>
        <div style={{ fontFamily: "monospace", fontSize: 12, color: PALETTE.ink, maxWidth: 180, lineHeight: 1.4 }}>
          relative angle <b style={{ color: PALETTE.violet }}>{gapDeg}°</b><br />
          <span style={{ color: PALETTE.dim, fontSize: 11 }}>set by the <b>gap</b> in positions, not where the words sit</span>
        </div>
      </div>

      <div style={{ display: "flex", justifyContent: "space-between", fontFamily: "monospace", fontSize: 11, color: PALETTE.dim, marginBottom: 4 }}>
        <span style={{ color: PALETTE.apple }}>move “{ROPE_WORDS[queryIdx]}” to position</span>
        <span style={{ color: PALETTE.ink, fontWeight: 700 }}>{queryIdx}</span>
      </div>
      <input type="range" min="0" max={n - 1} step="1" value={queryIdx}
        onChange={(e) => setQueryIdx(parseInt(e.target.value))} style={sliderStyle(PALETTE.apple)} />

      <div style={{ fontFamily: "monospace", fontSize: 11, color: PALETTE.dim, textAlign: "center", marginTop: 12, lineHeight: 1.5 }}>
        thicker link = stronger attention. Move the word and the links re-weight —<br />nearby words connect strongly, distant ones fade. That's position, encoded as rotation.
      </div>
    </div>
  );
}

// ===========================================================================
// ATTENTION — the heart of the "go deeper" build-out.
// Shows: (1) one token projecting into Q, K, V; (2) the causal attention-weight
// matrix as a heatmap (upper triangle masked out); (3) a chosen query "shopping"
// across keys and pulling a weighted blend of values. Toggle a sentence to see
// how "apple" attends to tech vs fruit context.
// ===========================================================================
const ATTN_SENTENCES = {
  tech:  { tokens: ["the", "Apple", "laptop", "screen", "is", "fast"], focus: 1, target: "computer",
           // attention weights FROM "Apple" (row) over all tokens (cols), causal
           weightsFrom: [0.18, 0.30, 0.34, 0.10, 0.05, 0.03] },
  fruit: { tokens: ["the", "ripe", "apple", "fell", "from", "tree"], focus: 2, target: "fruit",
           weightsFrom: [0.12, 0.40, 0.28, 0.12, 0.05, 0.03] },
};

function Attention() {
  const [variant, setVariant] = useState("tech");
  const s = ATTN_SENTENCES[variant];
  const n = s.tokens.length;
  const accent = variant === "tech" ? PALETTE.tech : PALETTE.fruit;

  // build a full causal weight matrix: each row i attends to cols 0..i, softmax-like
  const matrix = useMemo(() => {
    const m = [];
    for (let i = 0; i < n; i++) {
      const row = [];
      // base affinity: nearby + the focus token get more; mask future (>i) to 0
      let raw = [];
      for (let j = 0; j < n; j++) {
        if (j > i) { raw.push(0); continue; }
        const recency = 1 / (1 + (i - j) * 0.6);
        const focusBoost = j === s.focus ? 1.6 : 1;
        raw.push(recency * focusBoost);
      }
      const sum = raw.reduce((a, b) => a + b, 0) || 1;
      for (let j = 0; j < n; j++) row.push(raw[j] / sum);
      m.push(row);
    }
    return m;
  }, [variant, n, s.focus]);

  const cell = 38, pad = 64;
  const gridW = pad + n * cell + 14;
  const gridH = 30 + n * cell;

  return (
    <div style={{ userSelect: "none" }}>
      {/* QKV projection strip — one word becomes three role-vectors */}
      <div style={{ fontFamily: "monospace", fontSize: 10, color: PALETTE.dim, textAlign: "center", marginBottom: 8 }}>
        every word turns itself into three vectors:
      </div>
      <div style={{ display: "flex", alignItems: "stretch", gap: 8, marginBottom: 8, justifyContent: "center" }}>
        <div style={{
          display: "flex", alignItems: "center",
          fontFamily: "'Georgia',serif", fontSize: 15, fontWeight: 700, color: PALETTE.apple,
          background: "#fff", border: `1px solid ${PALETTE.line}`, borderRadius: 4, padding: "0 12px",
        }}>“apple”</div>
        <span style={{ display: "flex", alignItems: "center", fontFamily: "monospace", fontSize: 20, color: PALETTE.dim }}>→</span>
        <div style={{ display: "flex", gap: 6 }}>
          {[["Q", "query", "what it's looking for"], ["K", "key", "what it offers others"], ["V", "value", "what it passes on"]].map(([k, role, lab]) => (
            <div key={k} style={{
              textAlign: "center", background: PALETTE.violet, borderRadius: 4, padding: "6px 8px", width: 84,
            }}>
              <div style={{ fontFamily: "'Georgia',serif", fontSize: 15, fontWeight: 700, color: "#fff" }}>{k} · {role}</div>
              <div style={{ fontFamily: "monospace", fontSize: 8.5, color: "rgba(255,255,255,0.85)", marginTop: 3, lineHeight: 1.25 }}>{lab}</div>
            </div>
          ))}
        </div>
      </div>
      <div style={{ fontFamily: "monospace", fontSize: 10, color: PALETTE.dim, textAlign: "center", marginBottom: 12 }}>
        a word with a matching <b style={{ color: PALETTE.violet }}>key</b> gets a big share of attention from your <b style={{ color: PALETTE.violet }}>query</b>
      </div>

      {/* heatmap */}
      <svg width="100%" viewBox={`0 0 ${gridW} ${gridH}`} style={{ display: "block" }}>
        {/* column labels (keys) */}
        {s.tokens.map((tk, j) => (
          <text key={"c" + j} x={pad + j * cell + cell / 2} y={20} textAnchor="middle"
            fontFamily="monospace" fontSize="10" fill={PALETTE.dim} transform={`rotate(-18 ${pad + j * cell + cell / 2} 20)`}>{tk}</text>
        ))}
        {matrix.map((row, i) => (
          <g key={i}>
            <text x={pad - 8} y={30 + i * cell + cell / 2 + 4} textAnchor="end"
              fontFamily="monospace" fontSize="10" fill={i === s.focus ? PALETTE.apple : PALETTE.ink}
              fontWeight={i === s.focus ? 700 : 400}>{s.tokens[i]}</text>
            {row.map((w, j) => {
              const masked = j > i;
              return (
                <g key={j}>
                  <rect x={pad + j * cell} y={30 + i * cell} width={cell - 3} height={cell - 3} rx={2}
                    fill={masked ? "#e9e0cd" : accent} opacity={masked ? 0.4 : 0.12 + w * 1.4}
                    stroke={i === s.focus && !masked ? PALETTE.apple : "none"} strokeWidth={i === s.focus ? 1.5 : 0} />
                  {masked
                    ? <text x={pad + j * cell + (cell - 3) / 2} y={30 + i * cell + (cell - 3) / 2 + 3} textAnchor="middle" fontFamily="monospace" fontSize="11" fill={PALETTE.line}>×</text>
                    : w > 0.05 && <text x={pad + j * cell + (cell - 3) / 2} y={30 + i * cell + (cell - 3) / 2 + 3} textAnchor="middle" fontFamily="monospace" fontSize="9" fill={w > 0.4 ? "#fff" : PALETTE.ink}>{(w * 100).toFixed(0)}</text>}
                </g>
              );
            })}
          </g>
        ))}
      </svg>

      <div style={{ display: "flex", gap: 18, fontFamily: "monospace", fontSize: 10, color: PALETTE.dim, justifyContent: "center", marginTop: 4 }}>
        <span>rows = queries · columns = keys · cells = attention weight %</span>
      </div>
      <div style={{ fontFamily: "monospace", fontSize: 10.5, color: PALETTE.dim, textAlign: "center", marginTop: 6 }}>
        the × cells are <b>causally masked</b> — a token can't attend to the future
      </div>

      {/* sentence toggle */}
      <div style={{ display: "flex", gap: 10, marginTop: 14 }}>
        {Object.keys(ATTN_SENTENCES).map((k) => {
          const on = variant === k;
          const c = k === "tech" ? PALETTE.tech : PALETTE.fruit;
          return (
            <button key={k} onClick={() => setVariant(k)} style={{
              flex: 1, background: on ? c : "transparent", color: on ? "#fff" : PALETTE.ink,
              border: `1px solid ${on ? c : PALETTE.line}`, borderRadius: 3, padding: "9px 12px",
              fontSize: 13, fontWeight: 600, cursor: "pointer", fontFamily: "inherit",
            }}>
              {k === "tech" ? "“the Apple laptop screen…”" : "“the ripe apple fell…”"}
            </button>
          );
        })}
      </div>
      <div style={{ fontFamily: "monospace", fontSize: 11, color: PALETTE.dim, textAlign: "center", marginTop: 8 }}>
        <b style={{ color: PALETTE.apple }}>apple</b>'s row shows where its meaning comes from — the heavy weights land on the {variant === "tech" ? "tech" : "orchard"} words
      </div>
    </div>
  );
}

// ===========================================================================
// TRANSFORMER BLOCK — a slow, captioned walkthrough of the residual stream.
// Instead of one fast pulse, it steps through each sublayer with a plain-English
// caption explaining WHAT happens and WHY, and why blocks repeat.
// ===========================================================================
const BLOCK_STEPS = [
  { layer: 0, part: "attn-read",  title: "Read a copy", why: "The block takes a normalised copy of the running vector — it never disturbs the original on the stream." },
  { layer: 0, part: "attn-do",    title: "Attention mixes", why: "Attention lets this word pull in information from the other words. This is where context arrives." },
  { layer: 0, part: "attn-add",   title: "Add it back", why: "The result is added onto the stream, not pasted over it. The stream now carries the word plus a little context." },
  { layer: 0, part: "mlp-read",   title: "Read again", why: "A second normalised copy is taken for the next sublayer." },
  { layer: 0, part: "mlp-do",     title: "MLP refines", why: "The feed-forward network works on this one word alone — it's where most stored knowledge gets applied." },
  { layer: 0, part: "mlp-add",    title: "Add it back", why: "Added on again. One block is done: the vector is a bit richer than it started." },
  { layer: 1, part: "repeat",     title: "Repeat the block", why: "An identical block runs on the now-improved vector. Each pass refines further — early blocks catch surface patterns, later ones catch meaning." },
];

function TransformerBlock() {
  const W = 360, H = 470;
  const [step, setStep] = useState(-1); // -1 idle
  const [playing, setPlaying] = useState(false);
  const N_BLOCKS = 4;

  useEffect(() => {
    if (!playing) return;
    if (step >= BLOCK_STEPS.length - 1) { setPlaying(false); return; }
    const id = setTimeout(() => setStep((s) => s + 1), 2100);
    return () => clearTimeout(id);
  }, [playing, step]);

  const advance = () => { setPlaying(false); setStep((s) => Math.min(BLOCK_STEPS.length - 1, s + 1)); };
  const restart = () => { setStep(0); setPlaying(true); };

  const cur = step >= 0 ? BLOCK_STEPS[step] : null;
  const streamX = 78;
  const top = 44, bottom = H - 40;
  const blockH = (bottom - top) / N_BLOCKS;

  // which block index is "active" (the highlighted one)
  const activeBlock = cur ? (cur.part === "repeat" ? 1 : 0) : -1;
  // how far the stream has been "refined" — fill grows as steps progress
  const fillFrac = step < 0 ? 0 : Math.min(1, (step + 1) / BLOCK_STEPS.length);

  return (
    <div style={{ userSelect: "none" }}>
      <div style={{ display: "flex", gap: 14 }}>
        {/* diagram */}
        <svg width="46%" viewBox={`0 0 ${W} ${H}`} style={{ display: "block", flexShrink: 0 }}>
          {/* stream spine with a "refined" fill rising from the bottom */}
          <line x1={streamX} y1={top} x2={streamX} y2={bottom} stroke={PALETTE.line} strokeWidth="10" strokeLinecap="round" />
          <line x1={streamX} y1={bottom} x2={streamX} y2={bottom - fillFrac * (bottom - top)} stroke={PALETTE.glow} strokeWidth="10" strokeLinecap="round" opacity="0.7" />
          <text x={streamX} y={bottom + 26} textAnchor="middle" fontFamily="monospace" fontSize="10" fill={PALETTE.apple} fontWeight={700}>word in</text>
          <text x={streamX} y={top - 18} textAnchor="middle" fontFamily="monospace" fontSize="10" fill={PALETTE.fruit} fontWeight={700}>richer vector</text>

          {Array.from({ length: N_BLOCKS }, (_, b) => {
            const yBot = bottom - b * blockH;
            const attnY = yBot - blockH * 0.34;
            const mlpY = yBot - blockH * 0.74;
            const isActive = b === activeBlock;
            const attnLit = isActive && cur && cur.part.startsWith("attn");
            const mlpLit = isActive && cur && cur.part.startsWith("mlp");
            const dim = activeBlock >= 0 && !isActive ? 0.3 : 1;
            return (
              <g key={b} opacity={dim}>
                <rect x={streamX - 14} y={yBot - blockH + 4} width={W - streamX - 6} height={blockH - 8} rx={6}
                  fill="none" stroke={isActive ? PALETTE.ink : PALETTE.line} strokeWidth={isActive ? 1.4 : 0.8} strokeDasharray={isActive ? "none" : "3 4"} />
                {/* attention sublayer */}
                <rect x={streamX + 30} y={attnY - 15} width={W - streamX - 56} height={30} rx={4}
                  fill={attnLit ? PALETTE.violet : "#fff"} stroke={PALETTE.violet} strokeWidth="1.3" />
                <text x={streamX + 30 + (W - streamX - 56) / 2} y={attnY + 4} textAnchor="middle" fontFamily="monospace" fontSize="10"
                  fill={attnLit ? "#fff" : PALETTE.violet} fontWeight={700}>attention</text>
                {/* mlp sublayer */}
                <rect x={streamX + 30} y={mlpY - 15} width={W - streamX - 56} height={30} rx={4}
                  fill={mlpLit ? PALETTE.tech : "#fff"} stroke={PALETTE.tech} strokeWidth="1.3" />
                <text x={streamX + 30 + (W - streamX - 56) / 2} y={mlpY + 4} textAnchor="middle" fontFamily="monospace" fontSize="10"
                  fill={mlpLit ? "#fff" : PALETTE.tech} fontWeight={700}>MLP</text>
                <text x={streamX - 20} y={(yBot - blockH / 2)} textAnchor="end" fontFamily="monospace" fontSize="9" fill={PALETTE.dim} fontWeight={700}>×{b + 1}</text>
                {/* norm dots */}
                <circle cx={streamX} cy={attnY + blockH * 0.17} r={3} fill={PALETTE.dim} />
                <circle cx={streamX} cy={mlpY + blockH * 0.17} r={3} fill={PALETTE.dim} />
              </g>
            );
          })}
          <text x={W / 2} y={H - 6} textAnchor="middle" fontFamily="monospace" fontSize="8.5" fill={PALETTE.dim}>● normalise before each step (pre-norm)</text>
        </svg>

        {/* caption panel */}
        <div style={{ flex: 1, display: "flex", flexDirection: "column", justifyContent: "center" }}>
          {cur ? (
            <div>
              <div style={{ fontFamily: "monospace", fontSize: 10, color: PALETTE.dim, marginBottom: 6 }}>
                step {step + 1} of {BLOCK_STEPS.length}
              </div>
              <div style={{ fontFamily: "'Georgia',serif", fontSize: 19, fontWeight: 700, color: cur.part.startsWith("mlp") ? PALETTE.tech : cur.part === "repeat" ? PALETTE.ink : PALETTE.violet, marginBottom: 8 }}>
                {cur.title}
              </div>
              <div style={{ fontSize: 14, lineHeight: 1.5, color: PALETTE.ink }}>{cur.why}</div>
            </div>
          ) : (
            <div style={{ fontSize: 14, lineHeight: 1.5, color: PALETTE.dim }}>
              A block has two steps — <b style={{ color: PALETTE.violet }}>attention</b> (mix across words) and an{" "}
              <b style={{ color: PALETTE.tech }}>MLP</b> (refine each word) — both added onto a shared running vector.
              Step through to see it, then watch why it repeats.
            </div>
          )}
        </div>
      </div>

      <div style={{ display: "flex", gap: 10, marginTop: 14 }}>
        <button onClick={playing ? () => setPlaying(false) : (step < 0 || step >= BLOCK_STEPS.length - 1 ? restart : () => setPlaying(true))} style={{
          flex: 1, background: PALETTE.ink, color: PALETTE.paper, border: "none", borderRadius: 3,
          padding: "11px 16px", fontSize: 14, fontWeight: 700, cursor: "pointer", fontFamily: "inherit",
        }}>
          {playing ? "⏸ Pause" : step >= BLOCK_STEPS.length - 1 ? "↻ Replay" : step < 0 ? "▶ Walk through a block" : "▶ Play"}
        </button>
        <button onClick={advance} disabled={step >= BLOCK_STEPS.length - 1} style={navBtn(step >= BLOCK_STEPS.length - 1)}>Step ▸</button>
      </div>
      <div style={{ fontFamily: "monospace", fontSize: 11, color: PALETTE.dim, textAlign: "center", marginTop: 8 }}>
        each step adds to the stream — nothing is overwritten, so the vector only ever gets richer
      </div>
    </div>
  );
}

// ===========================================================================
// NEURAL NET — learning the embedding document-by-document, then streaming the
// "millions." Kept from the original; the copy now reinforces that what's stored
// is WEIGHTS (a lossy compression), not a library of the example documents.
// ===========================================================================
const TRAIN_DOCS = [
  { text: "She picked a ripe apple from the orchard.", cue: ["ripe", "orchard"], pull: "fruit" },
  { text: "The apple chip outperforms last year's processor.", cue: ["chip", "processor"], pull: "tech" },
  { text: "He sliced the apple into the fruit salad.", cue: ["sliced", "fruit"], pull: "fruit" },
  { text: "Update the software on your Apple laptop.", cue: ["software", "laptop"], pull: "tech" },
  { text: "We baked an apple pie for dessert.", cue: ["baked", "pie"], pull: "fruit" },
];
const NN_LAYERS = [4, 6, 6, 5];

function NeuralNet() {
  const W = 560, H = 420;
  const embeddingDims = 8;
  const [docsTrained, setDocsTrained] = useState(0);
  const [activeDoc, setActiveDoc] = useState(null);
  const [pulse, setPulse] = useState(0);
  const [autoPlaying, setAutoPlaying] = useState(false);
  const [converged, setConverged] = useState(false);
  const [embTargets, setEmbTargets] = useState(() => Array.from({ length: embeddingDims }, () => 0.5));
  const [embVals, setEmbVals] = useState(() => Array.from({ length: embeddingDims }, () => 0.5));
  const [prevTargets, setPrevTargets] = useState(() => Array.from({ length: embeddingDims }, () => 0.5)); // ghost (where the bars were before this nudge)
  const [volatility, setVolatility] = useState(0); // smoothed "how much it just moved"
  const pulseRaf = useRef(null);
  const autoRaf = useRef(null);
  const docsRef = useRef(0);
  const CONVERGE_AT = 2_000_000;
  const settle = Math.min(1, Math.log10(docsTrained + 1) / Math.log10(CONVERGE_AT));

  useEffect(() => {
    let f;
    const step = () => { setEmbVals((cur) => cur.map((v, i) => v + (embTargets[i] - v) * 0.12)); f = requestAnimationFrame(step); };
    f = requestAnimationFrame(step);
    return () => cancelAnimationFrame(f);
  }, [embTargets]);

  const firePulse = () => {
    setPulse(0); cancelAnimationFrame(pulseRaf.current);
    const start = performance.now(); const dur = 900;
    const tick = (now) => { const t = Math.min(1, (now - start) / dur); setPulse(t); if (t < 1) pulseRaf.current = requestAnimationFrame(tick); };
    pulseRaf.current = requestAnimationFrame(tick);
  };

  const nudgeEmbedding = (pull, count) => {
    const progress = Math.min(1, Math.log10(count + 1) / Math.log10(CONVERGE_AT));
    const stepSize = 0.28 * (1 - progress) ** 1.5;
    setEmbTargets((cur) => {
      setPrevTargets(cur); // remember where we were, for the ghost trail
      const next = cur.map((v, i) => {
        const dir = pull === "fruit" ? (i % 2 === 0 ? 1 : -1) : (i % 2 === 0 ? -1 : 1);
        const goal = 0.5 + dir * (0.3 + 0.15 * Math.sin(i * 1.7));
        return v + (goal - v) * stepSize;
      });
      // measure how far it moved this step → drives the wobble meter
      const moved = next.reduce((a, v, i) => a + Math.abs(v - cur[i]), 0) / embeddingDims;
      setVolatility((prev) => prev * 0.4 + moved * 8 * 0.6); // smoothed, scaled for display
      return next;
    });
  };

  const fireDoc = (idx) => {
    setActiveDoc(idx); firePulse();
    setTimeout(() => setActiveDoc((d) => (d === idx ? null : d)), 950);
    const count = docsRef.current + 1; docsRef.current = count; setDocsTrained(count);
    nudgeEmbedding(TRAIN_DOCS[idx].pull, count);
  };

  const startAuto = () => {
    if (autoPlaying) return;
    setConverged(false); setAutoPlaying(true); setActiveDoc(null);
    let i = 0;
    const burst = () => {
      const next = Math.round((docsRef.current + 1) * 2.4 + 25);
      docsRef.current = next; setDocsTrained(next); firePulse();
      nudgeEmbedding(i % 2 === 0 ? "fruit" : "tech", next); i++;
      if (next >= CONVERGE_AT) {
        docsRef.current = CONVERGE_AT; setDocsTrained(CONVERGE_AT);
        setConverged(true); setAutoPlaying(false); setPulse(0); return;
      }
      // slow the cadence as it converges, so the eye catches the settling
      const prog = Math.min(1, Math.log10(next + 1) / Math.log10(CONVERGE_AT));
      const delay = 150 + prog * prog * 620; // ~150ms early → ~770ms near the end
      autoRaf.current = setTimeout(burst, delay);
    };
    burst();
  };
  const stopAuto = () => { setAutoPlaying(false); clearTimeout(autoRaf.current); cancelAnimationFrame(pulseRaf.current); setActiveDoc(null); setPulse(0); };
  useEffect(() => () => { clearTimeout(autoRaf.current); cancelAnimationFrame(pulseRaf.current); }, []);
  const reset = () => { stopAuto(); setConverged(false); docsRef.current = 0; setDocsTrained(0); setEmbTargets(Array.from({ length: embeddingDims }, () => 0.5)); setPrevTargets(Array.from({ length: embeddingDims }, () => 0.5)); setVolatility(0); };

  const colW = (W - 150) / NN_LAYERS.length;
  const nodePos = (li, ni, count) => ({ x: 40 + li * colW, y: H / 2 + (ni - (count - 1) / 2) * (H / (count + 1)) });
  const sweep = pulse * (NN_LAYERS.length + 0.4);
  const firing = pulse > 0 && pulse < 1;
  const edges = [];
  for (let li = 0; li < NN_LAYERS.length - 1; li++)
    for (let a = 0; a < NN_LAYERS[li]; a++)
      for (let b = 0; b < NN_LAYERS[li + 1]; b++) edges.push({ li, a, b });
  const embX = 40 + (NN_LAYERS.length - 1) * colW + 70;
  const fmtDocs = (n) => n >= 1e6 ? (n / 1e6).toFixed(1) + "M" : n >= 1e3 ? (n / 1e3).toFixed(1) + "k" : n;

  return (
    <div style={{ userSelect: "none" }}>
      <svg width="100%" viewBox={`0 0 ${W} ${H}`} style={{ display: "block" }}>
        {edges.map((e, i) => {
          const p1 = nodePos(e.li, e.a, NN_LAYERS[e.li]);
          const p2 = nodePos(e.li + 1, e.b, NN_LAYERS[e.li + 1]);
          const w = 0.5 + 0.5 * Math.sin(i * 12.9 + docsTrained * 0.6);
          const lit = firing && sweep > e.li && sweep < e.li + 1.4;
          return (<line key={i} x1={p1.x} y1={p1.y} x2={p2.x} y2={p2.y}
            stroke={lit ? PALETTE.glow : PALETTE.tech}
            strokeWidth={lit ? 1.8 : 0.6 + w * 0.9 * (1 - settle * 0.5)}
            opacity={lit ? 0.9 : 0.1 + w * 0.2 * (1 - settle * 0.4)} />);
        })}
        {NN_LAYERS.flatMap((count, li) => Array.from({ length: count }, (_, ni) => {
          const nd = nodePos(li, ni, count);
          const lit = firing && sweep > li - 0.3 && sweep < li + 0.9;
          const r = li === 0 ? 9 : 7;
          const color = li === 0 ? PALETTE.apple : li === NN_LAYERS.length - 1 ? PALETTE.fruit : PALETTE.tech;
          return (<g key={`${li}-${ni}`}>{lit && <circle cx={nd.x} cy={nd.y} r={r + 6} fill={color} opacity={0.28} />}
            <circle cx={nd.x} cy={nd.y} r={r} fill={color} opacity={lit ? 1 : 0.5} /></g>);
        }))}
        <text x={40} y={H / 2 - (NN_LAYERS[0] / 2) * (H / (NN_LAYERS[0] + 1)) - 16} textAnchor="middle" fontFamily="'Georgia', serif" fontSize="15" fontWeight="700" fill={PALETTE.apple}>“apple”</text>
        <text x={40} y={H - 14} textAnchor="middle" fontFamily="monospace" fontSize="11" fill={PALETTE.dim}>token in</text>
        <text x={40 + colW * 1.5} y={26} textAnchor="middle" fontFamily="monospace" fontSize="11" fill={PALETTE.dim}>{firing ? "context flowing through the weights" : "hidden layers"}</text>
        <text x={embX + 14} y={H / 2 - embeddingDims * 11 - 16} textAnchor="middle" fontFamily="monospace" fontSize="11" fill={PALETTE.dim}>embedding vector</text>
        {embVals.map((v, i) => {
          const barY = H / 2 - embeddingDims * 11 + i * 22;
          const ghost = prevTargets[i]; // where it was heading before the latest nudge
          const moved = Math.abs(v - ghost) > 0.012;
          return (<g key={i}>
            <rect x={embX} y={barY} width={70} height={14} rx={2} fill="#e6dcc8" />
            {/* ghost outline of the previous position — the "wobble" the eye should catch */}
            {moved && <rect x={embX} y={barY} width={70 * Math.max(0.04, ghost)} height={14} rx={2} fill="none" stroke={PALETTE.glow} strokeWidth="1.5" strokeDasharray="2 2" opacity={0.8} />}
            <rect x={embX} y={barY} width={70 * Math.max(0.04, v)} height={14} rx={2} fill={PALETTE.apple} opacity={0.5 + settle * 0.5} />
            <text x={embX - 6} y={barY + 11} textAnchor="end" fontFamily="monospace" fontSize="9" fill={PALETTE.dim}>{(v * 2 - 1).toFixed(2)}</text>
          </g>);
        })}
        <line x1={40 + (NN_LAYERS.length - 1) * colW} y1={H / 2} x2={embX - 4} y2={H / 2} stroke={PALETTE.line} strokeWidth="1.5" strokeDasharray="3 3" />
      </svg>

      <div style={{ display: "flex", alignItems: "center", gap: 12, margin: "6px 0 12px" }}>
        <span style={{ fontFamily: "monospace", fontSize: 12, color: PALETTE.dim }}>documents trained</span>
        <span style={{ fontFamily: "monospace", fontSize: 18, fontWeight: 700, color: PALETTE.apple }}>{fmtDocs(docsTrained)}</span>
        <div style={{ flex: 1, height: 6, background: "#e6dcc8", borderRadius: 3, overflow: "hidden" }}>
          <div style={{ width: `${settle * 100}%`, height: "100%", background: converged || settle >= 0.95 ? PALETTE.fruit : PALETTE.glow, transition: "width 0.3s" }} />
        </div>
        <span style={{ fontFamily: "monospace", fontSize: 11, fontWeight: converged ? 700 : 400, color: converged || settle >= 0.95 ? PALETTE.fruit : PALETTE.dim }}>{converged || settle >= 0.95 ? "✓ stable" : "learning"}</span>
      </div>

      {/* vector-movement (volatility) meter — draws the eye to the settling */}
      <div style={{ display: "flex", alignItems: "center", gap: 12, margin: "0 0 14px" }}>
        <span style={{ fontFamily: "monospace", fontSize: 12, color: PALETTE.dim, whiteSpace: "nowrap" }}>vector movement</span>
        <div style={{ flex: 1, height: 6, background: "#e6dcc8", borderRadius: 3, overflow: "hidden" }}>
          <div style={{ width: `${Math.min(100, volatility * 100)}%`, height: "100%", background: volatility > 0.25 ? PALETTE.glow : PALETTE.fruit, transition: "width 0.25s, background 0.25s" }} />
        </div>
        <span style={{ fontFamily: "monospace", fontSize: 11, color: volatility > 0.25 ? PALETTE.glow : PALETTE.fruit, fontWeight: 700, whiteSpace: "nowrap", width: 78, textAlign: "right" }}>
          {volatility > 0.5 ? "big jumps" : volatility > 0.15 ? "settling" : "barely moving"}
        </span>
      </div>

      <div style={{ fontFamily: "monospace", fontSize: 10, textTransform: "uppercase", letterSpacing: 0.5, color: PALETTE.dim, marginBottom: 6 }}>a few example documents</div>
      <div style={{ position: "relative" }}>
        <div style={{ display: "flex", flexDirection: "column", gap: 7, opacity: autoPlaying ? 0.25 : 1, filter: autoPlaying ? "blur(1px)" : "none", transition: "opacity 0.3s, filter 0.3s" }}>
          {TRAIN_DOCS.map((doc, idx) => {
            const isActive = activeDoc === idx;
            const acc = doc.pull === "fruit" ? PALETTE.fruit : PALETTE.tech;
            return (<div key={idx} style={{ display: "flex", alignItems: "center", gap: 10, background: isActive ? "#fff" : "transparent", border: `1px solid ${isActive ? acc : PALETTE.line}`, borderLeft: `4px solid ${acc}`, borderRadius: 3, padding: "8px 10px" }}>
              <span style={{ flex: 1, fontSize: 14, lineHeight: 1.4 }}>
                {doc.text.split(/(\s+)/).map((word, wi) => {
                  const clean = word.replace(/[^a-zA-Z]/g, "").toLowerCase();
                  const isCue = doc.cue.includes(clean);
                  return (<span key={wi} style={{ background: isCue && isActive ? PALETTE.glow : "transparent", fontWeight: isCue ? 700 : 400, color: isCue ? PALETTE.ink : "inherit", borderRadius: 3, padding: isCue && isActive ? "1px 3px" : 0, transition: "background 0.3s" }}>{word}</span>);
                })}
              </span>
              <button onClick={() => !autoPlaying && fireDoc(idx)} disabled={autoPlaying} style={{ background: autoPlaying ? "transparent" : acc, color: autoPlaying ? PALETTE.line : "#fff", border: "none", borderRadius: 3, padding: "6px 12px", fontSize: 13, fontWeight: 600, cursor: autoPlaying ? "default" : "pointer", fontFamily: "inherit", whiteSpace: "nowrap" }}>Train ▸</button>
            </div>);
          })}
        </div>
        {autoPlaying && (
          <div style={{ position: "absolute", inset: 0, display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", textAlign: "center", gap: 4 }}>
            <div style={{ fontFamily: "monospace", fontSize: 13, fontWeight: 700, color: PALETTE.ink }}>streaming millions of other documents…</div>
            <div style={{ fontFamily: "monospace", fontSize: 11, color: PALETTE.dim, maxWidth: 320 }}>real text from across the web — far more varied than these five examples</div>
          </div>
        )}
      </div>

      <div style={{ display: "flex", gap: 10, marginTop: 12 }}>
        <button onClick={autoPlaying ? stopAuto : startAuto} disabled={converged} style={{ flex: 1, background: converged ? "#e6dcc8" : PALETTE.ink, color: converged ? PALETTE.dim : PALETTE.paper, border: "none", borderRadius: 3, padding: "11px 16px", fontSize: 14, fontWeight: 700, cursor: converged ? "default" : "pointer", fontFamily: "inherit" }}>
          {converged ? "✓ Training complete" : autoPlaying ? "⏸ Pause training" : "▶ Train on millions of documents"}
        </button>
        <button onClick={reset} style={navBtn(false)}>↻ Reset</button>
      </div>
      <div style={{ fontFamily: "monospace", fontSize: 11, textAlign: "center", marginTop: 8, color: converged ? PALETTE.fruit : PALETTE.dim, fontWeight: converged ? 700 : 400 }}>
        {converged
          ? "✓ converged — the documents aren't stored; their pattern is compressed into the weights"
          : autoPlaying
            ? "streaming documents — each one nudges the weights less than the last"
            : "click a document to teach the model, or press play"}
      </div>
    </div>
  );
}

// ===========================================================================
// TRAINING PIPELINE — the three stages that turn a raw predictor into an
// assistant: pretraining (next-token), SFT (demonstrations), and preference
// optimisation (RLHF / DPO). Click a stage to see what changes.
// ===========================================================================
const TRAIN_STAGES = [
  {
    key: "pretrain", n: 1, title: "Pretraining", color: PALETTE.apple,
    sub: "self-supervised · ~15T tokens",
    what: "Predict the next token over a huge web corpus. No labels — the text supervises itself via cross-entropy loss.",
    prompt: "The capital of France is",
    base: "the largest city in the country, with a population of over two million people and…",
    label: "completes text, but rambles — it has knowledge, not manners",
  },
  {
    key: "sft", n: 2, title: "Supervised fine-tuning", color: PALETTE.tech,
    sub: "demonstrations · thousands of pairs",
    what: "Continue training on curated (instruction, good-response) pairs. Still next-token prediction — just on demonstrations of being helpful.",
    prompt: "What is the capital of France?",
    base: "The capital of France is Paris.",
    label: "now it answers the question directly",
  },
  {
    key: "rlhf", n: 3, title: "Preference optimisation", color: PALETTE.fruit,
    sub: "RLHF / DPO · human or AI rankings",
    what: "Humans (or an AI) rank competing responses; the model is steered toward the preferred ones — via a reward model + PPO (RLHF) or directly (DPO).",
    prompt: "What is the capital of France?",
    base: "Paris is the capital of France. It sits on the Seine and has been the political and cultural centre since the 12th century. Anything else you'd like to know?",
    label: "helpful, well-judged tone — the aligned assistant you talk to",
  },
];

function Training() {
  const [stage, setStage] = useState(0);
  const s = TRAIN_STAGES[stage];
  return (
    <div style={{ userSelect: "none" }}>
      {/* stage rail */}
      <div style={{ display: "flex", gap: 6, marginBottom: 16 }}>
        {TRAIN_STAGES.map((st, i) => {
          const on = i === stage; const passed = i < stage;
          return (
            <button key={st.key} onClick={() => setStage(i)} style={{
              flex: 1, textAlign: "left", cursor: "pointer", fontFamily: "inherit",
              background: on ? "#fff" : "transparent",
              borderStyle: "solid", borderWidth: "4px 1px 1px 1px",
              borderColor: `${on || passed ? st.color : PALETTE.line} ${on ? st.color : PALETTE.line} ${on ? st.color : PALETTE.line} ${on ? st.color : PALETTE.line}`,
              borderRadius: 3, padding: "8px 10px",
            }}>
              <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
                <span style={{ width: 18, height: 18, borderRadius: "50%", background: on || passed ? st.color : PALETTE.line, color: "#fff", display: "flex", alignItems: "center", justifyContent: "center", fontFamily: "monospace", fontSize: 11, fontWeight: 700 }}>{st.n}</span>
                <span style={{ fontFamily: "'Georgia',serif", fontSize: 13, fontWeight: 700, color: PALETTE.ink, lineHeight: 1.1 }}>{st.title}</span>
              </div>
              <div style={{ fontFamily: "monospace", fontSize: 9, color: PALETTE.dim, marginTop: 4 }}>{st.sub}</div>
            </button>
          );
        })}
      </div>

      <div style={{ background: "#f6efe1", border: `1px solid ${PALETTE.line}`, borderLeft: `4px solid ${s.color}`, borderRadius: 4, padding: "12px 14px", marginBottom: 14 }}>
        <div style={{ fontSize: 13.5, lineHeight: 1.5, color: PALETTE.ink }}>{s.what}</div>
      </div>

      {/* same prompt, evolving answer */}
      <div style={{ fontFamily: "monospace", fontSize: 10, textTransform: "uppercase", letterSpacing: 0.5, color: PALETTE.dim, marginBottom: 5 }}>same prompt → this stage's behaviour</div>
      <div style={{ background: "#fff", border: `1px solid ${PALETTE.line}`, borderRadius: 4, padding: "12px 14px" }}>
        <div style={{ fontFamily: "'Georgia',serif", fontSize: 15, color: PALETTE.dim, marginBottom: 8 }}>
          <span style={{ fontFamily: "monospace", fontSize: 10, color: PALETTE.dim }}>prompt&nbsp;&nbsp;</span>{s.prompt}
        </div>
        <div style={{ height: 1, background: PALETTE.line, margin: "0 0 8px", opacity: 0.6 }} />
        <div style={{ fontFamily: "'Georgia',serif", fontSize: 16, lineHeight: 1.45, color: PALETTE.ink }}>
          <span style={{ fontFamily: "monospace", fontSize: 10, color: s.color, fontWeight: 700 }}>output&nbsp;&nbsp;</span>{s.base}
        </div>
      </div>
      <div style={{ fontFamily: "monospace", fontSize: 11, color: s.color, textAlign: "center", marginTop: 10, fontWeight: 600 }}>{s.label}</div>
      <div style={{ fontFamily: "monospace", fontSize: 10.5, color: PALETTE.dim, textAlign: "center", marginTop: 8 }}>
        only stage 1 is the “base model” · stages 2–3 are <b>alignment</b> — same weights, new behaviour
      </div>
    </div>
  );
}

// ===========================================================================
// CONTEXT WINDOW — the n² cost of attention and the growing KV cache. Drag the
// length slider: the attention area grows quadratically while the sliding-window
// band stays linear, and the KV-cache memory bar fills.
// ===========================================================================
function ContextWindow() {
  const [len, setLen] = useState(8);
  const MAX = 24, WIN = 5;
  const cell = Math.min(11, 240 / MAX);
  const grid = MAX * cell;

  const fullCost = len * (len + 1) / 2;             // causal: lower triangle
  const windowCost = Math.max(0, len * WIN - WIN * (WIN - 1) / 2);
  const maxFull = MAX * (MAX + 1) / 2;

  return (
    <div style={{ userSelect: "none" }}>
      <div style={{ display: "flex", gap: 20, justifyContent: "center", marginBottom: 8 }}>
        {/* full causal attention */}
        <div style={{ textAlign: "center" }}>
          <div style={{ fontFamily: "monospace", fontSize: 11, color: PALETTE.apple, fontWeight: 700, marginBottom: 6 }}>full attention · O(n²)</div>
          <svg width={grid} height={grid} style={{ display: "block" }}>
            {Array.from({ length: MAX }, (_, i) => Array.from({ length: MAX }, (_, j) => {
              const inRange = i < len && j < len;
              const causal = j <= i;
              const on = inRange && causal;
              return <rect key={`${i}-${j}`} x={j * cell} y={i * cell} width={cell - 1} height={cell - 1}
                fill={on ? PALETTE.apple : "#e6dcc8"} opacity={on ? 0.55 : 0.25} />;
            }))}
          </svg>
        </div>
        {/* sliding window */}
        <div style={{ textAlign: "center" }}>
          <div style={{ fontFamily: "monospace", fontSize: 11, color: PALETTE.tech, fontWeight: 700, marginBottom: 6 }}>sliding window · O(n·w)</div>
          <svg width={grid} height={grid} style={{ display: "block" }}>
            {Array.from({ length: MAX }, (_, i) => Array.from({ length: MAX }, (_, j) => {
              const inRange = i < len && j < len;
              const inWindow = j <= i && j > i - WIN;
              const on = inRange && inWindow;
              return <rect key={`${i}-${j}`} x={j * cell} y={i * cell} width={cell - 1} height={cell - 1}
                fill={on ? PALETTE.tech : "#e6dcc8"} opacity={on ? 0.6 : 0.25} />;
            }))}
          </svg>
        </div>
      </div>

      <div style={{ display: "flex", justifyContent: "space-between", fontFamily: "monospace", fontSize: 11, color: PALETTE.dim, marginBottom: 4 }}>
        <span>context length</span><span style={{ color: PALETTE.ink, fontWeight: 700 }}>{len} tokens</span>
      </div>
      <input type="range" min="1" max={MAX} step="1" value={len} onChange={(e) => setLen(parseInt(e.target.value))} style={sliderStyle(PALETTE.apple)} />

      <div style={{ display: "flex", gap: 12, marginTop: 16 }}>
        <div style={{ flex: 1 }}>
          <div style={{ display: "flex", justifyContent: "space-between", fontFamily: "monospace", fontSize: 10, color: PALETTE.dim, marginBottom: 3 }}>
            <span>attention pair-comparisons</span><span style={{ color: PALETTE.apple, fontWeight: 700 }}>{fullCost}</span>
          </div>
          <div style={{ height: 8, background: "#e6dcc8", borderRadius: 3, overflow: "hidden" }}>
            <div style={{ width: `${(fullCost / maxFull) * 100}%`, height: "100%", background: PALETTE.apple, transition: "width 0.15s" }} />
          </div>
        </div>
        <div style={{ flex: 1 }}>
          <div style={{ display: "flex", justifyContent: "space-between", fontFamily: "monospace", fontSize: 10, color: PALETTE.dim, marginBottom: 3 }}>
            <span>KV-cache entries (∝ length)</span><span style={{ color: PALETTE.tech, fontWeight: 700 }}>{len}</span>
          </div>
          <div style={{ height: 8, background: "#e6dcc8", borderRadius: 3, overflow: "hidden" }}>
            <div style={{ width: `${(len / MAX) * 100}%`, height: "100%", background: PALETTE.tech, transition: "width 0.15s" }} />
          </div>
        </div>
      </div>
      <div style={{ fontFamily: "monospace", fontSize: 11, color: PALETTE.dim, textAlign: "center", marginTop: 12 }}>
        doubling the length roughly <b>quadruples</b> full-attention cost — why long context is hard, and why windows & KV-cache tricks exist
      </div>
    </div>
  );
}

// ===========================================================================
// TOKEN GENERATION — the four-beat autoregressive loop. CORRECTED: the "blend"
// is a dot product between the final hidden vector and every token's row of the
// unembedding matrix, then softmax — NOT cosine similarity.
// ===========================================================================
const GEN_PROMPT = "How fast is the new Apple chip?";
const GEN_PROMPT_CUES = ["fast", "chip"];
const GEN_STEPS = [
  { chosen: "The", cands: [["The", 0.52], ["Apple", 0.27], ["Its", 0.13], ["A", 0.08]] },
  { chosen: "Apple", cands: [["Apple", 0.64], ["new", 0.19], ["M5", 0.10], ["latest", 0.07]] },
  { chosen: "chip", cands: [["chip", 0.66], ["silicon", 0.17], ["laptop", 0.11], ["team", 0.06]] },
  { chosen: "is", cands: [["is", 0.58], ["runs", 0.24], ["was", 0.12], ["feels", 0.06]] },
  { chosen: "remarkably", cands: [["remarkably", 0.44], ["very", 0.31], ["much", 0.16], ["so", 0.09]] },
  { chosen: "fast", cands: [["fast", 0.69], ["quick", 0.18], ["efficient", 0.08], ["snappy", 0.05]] },
  { chosen: ".", cands: [[".", 0.84], ["now", 0.09], ["today", 0.04], ["indeed", 0.03]] },
];
const GEN_BEATS = ["feed", "score", "pick", "append"];

function TokenGen() {
  const [step, setStep] = useState(0);
  const [beat, setBeat] = useState(0);
  const [playing, setPlaying] = useState(false);
  const done = step >= GEN_STEPS.length;
  const current = !done ? GEN_STEPS[step] : null;
  const emitted = GEN_STEPS.slice(0, step);

  const advance = () => {
    if (done) return;
    setBeat((b) => { if (b < 3) return b + 1; setStep((s) => s + 1); return 0; });
  };
  useEffect(() => {
    if (!playing || done) return;
    const id = setTimeout(advance, beat === 1 ? 1100 : 850);
    return () => clearTimeout(id);
  }, [playing, beat, step, done]);
  const replay = () => { setStep(0); setBeat(0); setPlaying(true); };

  const beatLabel = [
    "1 · feed the sequence so far back into the model",
    "2 · score every token: hidden vector · unembedding matrix → logits",
    "3 · softmax → probabilities, then sample one token",
    "4 · append it, then repeat",
  ];

  return (
    <div style={{ userSelect: "none" }}>
      <div style={{ marginBottom: 8 }}>
        <div style={{ fontFamily: "monospace", fontSize: 10, textTransform: "uppercase", letterSpacing: 0.5, color: PALETTE.dim, marginBottom: 4 }}>the question (prompt)</div>
        <div style={{ background: "#f6efe1", border: `1px solid ${PALETTE.line}`, borderLeft: `4px solid ${PALETTE.tech}`, borderRadius: 3, padding: "9px 12px", fontFamily: "'Georgia', serif", fontSize: 16 }}>
          {GEN_PROMPT.split(/(\s+)/).map((w, i) => {
            const clean = w.replace(/[^a-zA-Z]/g, "").toLowerCase();
            const isCue = GEN_PROMPT_CUES.includes(clean);
            return (<span key={i} style={{ background: isCue ? PALETTE.glow : "transparent", fontWeight: isCue ? 700 : 400, borderRadius: 3, padding: isCue ? "1px 3px" : 0 }}>{w}</span>);
          })}
        </div>
        <div style={{ fontFamily: "monospace", fontSize: 10, color: PALETTE.dim, marginTop: 4 }}>“fast” + “chip” fix <i>Apple</i> as the tech meaning — that context shapes every token below</div>
      </div>

      <div style={{ minHeight: 64, background: "#fff", border: `1px solid ${PALETTE.line}`, borderRadius: 4, padding: "14px 16px", fontFamily: "'Georgia', serif", fontSize: 21, lineHeight: 1.4, display: "flex", flexWrap: "wrap", alignItems: "center" }}>
        <span style={{ fontFamily: "monospace", fontSize: 11, color: PALETTE.dim, marginRight: 8 }}>answer:</span>
        {emitted.map((g, i) => (<span key={i} style={{ marginRight: g.chosen === "." ? 0 : 7 }}>{g.chosen}</span>))}
        {!done && beat === 3 && (<span style={{ marginRight: current.chosen === "." ? 0 : 7, color: PALETTE.apple, fontWeight: 700, background: PALETTE.glow, borderRadius: 3, padding: "0 4px" }}>{current.chosen}</span>)}
        {!done && beat < 3 && <span style={{ display: "inline-block", width: 10, height: 22, background: PALETTE.apple, marginLeft: 2, animation: "blink 1s steps(2) infinite" }} />}
        {done && <span style={{ color: PALETTE.fruit, fontSize: 13, fontFamily: "monospace", marginLeft: 10 }}>✓ stop token — done</span>}
      </div>

      <div style={{ display: "flex", gap: 6, margin: "12px 0 14px" }}>
        {GEN_BEATS.map((b, i) => {
          const active = !done && beat === i;
          return (<div key={b} style={{ flex: 1, textAlign: "center" }}>
            <div style={{ height: 4, borderRadius: 2, marginBottom: 5, background: active ? PALETTE.apple : (!done && i < beat) ? PALETTE.fruit : PALETTE.line, transition: "background 0.3s" }} />
            <span style={{ fontFamily: "monospace", fontSize: 10, textTransform: "uppercase", letterSpacing: 0.5, color: active ? PALETTE.apple : PALETTE.dim, fontWeight: active ? 700 : 400 }}>{b}</span>
          </div>);
        })}
      </div>

      <div style={{ minHeight: 188 }}>
        <div style={{ fontFamily: "monospace", fontSize: 11, color: PALETTE.ink, marginBottom: 10, minHeight: 16 }}>{done ? "sequence complete" : beatLabel[beat]}</div>
        {!done && beat === 0 && (
          <div style={{ fontSize: 14, lineHeight: 1.5, color: PALETTE.dim }}>
            The model reads the question plus everything generated so far —{" "}
            <span style={{ fontFamily: "'Georgia',serif", color: PALETTE.ink }}>“{GEN_PROMPT}{emitted.length ? " → " + emitted.map((g) => g.chosen).join(" ") : ""} __”</span>{" "}
            — as context-aware vectors, and must predict what comes next.
          </div>
        )}
        {!done && beat === 1 && (
          <div>
            <div style={{ fontSize: 13, lineHeight: 1.5, color: PALETTE.dim, marginBottom: 10 }}>
              It takes its final hidden vector and computes a <b>dot product with every token's row</b> of the unembedding matrix. Each dot product is one <b>logit</b> — a raw score for that token. Top candidates:
            </div>
            <DistRows cands={current.cands} highlightTop logits />
          </div>
        )}
        {!done && beat >= 2 && (
          <div>
            <div style={{ fontSize: 13, lineHeight: 1.5, color: PALETTE.dim, marginBottom: 10 }}>
              {beat === 2
                ? <><b>Softmax</b> turns those logits into probabilities, then a token is <b>sampled</b> (a weighted draw — see the next slides). This step picks:</>
                : <>Picked <b style={{ color: PALETTE.apple }}>{current.chosen === "." ? "“.”" : current.chosen}</b>. It's appended to the sequence, and the loop runs again.</>}
            </div>
            <DistRows cands={current.cands} chosen={current.chosen} />
          </div>
        )}
      </div>

      <div style={{ display: "flex", gap: 10, marginTop: 6 }}>
        {!done && (<button onClick={() => { setPlaying(false); advance(); }} style={{ flex: 1, background: PALETTE.ink, color: PALETTE.paper, border: "none", borderRadius: 3, padding: "11px 16px", fontSize: 14, fontWeight: 700, cursor: "pointer", fontFamily: "inherit" }}>{beat < 3 ? "Step ▸" : "Next token ▸"}</button>)}
        <button onClick={() => setPlaying((p) => !p)} disabled={done} style={navBtn(done)}>{playing ? "⏸ Pause" : "▶ Auto"}</button>
        <button onClick={replay} style={navBtn(false)}>↻ Replay</button>
      </div>
      <style>{`@keyframes blink { 0%,50%{opacity:1} 50.01%,100%{opacity:0} }`}</style>
    </div>
  );
}

function DistRows({ cands, highlightTop, chosen, logits }) {
  const maxP = Math.max(...cands.map((c) => c[1]));
  return (
    <div>
      {cands.map(([tok, prob], i) => {
        const isChosen = chosen != null && tok === chosen;
        const isTop = highlightTop && i === 0;
        const accent = isChosen ? PALETTE.apple : isTop ? PALETTE.apple : PALETTE.tech;
        return (
          <div key={tok} style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 6 }}>
            <span style={{ width: 86, textAlign: "right", fontFamily: "monospace", fontSize: 13, fontWeight: isChosen || isTop ? 700 : 400, color: isChosen || isTop ? PALETTE.apple : PALETTE.ink }}>{tok === "." ? "“.”" : tok}</span>
            <div style={{ flex: 1, height: 17, background: "#e6dcc8", borderRadius: 3, overflow: "hidden", position: "relative" }}>
              <div style={{ width: `${(prob / maxP) * 100}%`, height: "100%", background: accent, opacity: isChosen ? 1 : isTop ? 0.9 : 0.5, transition: "width 0.4s, opacity 0.3s" }} />
            </div>
            <span style={{ width: 40, fontFamily: "monospace", fontSize: 12, color: PALETTE.dim }}>{logits ? (Math.log(prob) + 4).toFixed(1) : (prob * 100).toFixed(0) + "%"}</span>
            {isChosen && <span style={{ fontFamily: "monospace", fontSize: 11, color: PALETTE.apple, fontWeight: 700 }}>◄ picked</span>}
          </div>
        );
      })}
      {logits && <div style={{ fontFamily: "monospace", fontSize: 10, color: PALETTE.dim, marginTop: 2, paddingLeft: 96 }}>bars show logits (raw scores) · softmax converts them to % next</div>}
    </div>
  );
}

// ===========================================================================
// SAMPLING — temperature reshapes, top-p truncates. Running example:
// "The Apple chip is ___".
// ===========================================================================
const BASE_TOKENS = [
  { tok: "fast",      logit: 3.0,  off: { x: 0.05,  y: 0.05,  z: 0.04 } },
  { tok: "powerful",  logit: 2.3,  off: { x: -0.06, y: 0.10,  z: -0.05 } },
  { tok: "efficient", logit: 1.6,  off: { x: 0.12,  y: -0.08, z: 0.06 } },
  { tok: "quick",     logit: 1.0,  off: { x: -0.10, y: -0.10, z: 0.10 } },
  { tok: "blazing",   logit: 0.3,  off: { x: 0.22,  y: 0.18,  z: -0.14 } },
  { tok: "snappy",    logit: -0.4, off: { x: -0.24, y: 0.20,  z: 0.18 } },
  { tok: "magical",   logit: -1.2, off: { x: 0.34,  y: -0.30, z: -0.28 } },
  { tok: "edible",    logit: -2.2, off: { x: -0.55, y: -0.45, z: 0.40 } },
];

function softmaxWithTemp(tokens, temp) {
  const t = Math.max(0.01, temp);
  const scaled = tokens.map((d) => d.logit / t);
  const max = Math.max(...scaled);
  const exps = scaled.map((s) => Math.exp(s - max));
  const sum = exps.reduce((a, b) => a + b, 0);
  return tokens.map((d, i) => ({ ...d, p: exps[i] / sum }));
}

function weightedPick(weights) {
  const total = weights.reduce((a, b) => a + b, 0);
  let r = Math.random() * total;
  for (let i = 0; i < weights.length; i++) { r -= weights[i]; if (r <= 0) return i; }
  return weights.length - 1;
}

function DistBars({ dist, mode, cutoffIndex, rolledTok }) {
  return (
    <div>
      <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 4 }}>
        <span style={{ width: 78 }} />
        <div style={{ flex: 1, display: "flex", justifyContent: "space-between", fontFamily: "monospace", fontSize: 9, color: PALETTE.dim }}>
          <span>0%</span><span>50%</span><span>100%</span>
        </div>
        <span style={{ width: 44 }} /><span style={{ width: 14 }} />
      </div>
      {dist.map((d, i) => {
        const kept = mode !== "topp" || i <= cutoffIndex;
        const isTop = i === 0;
        const isRolled = rolledTok != null && d.tok === rolledTok;
        const barColor = isRolled ? PALETTE.apple : !kept ? PALETTE.line : isTop ? PALETTE.apple : PALETTE.tech;
        return (
          <div key={d.tok} style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 6, opacity: kept ? 1 : 0.4 }}>
            <span style={{ width: 78, textAlign: "right", fontFamily: "monospace", fontSize: 13, fontWeight: (isTop && kept) || isRolled ? 700 : 400, color: isRolled ? PALETTE.apple : kept ? PALETTE.ink : PALETTE.dim, textDecoration: kept ? "none" : "line-through" }}>{d.tok}</span>
            <div style={{ flex: 1, height: 17, background: "#e6dcc8", borderRadius: 3, overflow: "hidden", boxShadow: isRolled ? `0 0 0 2px ${PALETTE.apple}` : "none", position: "relative" }}>
              <div style={{ position: "absolute", left: "50%", top: 0, bottom: 0, width: 1, background: "#d8cbb0" }} />
              <div style={{ width: `${d.p * 100}%`, height: "100%", background: barColor, opacity: kept ? (isRolled || isTop ? 1 : 0.6) : 0.4, transition: "width 0.25s, background 0.25s, opacity 0.25s" }} />
            </div>
            <span style={{ width: 44, fontFamily: "monospace", fontSize: 12, color: PALETTE.dim }}>{(d.p * 100).toFixed(0) + "%"}</span>
            {isRolled ? <span style={{ width: 14, fontFamily: "monospace", fontSize: 11, color: PALETTE.apple, fontWeight: 700 }}>◄</span> : <span style={{ width: 14 }} />}
          </div>
        );
      })}
      {mode === "topp" && (<div style={{ fontFamily: "monospace", fontSize: 10, color: PALETTE.dim, marginTop: 4, paddingLeft: 88 }}>within the nucleus these renormalise to 100% before the draw</div>)}
    </div>
  );
}

function DiceRoll({ tokens, weights, onResult, color }) {
  const [rolling, setRolling] = useState(false);
  const timer = useRef(null);
  const roll = () => {
    if (rolling) return;
    setRolling(true);
    let ticks = 0;
    const spin = () => {
      ticks++;
      onResult(tokens[Math.floor(Math.random() * tokens.length)], false);
      if (ticks < 11) { timer.current = setTimeout(spin, 70 + ticks * 12); }
      else { const idx = weightedPick(weights); onResult(tokens[idx], true); setRolling(false); }
    };
    spin();
  };
  useEffect(() => () => clearTimeout(timer.current), []);
  return (
    <button onClick={roll} disabled={rolling} style={{ background: color, color: "#fff", border: "none", borderRadius: 3, padding: "9px 16px", fontSize: 14, fontWeight: 700, cursor: rolling ? "default" : "pointer", fontFamily: "inherit", whiteSpace: "nowrap" }}>
      {rolling ? "🎲 rolling…" : "🎲 Sample a token"}
    </button>
  );
}

function sliderStyle(color) {
  return { width: "100%", appearance: "none", WebkitAppearance: "none", height: 6, borderRadius: 3, background: `linear-gradient(90deg, ${color} 0%, ${color} 100%)`, outline: "none", cursor: "pointer" };
}

function TempScatter({ dist, temp }) {
  const W = 520, H = 340;
  const [rotY, setRotY] = useState(0.6);
  const rafRef = useRef(null);
  const dragRef = useRef(null);
  useEffect(() => { const step = () => { setRotY((r) => r + 0.003); rafRef.current = requestAnimationFrame(step); }; rafRef.current = requestAnimationFrame(step); return () => cancelAnimationFrame(rafRef.current); }, []);
  const onDown = (e) => { const pt = e.touches ? e.touches[0] : e; dragRef.current = { x: pt.clientX, r: rotY }; cancelAnimationFrame(rafRef.current); };
  const onMove = (e) => { if (!dragRef.current) return; const pt = e.touches ? e.touches[0] : e; setRotY(dragRef.current.r + (pt.clientX - dragRef.current.x) * 0.008); };
  const onUp = () => { dragRef.current = null; };
  const rotX = -0.3;
  const anchor = { x: 0.2, y: 0.15, z: 0.1 };
  const spread = 0.7 + Math.min(3.2, temp * 1.9);
  const pAnchor = project(anchor, rotY, rotX, W, H);
  const pts = dist.map((d) => { const pos = { x: anchor.x + d.off.x * spread, y: anchor.y + d.off.y * spread, z: anchor.z + d.off.z * spread }; return { ...d, pr: project(pos, rotY, rotX, W, H) }; }).sort((a, b) => b.pr.depth - a.pr.depth);
  return (
    <svg width="100%" viewBox={`0 0 ${W} ${H}`} style={{ display: "block", touchAction: "none", cursor: dragRef.current ? "grabbing" : "grab", marginBottom: 6 }}
      onMouseDown={onDown} onMouseMove={onMove} onMouseUp={onUp} onMouseLeave={onUp} onTouchStart={onDown} onTouchMove={onMove} onTouchEnd={onUp}>
      {pts.map((d, i) => (<line key={"l" + i} x1={pAnchor.sx} y1={pAnchor.sy} x2={d.pr.sx} y2={d.pr.sy} stroke={PALETTE.line} strokeWidth="0.8" opacity="0.4" />))}
      {pts.map((d, i) => {
        const isTop = d.tok === dist[0].tok;
        const r = (4 + d.p * 26) * d.pr.persp;
        const color = isTop ? PALETTE.apple : PALETTE.tech;
        return (<g key={i}>
          <circle cx={d.pr.sx} cy={d.pr.sy} r={r + 4} fill={color} opacity={0.14} />
          <circle cx={d.pr.sx} cy={d.pr.sy} r={Math.max(3, r)} fill={color} opacity={isTop ? 1 : 0.6} />
          <text x={d.pr.sx} y={d.pr.sy - Math.max(3, r) - 5} textAnchor="middle" fontFamily="monospace" fontSize="11" fontWeight={isTop ? 700 : 400} fill={PALETTE.ink} opacity={d.p > 0.02 || isTop ? 1 : 0.5}>{d.tok}</text>
        </g>);
      })}
      <text x={14} y={H - 12} fontFamily="monospace" fontSize="10" fill={PALETTE.dim}>“The Apple chip is ___” · dot size = probability</text>
    </svg>
  );
}

function Temperature() {
  const [temp, setTemp] = useState(1.0);
  const [rolled, setRolled] = useState(null);
  const [settled, setSettled] = useState(false);
  const dist = softmaxWithTemp(BASE_TOKENS, temp);
  const label = temp <= 0.2 ? "deterministic · always the top token" : temp < 0.9 ? "focused · sticks to safe choices" : temp <= 1.2 ? "balanced · the model's natural voice" : "adventurous · the tail comes alive";
  return (
    <div style={{ userSelect: "none" }}>
      <TempScatter dist={dist} temp={temp} />
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline", marginBottom: 6 }}>
        <span style={{ fontFamily: "monospace", fontSize: 12, color: PALETTE.dim }}>temperature</span>
        <span style={{ fontFamily: "monospace", fontSize: 22, fontWeight: 700, color: PALETTE.apple }}>{temp.toFixed(2)}</span>
      </div>
      <input type="range" min="0.05" max="2" step="0.05" value={temp} onChange={(e) => { setTemp(parseFloat(e.target.value)); setRolled(null); setSettled(false); }} style={sliderStyle(PALETTE.apple)} />
      <div style={{ display: "flex", justifyContent: "space-between", fontFamily: "monospace", fontSize: 10, color: PALETTE.dim, marginTop: 2, marginBottom: 14 }}><span>0 · rigid</span><span>1 · natural</span><span>2 · wild</span></div>
      <DistBars dist={dist} mode="temperature" rolledTok={rolled} />
      <div style={{ display: "flex", alignItems: "center", gap: 12, marginTop: 12 }}>
        <DiceRoll tokens={dist.map((d) => d.tok)} weights={dist.map((d) => d.p)} color={PALETTE.apple} onResult={(tok, final) => { setRolled(tok); setSettled(final); }} />
        <span style={{ fontFamily: "monospace", fontSize: 12, color: PALETTE.dim }}>{settled && rolled ? <>drew <b style={{ color: PALETTE.apple }}>{rolled}</b> — bigger bars win more often, not always</> : "weighted random draw over the bars"}</span>
      </div>
      <div style={{ fontFamily: "monospace", fontSize: 11, color: PALETTE.dim, textAlign: "center", marginTop: 8 }}>{label}</div>
    </div>
  );
}

function TopP() {
  const [p, setP] = useState(0.9);
  const [rolled, setRolled] = useState(null);
  const [settled, setSettled] = useState(false);
  const dist = softmaxWithTemp(BASE_TOKENS, 1.0);
  let cum = 0, cutoffIndex = 0;
  for (let i = 0; i < dist.length; i++) { cum += dist[i].p; cutoffIndex = i; if (cum >= p) break; }
  const keptCount = cutoffIndex + 1;
  const kept = dist.slice(0, keptCount);
  return (
    <div style={{ userSelect: "none" }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline", marginBottom: 6 }}>
        <span style={{ fontFamily: "monospace", fontSize: 12, color: PALETTE.dim }}>top-p (nucleus)</span>
        <span style={{ fontFamily: "monospace", fontSize: 22, fontWeight: 700, color: PALETTE.tech }}>{p.toFixed(2)}</span>
      </div>
      <input type="range" min="0.1" max="1" step="0.01" value={p} onChange={(e) => { setP(parseFloat(e.target.value)); setRolled(null); setSettled(false); }} style={sliderStyle(PALETTE.tech)} />
      <div style={{ display: "flex", justifyContent: "space-between", fontFamily: "monospace", fontSize: 10, color: PALETTE.dim, marginTop: 2, marginBottom: 14 }}><span>0.1 · narrow</span><span>1.0 · everything</span></div>
      <DistBars dist={dist} mode="topp" cutoffIndex={cutoffIndex} rolledTok={rolled} />
      <div style={{ display: "flex", alignItems: "center", gap: 12, marginTop: 12 }}>
        <DiceRoll tokens={kept.map((d) => d.tok)} weights={kept.map((d) => d.p)} color={PALETTE.tech} onResult={(tok, final) => { setRolled(tok); setSettled(final); }} />
        <span style={{ fontFamily: "monospace", fontSize: 12, color: PALETTE.dim }}>{settled && rolled ? <>drew <b style={{ color: PALETTE.apple }}>{rolled}</b> — only the {keptCount} kept token{keptCount > 1 ? "s are" : " is"} eligible</> : `sampled only from the ${keptCount}-token nucleus`}</span>
      </div>
      <div style={{ fontFamily: "monospace", fontSize: 11, color: PALETTE.dim, textAlign: "center", marginTop: 8 }}>nucleus = {keptCount} token{keptCount > 1 ? "s" : ""} kept · probabilities renormalised, the rest dropped</div>
    </div>
  );
}

function HighlightedPrompt({ prompt, active }) {
  const words = prompt.text.split(/(\s+)/);
  return (
    <span>
      {words.map((w, i) => {
        const clean = w.replace(/[^a-zA-Z]/g, "").toLowerCase();
        const isTrigger = active && prompt.triggers.includes(clean);
        return (<span key={i} style={{ background: isTrigger ? PALETTE.glow : "transparent", color: isTrigger ? PALETTE.ink : "inherit", fontWeight: isTrigger ? 700 : "inherit", borderRadius: 3, padding: isTrigger ? "1px 3px" : 0, transition: "background 0.3s" }}>{w}</span>);
      })}
    </span>
  );
}

// ===========================================================================
// PIPELINE — the whole journey at a glance, now reflecting the full deck.
// ===========================================================================
const PIPELINE_STAGES = [
  { n: 1, label: "Tokenise", note: "text → sub-word tokens (bytes underneath)", color: PALETTE.dim },
  { n: 2, label: "Embed", note: "each token → a learned vector (a point in space)", color: PALETTE.apple },
  { n: 3, label: "Add position", note: "rotate by position (RoPE) so order is encoded", color: PALETTE.violet },
  { n: 4, label: "Attention", note: "each token blends in others → contextual vectors", color: PALETTE.violet, deeper: "Q,K,V · heads · causal mask" },
  { n: 5, label: "MLP + repeat", note: "per-token knowledge layer; stack many blocks", color: PALETTE.tech, deeper: "residual stream · RMSNorm" },
  { n: 6, label: "Predict", note: "hidden vector · unembedding → logits → softmax", color: PALETTE.tech },
  { n: 7, label: "Sample", note: "temperature + top-p, then a weighted draw", color: PALETTE.fruit },
  { n: 8, label: "Loop", note: "append the token, repeat until a stop token", color: PALETTE.apple },
];

function Pipeline() {
  return (
    <div style={{ userSelect: "none" }}>
      <div style={{ display: "flex", flexDirection: "column", gap: 0 }}>
        {PIPELINE_STAGES.map((s, i) => (
          <div key={s.n}>
            <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
              <div style={{ width: 28, height: 28, borderRadius: "50%", flexShrink: 0, background: s.color, color: "#fff", display: "flex", alignItems: "center", justifyContent: "center", fontFamily: "monospace", fontSize: 13, fontWeight: 700 }}>{s.n}</div>
              <div style={{ flex: 1 }}>
                <div style={{ display: "flex", alignItems: "baseline", gap: 10, flexWrap: "wrap" }}>
                  <span style={{ fontFamily: "'Georgia', serif", fontSize: 16, fontWeight: 700 }}>{s.label}</span>
                  {s.deeper && <Tag>go deeper: {s.deeper}</Tag>}
                </div>
                <div style={{ fontSize: 12.5, color: PALETTE.dim, lineHeight: 1.4 }}>{s.note}</div>
              </div>
            </div>
            {i < PIPELINE_STAGES.length - 1 && (<div style={{ width: 2, height: 13, background: PALETTE.line, marginLeft: 13 }} />)}
          </div>
        ))}
      </div>
      <div style={{ marginTop: 16, padding: "12px 14px", background: "#f6efe1", border: `1px solid ${PALETTE.line}`, borderRadius: 4 }}>
        <div style={{ fontFamily: "monospace", fontSize: 10, textTransform: "uppercase", letterSpacing: 0.5, color: PALETTE.dim, marginBottom: 6 }}>still beyond this deck</div>
        <div style={{ fontSize: 13, lineHeight: 1.5, color: PALETTE.ink }}>
          We kept these light: <b>scaling laws</b> (how loss falls with params, data, and compute), <b>mixture-of-experts</b> (routing tokens to sub-networks), <b>quantisation & inference tricks</b> (smaller weights, paged KV cache), and <b>multimodality</b> (images and audio as tokens too).
        </div>
      </div>
    </div>
  );
}

// ===========================================================================
// SLIDES
// ===========================================================================
const SLIDES = [
  {
    kind: "text",
    title: "Words as points in space",
    body: (
      <>
        <p>A large language model never sees letters the way we do. Before it can reason about language, it turns every token into a <b>vector</b> — a long list of numbers that locates that token as a <b>point in a high-dimensional space</b>.</p>
        <p>Real models use thousands of dimensions — far too many to draw. We'll collapse it down to <b>three axes (x, y, z)</b> you can rotate.</p>
        <p style={{ color: PALETTE.dim }}>The big idea: <i>meaning becomes geometry.</i> Tokens used in similar ways end up near each other; unrelated ones end up far apart.</p>
      </>
    ),
  },
  {
    kind: "tokeniser",
    title: "First: text becomes tokens",
    body: (
      <>
        <p>One detail before the geometry. The model doesn't read whole words — it chops text into <Term k="token">tokens</Term>, sub-word chunks learned by an algorithm like <Term k="BPE">byte-pair encoding</Term>. A short common word is often one token; longer or rarer words split into several.</p>
        <p>Press <b>✂ Tokenise</b>. <i>apple</i> is one token while <i>apples</i> splits, and a leading space makes <i>“ apple”</i> a <b>different token id</b> from <i>“apple”</i>. The tokeniser works on raw <b>UTF-8 bytes</b>, so an emoji — four bytes — becomes several byte tokens. It's these tokens, not words, that get turned into vectors.</p>
        <p style={{ color: PALETTE.dim }}>We'll keep saying “word” for clarity, but under the hood it's always tokens. This is also why models charge and count length in tokens.</p>
      </>
    ),
  },
  {
    kind: "scene",
    title: "Three words, two neighbourhoods",
    body: (
      <>
        <p>Here are three points. <b style={{ color: PALETTE.tech }}>computer</b> sits in the tech neighbourhood. <b style={{ color: PALETTE.fruit }}>fruit</b> sits in the food neighbourhood. And <b style={{ color: PALETTE.apple }}>apple</b> — being ambiguous — floats <i>in between</i>.</p>
        <p>To compare two embeddings we can use <Term k="cosine similarity">cosine similarity</Term>: the cosine of the angle between their vectors. <code>1.0</code> means pointing the same way, <code>0</code> means unrelated. (It's a handy way to <i>read</i> the space — though, as we'll see, the model's own forward pass uses raw <Term k="dot product">dot products</Term>, not cosine.)</p>
        <p style={{ color: PALETTE.dim }}>Drag the chart to rotate it. Watch the two similarity bars below.</p>
      </>
    ),
  },
  {
    kind: "text",
    title: "How the numbers get there",
    body: (
      <>
        <p>Those coordinates aren't hand-placed. They're <b>learned</b> by a neural network trained on enormous amounts of text — nudged so tokens appearing in similar contexts get similar vectors. This starting vector is the token's <Term k="embedding">embedding</Term>.</p>
        <p>But one fixed point can't capture that <i>apple</i> means two things. Older embeddings (word2vec, GloVe) were <b>static</b> — one vector per word, forever. Transformers make embeddings <b>contextual</b>: as your prompt flows through <Term k="attention">attention</Term> layers, each token's vector is recomputed as a blend of itself and the tokens around it. <Tag>go deeper: attention · layers</Tag></p>
        <p style={{ color: PALETTE.dim }}>The result is a <b>contextual embedding</b>: <i>apple</i> stops floating in the middle and slides toward whichever neighbourhood the surrounding words belong to.</p>
      </>
    ),
  },
  {
    kind: "neuralnet",
    title: "Teach the model, document by document",
    body: (
      <>
        <p>The model learns what <b style={{ color: PALETTE.apple }}>apple</b> means by reading text. The five sentences below are just <b>illustrative examples</b>. Press <b>Train ▸</b> on one: its context words light up, a pulse fires through the network's <b>weights</b>, and the <b>embedding vector</b> shifts a little.</p>
        <p>No single example decides the meaning — it's the <b>accumulated pattern</b> across many. Then press <b>▶ Train on millions of documents</b>: it streams far more varied text and the embedding <b>converges</b> to a stable vector.</p>
        <p style={{ color: PALETTE.dim }}>Crucially, the documents themselves aren't kept anywhere. What survives training is the <b>weights</b> — a lossy compression of the patterns in all that text.</p>
      </>
    ),
  },
  {
    kind: "rope",
    title: "Telling tokens where they sit",
    body: (
      <>
        <p>Attention has a quirk: on its own it's <b>order-blind</b>. Shuffle the tokens and it computes the same thing — so “dog bites man” and “man bites dog” would look identical. The model needs position information injected.</p>
        <p>Most current models use <Term k="RoPE">RoPE</Term> — rotary position embedding. Each token's <Term k="query">query</Term> and <Term k="key">key</Term> vectors are <b>rotated</b> by an angle that grows with their position. The neat consequence: the score between two tokens depends only on the <i>gap</i> between them, so attention naturally feels <b>relative</b> distance.</p>
        <p>Drag the word to a new position. Its links to the other words re-weight — nearby words stay strongly connected, far ones fade. That's all position is: a rotation that makes closeness matter.</p>
        <p style={{ color: PALETTE.dim }}>Earlier approaches added fixed <b>sinusoidal</b> patterns or learned a lookup table of positions. Rotating instead of adding tends to stretch to longer sequences more gracefully.</p>
      </>
    ),
  },
  {
    kind: "attention",
    title: "Attention: how context gets in",
    body: (
      <>
        <p>This is the mechanism that makes embeddings contextual. Each word turns itself into three vectors: a <Term k="query"><b style={{ color: PALETTE.violet }}>query</b></Term> (what it's looking for), a <Term k="key"><b style={{ color: PALETTE.violet }}>key</b></Term> (what it offers), and a <Term k="value"><b style={{ color: PALETTE.violet }}>value</b></Term> (what it contributes). A word's query is matched against every key to get <b>attention weights</b> — then it pulls in a weighted blend of the values.</p>
        <p>The heatmap shows those weights. Read <b style={{ color: PALETTE.apple }}>apple</b>'s row: the heavy cells reveal which words it draws meaning from. Toggle the sentence and watch the weight land on tech words or orchard words. The greyed <b>× cells are a <Term k="causal mask">causal mask</Term></b> — a word can't attend to the future, which is what keeps generation left-to-right.</p>
        <p style={{ color: PALETTE.dim }}>Under the hood that matching is a <Term k="dot product">dot product</Term> of query and key, passed through <Term k="softmax">softmax</Term> to make the weights sum to 100%. Real models run many such heads in parallel (<b>multi-head attention</b>) — dozens of them — each free to specialise in a different relationship: grammar, references back to earlier words, copying.</p>
      </>
    ),
  },
  {
    kind: "block",
    title: "Stacking it into a tower",
    body: (
      <>
        <p>Attention is one half of a <b>transformer block</b>. The other is an <Term k="MLP">MLP</Term> (feed-forward network) that processes each token on its own. The two sit on a <Term k="residual stream">residual stream</Term>: each step reads a normalised copy and <b>adds</b> its result back — it never overwrites.</p>
        <p>Step through the walkthrough on the right. <b>Attention</b> mixes information <i>across</i> words; the <b>MLP</b> refines each word and is where most of the model's <b>knowledge</b> lives (roughly two-thirds of its parameters). Modern blocks normalise <i>before</i> each step (<Term k="pre-norm">pre-norm</Term>, using <Term k="RMSNorm">RMSNorm</Term>) for stable training.</p>
        <p style={{ color: PALETTE.dim }}>Then the block repeats — dozens of identical copies stacked up. Each pass refines the vector a little more: surface patterns early, meaning and prediction late.</p>
      </>
    ),
  },
  {
    kind: "interactive",
    title: "Context bends the vector",
    body: (
      <p>Now put it together. Pick a question. The context-carrying words get <b>highlighted</b>, the connecting tokens appear on the chart, and <b style={{ color: PALETTE.apple }}>apple</b>'s vector animates toward the neighbourhood that attention pulls it into. Watch the cosine bars react.</p>
    ),
  },
  {
    kind: "tokengen",
    title: "Answering, one token at a time",
    body: (
      <>
        <p>The model writes its answer by repeating one small loop. The question already fixed the <b>context</b> — “fast” and “chip” pin <i>Apple</i> to its tech meaning. Press <b>Step ▸</b> to walk the four beats: <b>feed</b>, <b>score</b>, <b>pick</b>, <b>append</b>. This is <Term k="autoregressive">autoregressive</Term> generation.</p>
        <p>The scoring step is the key part — and a common place explainers get it wrong. The model takes its final hidden vector and computes a <Term k="dot product"><b>dot product</b></Term> against every token's row of the unembedding matrix. Each result is a <Term k="logit">logit</Term>; <Term k="softmax">softmax</Term> turns the whole set into a probability over the vocabulary.</p>
        <p style={{ color: PALETTE.dim }}>Why does <i>chip</i> get followed by <i>is</i>? Because the space encodes <b>more than topic</b>. Function words sit in their own <b>grammatical region</b>; after a finished subject the vector expects a verb. <b>Topic</b> and <b>syntax</b> share one space, satisfied together each step.</p>
      </>
    ),
  },
  {
    kind: "text",
    title: "Why guessing the next word is enough",
    body: (
      <>
        <p>It's fair to wonder how something this simple — <i>repeatedly guess the next token</i> — produces answers that look intelligent. The trick is in the <b>training goal</b>.</p>
        <p>To predict the next token well across <b>trillions of tokens</b> of real text, the model is forced to absorb whatever makes that prediction possible: <b>grammar</b>, <b>facts</b>, <b>reasoning patterns</b>, translation, code, tone. Finishing “The capital of France is ___” correctly requires actually <i>knowing</i> it. The next-token objective quietly drags all of that into the weights.</p>
        <p style={{ color: PALETTE.dim }}>So nothing here stores facts in a lookup table. The “knowledge” lives in the same <b>vectors and weights</b> you've been watching — squeezed in by the pressure to predict what comes next.</p>
      </>
    ),
  },
  {
    kind: "temperature",
    title: "Temperature: how adventurous to be",
    body: (
      <>
        <p>The model has scored the candidates for “The Apple chip is ___”: <i>fast</i>, <i>powerful</i>, <i>blazing</i>, even <i>magical</i>. <b>Temperature</b> divides the logits before softmax, stretching or flattening the curve before we sample.</p>
        <p>Drag the slider. Near <b>0</b>, almost all probability piles onto the single best token — output is <b>deterministic</b>. Push it <b>higher</b> and the leading bar <b>shrinks</b> as mass spreads to the tail: far-fetched words get a real share, so output turns <b>creative</b>, sometimes incoherent.</p>
        <p style={{ color: PALETTE.dim }}>The <b>ranking never changes</b> — temperature only controls how tightly the candidates cluster on the front-runner. Hit <b>🎲 Sample a token</b> a few times: the pick is a <b>weighted random draw</b>, so bigger bars win more often but not every time.</p>
      </>
    ),
  },
  {
    kind: "topp",
    title: "Top-p: trimming the long tail",
    body: (
      <>
        <p>Temperature reshapes the whole curve; <b>top-p</b> (nucleus sampling) instead <b>cuts it off</b>. It sorts tokens by probability and keeps only the smallest set whose probabilities <b>add up to p</b> — then samples from just that nucleus.</p>
        <p>Drag the slider. At <b>p = 1.0</b> every token is eligible. Lower it and the long tail — <i>snappy</i>, <i>magical</i>, <i>edible</i> — gets <b>discarded</b> before sampling, so the model stays varied among strong options while never reaching for something absurd.</p>
        <p style={{ color: PALETTE.dim }}>Same <b>weighted dice roll</b>, but only across the survivors. Temperature and top-p usually combine: temperature sets <i>boldness</i>, top-p sets a <i>safety fence</i>.</p>
      </>
    ),
  },
  {
    kind: "training",
    title: "From raw predictor to assistant",
    body: (
      <>
        <p>Everything so far describes a <b>base model</b> — a brilliant autocomplete. It has knowledge but no instinct to answer questions or be helpful. Turning it into an assistant takes two more stages of training.</p>
        <p>Click through the rail. <b>Pretraining</b> learns language and facts from raw text. <b><Term k="SFT">Supervised fine-tuning</Term></b> teaches it to follow instructions from curated demonstrations. <b>Preference optimisation</b> (<Term k="RLHF">RLHF</Term> or the simpler <Term k="DPO">DPO</Term>) ranks competing answers and steers the model toward the preferred ones — tone, judgement, helpfulness.</p>
        <p style={{ color: PALETTE.dim }}>Same weights throughout, reshaped at each stage. The model you actually chat with is the aligned one; the base model underneath would just keep completing your sentence.</p>
      </>
    ),
  },
  {
    kind: "context",
    title: "How much it can see at once",
    body: (
      <>
        <p>The <b>context window</b> is the maximum number of tokens — prompt plus generation — the model can attend to at once. Everything outside it is invisible; there's no built-in memory beyond the window.</p>
        <p>Why not make it huge? Drag the slider. Full attention compares <b>every token with every other</b>, so cost grows with the <b>square</b> of the length — double the context, quadruple the work. The <Term k="KV cache">KV cache</Term> (storing past keys and values so each step is cheap) grows linearly but eats memory.</p>
        <p style={{ color: PALETTE.dim }}>Hence the tricks: <b>sliding-window</b> attention (each token sees only the last w), RoPE <b>interpolation</b> to stretch the trained length, and grouped-query attention to shrink the cache.</p>
      </>
    ),
  },
  {
    kind: "text",
    title: "So, in one breath",
    body: (
      <>
        <p>The model splits text into <b>tokens</b>, turns each into a <b>vector</b>, and marks its <b>position</b> by rotation. <b>Attention</b> blends in surrounding tokens to make each vector <b>contextual</b>; an <b>MLP</b> refines it and holds the knowledge; a tall <b>stack</b> of these blocks repeats the refinement.</p>
        <p>To speak, it projects the final vector against the whole vocabulary — a <b>dot product → softmax</b> — and gets a probability for every next token. <b>Temperature</b> and <b>top-p</b> shape that distribution, one token is <b>sampled</b>, appended, and the loop runs again.</p>
        <p style={{ color: PALETTE.dim }}>All of it learned by <b>predicting the next token</b> over trillions of words, then aligned into an assistant. Geometry first, language second.</p>
      </>
    ),
  },
  {
    kind: "pipeline",
    title: "The whole pipeline at a glance",
    body: (
      <>
        <p>Every answer runs through the same stages — each one something you saw in action earlier. Here's the full journey in one view.</p>
        <p style={{ color: PALETTE.dim }}>The dashed tags and the box at the end flag the parts we kept lightweight on purpose — good threads to pull on next.</p>
      </>
    ),
  },
];

// ===========================================================================
// APP
// ===========================================================================
export default function App() {
  const [slide, setSlide] = useState(0);
  const [activePromptId, setActivePromptId] = useState(null);
  const [isMobile, setIsMobile] = useState(false);

  useEffect(() => {
    const check = () => setIsMobile(window.innerWidth < 760);
    check();
    window.addEventListener("resize", check);
    return () => window.removeEventListener("resize", check);
  }, []);

  const cur = SLIDES[slide];
  const activePrompt = PROMPTS.find((p) => p.id === activePromptId) || null;

  useEffect(() => { if (cur.kind !== "interactive") setActivePromptId(null); }, [slide]);

  const hasVisual = ["scene", "interactive", "neuralnet", "tokengen", "temperature", "topp", "tokeniser", "pipeline", "rope", "attention", "block", "training", "context"].includes(cur.kind);
  const go = (d) => setSlide((s) => Math.max(0, Math.min(SLIDES.length - 1, s + d)));

  useEffect(() => {
    const onKey = (e) => { if (e.key === "ArrowRight") go(1); if (e.key === "ArrowLeft") go(-1); };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  return (
    <div style={{ minHeight: "100vh", background: PALETTE.paper, color: PALETTE.ink, fontFamily: "'Helvetica Neue', Arial, sans-serif", padding: "28px 20px", boxSizing: "border-box" }}>
      <div style={{ maxWidth: 1080, margin: "0 auto" }}>
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline", marginBottom: 20 }}>
          <div style={{ fontFamily: "monospace", fontSize: 12, letterSpacing: 2, color: PALETTE.dim, textTransform: "uppercase" }}>Apple in space · how LLMs read meaning</div>
          <div style={{ fontFamily: "monospace", fontSize: 12, color: PALETTE.dim }}>{slide + 1} / {SLIDES.length}</div>
        </div>

        <div style={{ display: hasVisual ? (isMobile ? "flex" : "grid") : "block", flexDirection: isMobile ? "column" : undefined, gridTemplateColumns: hasVisual && !isMobile ? "1fr 1fr" : undefined, gap: isMobile ? 18 : 28, alignItems: "start" }}>
          {hasVisual && (
            <div style={{ background: PALETTE.panel, border: `1px solid ${PALETTE.line}`, borderRadius: 4, padding: isMobile ? 12 : 18, order: isMobile ? 0 : 2, overflow: "visible", minHeight: isMobile ? 0 : 540, display: "flex", flexDirection: "column", justifyContent: "center" }}>
              {(cur.kind === "scene" || cur.kind === "interactive") && (<Scene activePrompt={activePrompt} autoRotate={!activePrompt} />)}
              {cur.kind === "neuralnet" && <NeuralNet />}
              {cur.kind === "tokengen" && <TokenGen />}
              {cur.kind === "temperature" && <Temperature />}
              {cur.kind === "topp" && <TopP />}
              {cur.kind === "tokeniser" && <Tokeniser />}
              {cur.kind === "pipeline" && <Pipeline />}
              {cur.kind === "rope" && <RoPE />}
              {cur.kind === "attention" && <Attention />}
              {cur.kind === "block" && <TransformerBlock />}
              {cur.kind === "training" && <Training />}
              {cur.kind === "context" && <ContextWindow />}
            </div>
          )}

          <div style={{ background: PALETTE.panel, border: `1px solid ${PALETTE.line}`, borderRadius: 4, padding: isMobile ? "22px 20px" : "30px 32px", minHeight: isMobile ? 0 : 540, order: isMobile ? 1 : 1, boxSizing: "border-box" }}>
            <h1 style={{ fontFamily: "'Georgia', serif", fontSize: isMobile ? 26 : 32, lineHeight: 1.12, margin: "0 0 18px", fontWeight: 700 }}>{cur.title}</h1>
            <div style={{ fontSize: isMobile ? 15 : 16, lineHeight: 1.6 }}>{cur.body}</div>

            {cur.kind === "interactive" && (
              <div style={{ marginTop: 20 }}>
                {PROMPTS.map((p) => {
                  const isActive = p.id === activePromptId;
                  const tgtColor = ANCHORS[p.target].color;
                  return (
                    <button key={p.id} onClick={() => setActivePromptId(isActive ? null : p.id)} style={{ display: "block", width: "100%", textAlign: "left", background: isActive ? "#fff" : "transparent", border: `1px solid ${isActive ? tgtColor : PALETTE.line}`, borderLeft: `4px solid ${isActive ? tgtColor : PALETTE.line}`, borderRadius: 3, padding: "11px 14px", marginBottom: 9, fontSize: 15, lineHeight: 1.4, cursor: "pointer", color: PALETTE.ink, fontFamily: "inherit", transition: "all 0.2s" }}>
                      <HighlightedPrompt prompt={p} active={isActive} />
                      {isActive && (<div style={{ marginTop: 8, fontSize: 13, color: PALETTE.dim, fontStyle: "italic" }}>{p.note}</div>)}
                    </button>
                  );
                })}
              </div>
            )}
          </div>
        </div>

        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginTop: 26 }}>
          <button onClick={() => go(-1)} disabled={slide === 0} style={navBtn(slide === 0)}>← Back</button>
          <div style={{ display: "flex", gap: 7, flexWrap: "wrap", justifyContent: "center", maxWidth: 420 }}>
            {SLIDES.map((_, i) => (<button key={i} onClick={() => setSlide(i)} aria-label={`Slide ${i + 1}`} style={{ width: 9, height: 9, borderRadius: "50%", border: "none", padding: 0, cursor: "pointer", background: i === slide ? PALETTE.apple : PALETTE.line }} />))}
          </div>
          <button onClick={() => go(1)} disabled={slide === SLIDES.length - 1} style={navBtn(slide === SLIDES.length - 1)}>Next →</button>
        </div>
      </div>
    </div>
  );
}
