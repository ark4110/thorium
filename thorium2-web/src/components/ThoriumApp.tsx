"use client";

import dynamic from "next/dynamic";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { fieldColor } from "@/lib/colors";
import { formatResults, type Results } from "@/lib/v1format";
import type { DrawMode, Mesh, View } from "./Viewer3D";

const Viewer3D = dynamic(() => import("./Viewer3D"), { ssr: false });

interface ModelInfo {
  id: string;
  name: string;
  example: boolean;
}
interface RunStatus {
  id: string;
  model: string;
  state: "running" | "done" | "failed" | "cancelled";
  started: string;
  tStart: number;
  tEnd: number;
  t: number;
  step: number;
  log: string[];
  error?: string;
}

async function api<T>(url: string, init?: RequestInit): Promise<T> {
  const r = await fetch(url, init);
  const body = await r.json().catch(() => ({ error: `HTTP ${r.status}` }));
  if (!r.ok) throw new Error(body.error ?? `HTTP ${r.status}`);
  return body as T;
}

const PALETTE = [0x4e79a7, 0xf28e2b, 0x59a14f, 0xb07aa1, 0x76b7b2, 0xedc948, 0xff9da7, 0x9c755f, 0xbab0ac, 0xe15759];
const fmtT = (v: number) => (Math.abs(v) >= 1e5 || (Math.abs(v) < 1e-2 && v !== 0) ? v.toExponential(3) : String(+v.toFixed(3)));

type Dialog = null | "open" | "upload" | "analysis" | "about";

export default function ThoriumApp() {
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [model, setModel] = useState<ModelInfo | null>(null);
  const [mesh, setMesh] = useState<Mesh | null>(null);
  const [summary, setSummary] = useState("");
  const [runs, setRuns] = useState<RunStatus[]>([]);
  const [run, setRun] = useState<RunStatus | null>(null);
  const [results, setResults] = useState<Results | null>(null);
  const [step, setStep] = useState(0);
  const [animate, setAnimate] = useState(false);
  const [strip, setStrip] = useState(true);
  const [entire, setEntire] = useState(true);
  const [mode, setMode] = useState<DrawMode>("contour");
  const [view, setView] = useState<View>({ dir: [1, 1, 1], n: 0 });
  const [selected, setSelected] = useState<{ e: number; x: number; y: number } | null>(null);
  const [dialog, setDialog] = useState<Dialog>(null);
  const [menu, setMenu] = useState<string | null>(null);
  const [error, setError] = useState("");
  const [busy, setBusy] = useState("");

  const loadModels = useCallback(() => api<ModelInfo[]>("/api/models").then(setModels).catch((e) => setError(String(e.message))), []);
  useEffect(() => {
    loadModels();
  }, [loadModels]);

  const openModel = async (m: ModelInfo, runId?: string) => {
    setDialog(null);
    setBusy(`Opening ${m.name}...`);
    setError("");
    setResults(null);
    setRun(null);
    setSelected(null);
    setAnimate(false);
    try {
      const r = await api<{ mesh: Mesh; summary: string }>(`/api/models/${m.id}/mesh`);
      setModel(m);
      setMesh(r.mesh);
      setSummary(r.summary);
      const rs = await api<RunStatus[]>(`/api/models/${m.id}/runs`);
      setRuns(rs);
      if (runId) await openResults(runId);
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy("");
    }
  };

  const openResults = async (runId: string) => {
    setBusy("Loading results...");
    try {
      const res = await api<Results>(`/api/runs/${runId}/results`);
      setResults(res);
      setStep(res.times.length - 1);
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy("");
    }
  };

  // Poll a running analysis.
  useEffect(() => {
    if (!run || run.state !== "running") return;
    const id = setInterval(async () => {
      try {
        const s = await api<RunStatus>(`/api/runs/${run.id}`);
        setRun(s);
        if (s.state === "done") {
          if (model) setRuns(await api<RunStatus[]>(`/api/models/${model.id}/runs`));
          await openResults(s.id);
        } else if (s.state !== "running") {
          setError(s.error ?? `Analysis ${s.state}`);
        }
      } catch (e) {
        setError((e as Error).message);
      }
    }, 1000);
    return () => clearInterval(id);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [run?.id, run?.state]);

  // Animation.
  useEffect(() => {
    if (!animate || !results) return;
    const id = setInterval(() => setStep((s) => (s + 1) % results.times.length), 150);
    return () => clearInterval(id);
  }, [animate, results]);

  // Colour scale limits.
  const range = useMemo(() => {
    if (!results) return null;
    const rows = entire ? results.T : [results.T[step]];
    let lo = Infinity,
      hi = -Infinity;
    for (const row of rows)
      for (const v of row) {
        lo = Math.min(lo, v);
        hi = Math.max(hi, v);
      }
    return { lo, hi };
  }, [results, step, entire]);

  // Element index -> results column.
  const column = useMemo(() => {
    if (!mesh || !results) return null;
    const pos = new Map(results.ids.map((id, k) => [id, k]));
    return mesh.elements.map((e) => pos.get(e.id) ?? -1);
  }, [mesh, results]);

  const colors = useMemo(() => {
    if (!mesh) return new Float32Array();
    const c = new Float32Array(mesh.elements.length * 3);
    mesh.elements.forEach((e, i) => {
      let rgb: [number, number, number];
      if (results && range && column && column[i] >= 0) rgb = fieldColor(results.T[step][column[i]], range.lo, range.hi);
      else {
        const p = PALETTE[Math.abs(e.pid) % PALETTE.length];
        rgb = [((p >> 16) & 255) / 255, ((p >> 8) & 255) / 255, (p & 255) / 255];
      }
      c.set(rgb, 3 * i);
    });
    return c;
  }, [mesh, results, range, column, step]);

  const startAnalysis = async (end: string, photons: string) => {
    if (!model) return;
    setDialog(null);
    setError("");
    setResults(null);
    try {
      const s = await api<RunStatus>(`/api/models/${model.id}/runs`, {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ end: end || undefined, photons: photons || undefined }),
      });
      setRun(s);
    } catch (e) {
      setError((e as Error).message);
    }
  };

  const stopAnalysis = async () => {
    if (!run) return;
    try {
      setRun(await api<RunStatus>(`/api/runs/${run.id}`, { method: "DELETE" }));
    } catch (e) {
      setError((e as Error).message);
    }
  };

  const saveTxt = () => {
    if (!results || !model) return;
    const blob = new Blob([formatResults(results)], { type: "text/plain" });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = `${model.name}.txt`;
    a.click();
    URL.revokeObjectURL(a.href);
  };

  const decimate = () => {
    if (!results) return;
    const answer = window.prompt("Leave only 1 of N time steps. N =", "10");
    const n = Number(answer);
    if (!answer || !Number.isInteger(n) || n < 2) return;
    const keep = results.times.map((_, k) => k).filter((k) => k % n === 0 || k === results.times.length - 1);
    setResults({ ids: results.ids, times: keep.map((k) => results.times[k]), T: keep.map((k) => results.T[k]), Q: keep.map((k) => results.Q[k]) });
    setStep(0);
  };

  const running = run?.state === "running";
  const progress = run ? Math.min(1, Math.max(0, (run.t - run.tStart) / (run.tEnd - run.tStart))) : 0;
  const sel = selected && mesh ? mesh.elements[selected.e] : null;
  const closeMenu = () => setMenu(null);

  return (
    <div className="flex h-screen flex-col bg-gray-200 text-sm text-gray-900" onClick={() => menu && setMenu(null)}>
      {/* Header: title and the menu of version 1 */}
      <header className="border-b border-gray-400 bg-gray-100" onClick={(e) => e.stopPropagation()}>
        <div className="bg-gradient-to-r from-blue-800 to-blue-500 px-2 py-0.5 text-white">
          T.H.O.R.I.U.M. 2{model ? ` — ${model.name}` : ""}
        </div>
        <nav className="flex">
          <Menu name="File" open={menu} setOpen={setMenu}>
            <MenuItem label="Open..." close={closeMenu} onClick={() => { loadModels(); setDialog("open"); }} />
            <MenuItem label="Upload model (*.bdf + *.top)..." close={closeMenu} onClick={() => setDialog("upload")} />
            <MenuItem label="Save results as txt" close={closeMenu} onClick={saveTxt} disabled={!results} />
          </Menu>
          <Menu name="Analysis" open={menu} setOpen={setMenu}>
            <MenuItem label="Thermal analysis" close={closeMenu} onClick={() => setDialog("analysis")} disabled={!mesh || running} />
          </Menu>
          <Menu name="Help" open={menu} setOpen={setMenu}>
            <MenuItem label="About" close={closeMenu} onClick={() => setDialog("about")} />
          </Menu>
        </nav>
      </header>

      <main className="relative flex min-h-0 flex-1">
        <section className="relative min-w-0 flex-1">
          {mesh ? (
            <Viewer3D mesh={mesh} view={view} colors={colors} mode={mode} selected={selected?.e ?? null} onPick={(e, x, y) => setSelected(e == null ? null : { e, x, y })} />
          ) : (
            <div className="flex h-full items-center justify-center text-gray-500">File → Open... to load a model</div>
          )}
          {results && range && strip && (
            <div className="absolute top-3 right-3 flex items-stretch gap-1 rounded bg-black/60 p-2 text-xs text-white" data-testid="legend">
              <div className="h-64 w-4" style={{ background: `linear-gradient(to top, ${[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1].map((k) => { const [r, g, b] = fieldColor(k, 0, 1); return `rgb(${r * 255},${g * 255},${b * 255}) ${k * 100}%`; }).join(",")})` }} />
              <div className="flex flex-col justify-between">
                {[1, 0.75, 0.5, 0.25, 0].map((k) => (
                  <span key={k}>{fmtT(range.lo + k * (range.hi - range.lo))} K</span>
                ))}
              </div>
            </div>
          )}
          {sel && selected && (
            <div className="absolute z-20 w-64 border border-gray-500 bg-yellow-50 p-2 text-xs shadow-lg" style={{ left: Math.min(selected.x + 12, 9999), top: selected.y + 12 }} data-testid="element-popup">
              <div className="mb-1 flex justify-between font-semibold">
                <span>{sel.type} {sel.id}</span>
                <button onClick={() => setSelected(null)}>✕</button>
              </div>
              <table className="w-full">
                <tbody>
                  <tr><td>Property / MEDIUM1</td><td className="text-right">{sel.pid} / {sel.medium}</td></tr>
                  {sel.thickness != null && <tr><td>Thickness, m</td><td className="text-right">{sel.thickness}</td></tr>}
                  {sel.area != null && <tr><td>Area, m²</td><td className="text-right">{fmtT(sel.area)}</td></tr>}
                  <tr><td>Volume, m³</td><td className="text-right">{fmtT(sel.volume)}</td></tr>
                  <tr><td>Mass, kg</td><td className="text-right">{fmtT(sel.mass)}</td></tr>
                  <tr><td>FTOP of faces</td><td className="text-right">{sel.optics.join(", ")}</td></tr>
                  <tr><td>Initial T, K</td><td className="text-right">{sel.t0}{sel.fixed ? " (T_SET)" : ""}</td></tr>
                  {sel.heat_source !== 0 && <tr><td>Heat source, W</td><td className="text-right">{fmtT(sel.heat_source)}</td></tr>}
                  {results && column && column[selected.e] >= 0 && (
                    <>
                      <tr><td>T at t = {fmtT(results.times[step])} s</td><td className="text-right font-semibold">{results.T[step][column[selected.e]]} K</td></tr>
                      <tr><td>Input heat power</td><td className="text-right">{results.Q[step][column[selected.e]]} W</td></tr>
                    </>
                  )}
                </tbody>
              </table>
            </div>
          )}
          {busy && <div className="absolute inset-x-0 top-0 bg-blue-100 px-2 py-1">{busy}</div>}
        </section>

        {/* Right panel: solution progress and result controls, as the expanded main window of version 1 */}
        {mesh && (
          <aside className="flex w-72 flex-col gap-3 overflow-y-auto border-l border-gray-400 bg-gray-100 p-3">
            {run && (run.state === "running" || run.state === "failed" || run.state === "cancelled") && (
              <div>
                <div className="mb-1 font-semibold">Solution</div>
                <div className="h-4 w-full border border-gray-500 bg-white">
                  <div className="h-full bg-green-600 transition-all" style={{ width: `${progress * 100}%` }} data-testid="progress" />
                </div>
                <div className="mt-1 text-xs">
                  {run.state === "running" ? `t = ${fmtT(run.t)} of ${fmtT(run.tEnd)} s, step ${run.step}` : `Analysis ${run.state}`}
                </div>
                {running && <button className="btn mt-1" onClick={stopAnalysis}>Stop analysis</button>}
              </div>
            )}
            {results ? (
              <div className="flex flex-col gap-2" data-testid="results-controls">
                <div className="font-semibold">Results</div>
                <div className="grid grid-cols-4 gap-1">
                  <button className="btn" title="To first step" onClick={() => setStep(0)}>|◀</button>
                  <button className="btn" title="Previous step" onClick={() => setStep((s) => Math.max(0, s - 1))}>◀</button>
                  <button className="btn" title="Next step" onClick={() => setStep((s) => Math.min(results.times.length - 1, s + 1))}>▶</button>
                  <button className="btn" title="To last step" onClick={() => setStep(results.times.length - 1)}>▶|</button>
                </div>
                <select className="border border-gray-500 bg-white p-1" value={step} onChange={(e) => setStep(Number(e.target.value))} aria-label="Step">
                  {results.times.map((t, k) => (
                    <option key={k} value={k}>Step {k}: t = {fmtT(t)} s</option>
                  ))}
                </select>
                <div className="grid grid-cols-2 gap-1">
                  <button className="btn" onClick={() => setAnimate(true)} disabled={animate}>Animate!</button>
                  <button className="btn" onClick={() => setAnimate(false)} disabled={!animate}>Stop!</button>
                </div>
                <label className="flex items-center gap-2"><input type="checkbox" checked={strip} onChange={(e) => setStrip(e.target.checked)} /> Value strip</label>
                <label className="flex items-center gap-2"><input type="radio" checked={!entire} onChange={() => setEntire(false)} /> Colors from current step</label>
                <label className="flex items-center gap-2"><input type="radio" checked={entire} onChange={() => setEntire(true)} /> Colors from entire solution</label>
                <select className="border border-gray-500 bg-white p-1" value={mode} onChange={(e) => setMode(e.target.value as DrawMode)} aria-label="Drawing mode">
                  <option value="contour">With contour</option>
                  <option value="plain">Without contour</option>
                  <option value="wireframe">Wireframe</option>
                </select>
                <button className="btn" onClick={decimate}>Decimate results</button>
                <button className="btn" onClick={saveTxt}>Save results as txt</button>
              </div>
            ) : (
              <div className="text-xs text-gray-600">
                <div className="mb-1 font-semibold text-gray-900">Model</div>
                Colours show element properties (PID). Click an element to see its properties. Analysis → Thermal analysis to solve.
                <select className="mt-2 w-full border border-gray-500 bg-white p-1" value={mode} onChange={(e) => setMode(e.target.value as DrawMode)} aria-label="Drawing mode">
                  <option value="contour">With contour</option>
                  <option value="plain">Without contour</option>
                  <option value="wireframe">Wireframe</option>
                </select>
              </div>
            )}
            <div>
              <div className="mb-1 font-semibold">View</div>
              <div className="grid grid-cols-7 gap-1">
                {([["Iso", [1, 1, 1]], ["+X", [1, 0, 0]], ["−X", [-1, 0, 0]], ["+Y", [0, 1, 0]], ["−Y", [0, -1, 0]], ["+Z", [0, 0, 1]], ["−Z", [0, 0, -1]]] as [string, [number, number, number]][]).map(([label, dir]) => (
                  <button key={label} className="btn px-0" title={`Look from ${label}`} onClick={() => setView((v) => ({ dir, n: v.n + 1 }))}>{label}</button>
                ))}
              </div>
            </div>
            {runs.filter((r) => r.state === "done").length > 0 && (
              <div className="text-xs">
                <div className="mb-1 font-semibold text-sm">Earlier analyses</div>
                {runs.filter((r) => r.state === "done").map((r) => (
                  <button key={r.id} className="block w-full truncate text-left text-blue-700 underline" onClick={() => openResults(r.id)}>
                    {new Date(r.started).toLocaleString()} — to {fmtT(r.tEnd)} s
                  </button>
                ))}
              </div>
            )}
            <details className="text-xs">
              <summary className="cursor-pointer font-semibold text-sm">Model check</summary>
              <pre className="mt-1 whitespace-pre-wrap break-words">{summary}</pre>
              {run && run.log.length > 0 && <pre className="mt-1 whitespace-pre-wrap break-words">{run.log.join("\n")}</pre>}
            </details>
          </aside>
        )}

        {dialog && (
          <div className="absolute inset-0 z-40 flex items-center justify-center bg-black/30" onClick={() => setDialog(null)}>
            <div className="w-[28rem] border border-gray-500 bg-gray-100 shadow-xl" onClick={(e) => e.stopPropagation()}>
              <div className="flex justify-between bg-gradient-to-r from-blue-800 to-blue-500 px-2 py-0.5 text-white">
                <span>{{ open: "Open", upload: "Upload model", analysis: "Thermal analysis", about: "About" }[dialog]}</span>
                <button onClick={() => setDialog(null)}>✕</button>
              </div>
              <div className="p-3">
                {dialog === "open" && (
                  <ul className="max-h-80 overflow-y-auto">
                    {models.map((m) => (
                      <li key={m.id}>
                        <button className="w-full px-2 py-1 text-left hover:bg-blue-600 hover:text-white" onClick={() => openModel(m)}>
                          {m.name} <span className="text-xs opacity-70">{m.example ? "(example of version 1)" : ""}</span>
                        </button>
                      </li>
                    ))}
                  </ul>
                )}
                {dialog === "upload" && <UploadForm onDone={async (m) => { await loadModels(); openModel(m); }} onError={setError} />}
                {dialog === "analysis" && mesh && <AnalysisForm mesh={mesh} onStart={startAnalysis} />}
                {dialog === "about" && (
                  <p>
                    Thorium 2 — transient radiation-conduction heat transfer: one finite element is one thermal node,
                    radiation by the Monte Carlo method, implicit BDF2 integration. Input: NASTRAN *.bdf and *.top of
                    T.H.O.R.I.U.M. 1 (A. Shaenko); results in the text format of version 1. GPL.
                  </p>
                )}
              </div>
            </div>
          </div>
        )}
      </main>

      {/* Footer: status line */}
      <footer className="flex justify-between border-t border-gray-400 bg-gray-100 px-2 py-0.5 text-xs">
        <span className={error ? "text-red-700" : ""} data-testid="status">
          {error ? `Error: ${error}` : running ? `Thermal analysis: ${Math.round(progress * 100)}%` : results ? `Step ${step} of ${results.times.length - 1}, t = ${fmtT(results.times[step])} s` : mesh ? `${mesh.elements.length} elements` : "Ready"}
        </span>
        <span>Thorium 2</span>
      </footer>
    </div>
  );
}

function MenuItem({ label, onClick, disabled, close }: { label: string; onClick: () => void; disabled?: boolean; close: () => void }) {
  return (
    <button
      className="block w-full px-4 py-1 text-left hover:bg-blue-600 hover:text-white disabled:text-gray-400 disabled:hover:bg-transparent"
      disabled={disabled}
      onClick={() => {
        close();
        onClick();
      }}
    >
      {label}
    </button>
  );
}

function Menu({ name, open, setOpen, children }: { name: string; open: string | null; setOpen: (m: string | null) => void; children: React.ReactNode }) {
  return (
    <div className="relative">
      <button className={`px-3 py-0.5 ${open === name ? "bg-blue-600 text-white" : "hover:bg-gray-300"}`} onClick={() => setOpen(open === name ? null : name)}>
        {name}
      </button>
      {open === name && <div className="absolute left-0 z-30 min-w-56 border border-gray-500 bg-gray-100 py-1 shadow-lg">{children}</div>}
    </div>
  );
}

function UploadForm({ onDone, onError }: { onDone: (m: ModelInfo) => void; onError: (e: string) => void }) {
  const ref = useRef<HTMLFormElement>(null);
  const [sending, setSending] = useState(false);
  return (
    <form
      ref={ref}
      className="flex flex-col gap-2"
      onSubmit={async (e) => {
        e.preventDefault();
        setSending(true);
        try {
          const m = await api<ModelInfo>("/api/models", { method: "POST", body: new FormData(ref.current!) });
          onDone(m);
        } catch (err) {
          onError((err as Error).message);
        } finally {
          setSending(false);
        }
      }}
    >
      <label>Model (*.bdf, *.dat) <input name="bdf" type="file" accept=".bdf,.dat,.BDF,.DAT" required className="block" /></label>
      <label>Properties (*.top) <input name="top" type="file" accept=".top,.TOP" required className="block" /></label>
      <button className="btn self-end" disabled={sending}>{sending ? "Uploading..." : "Upload and open"}</button>
    </form>
  );
}

function AnalysisForm({ mesh, onStart }: { mesh: Mesh; onStart: (end: string, photons: string) => void }) {
  const [end, setEnd] = useState(String(mesh.time.end));
  const [photons, setPhotons] = useState(String(mesh.photons));
  const steps = Math.ceil((Number(end) - mesh.time.start) / mesh.time.step);
  return (
    <form className="flex flex-col gap-2" onSubmit={(e) => (e.preventDefault(), onStart(end, photons))}>
      <div>I_PARAM: t = {mesh.time.start} … {mesh.time.end} s, step {mesh.time.step} s, N_PH = {mesh.photons}</div>
      <label className="flex items-center justify-between gap-2">End time, s <input className="w-40 border border-gray-500 p-1" value={end} onChange={(e) => setEnd(e.target.value)} /></label>
      <label className="flex items-center justify-between gap-2">Photons per face and step <input className="w-40 border border-gray-500 p-1" value={photons} onChange={(e) => setPhotons(e.target.value)} /></label>
      <div className="text-xs text-gray-600">{Number.isFinite(steps) && steps > 0 ? `${steps} steps` : "check the end time"}</div>
      <button className="btn self-end">Start</button>
    </form>
  );
}
