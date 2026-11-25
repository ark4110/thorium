// Models, viewer geometry and analysis runs on the server. Everything lives under data/.
import { spawn, type ChildProcess } from "node:child_process";
import { randomBytes } from "node:crypto";
import { existsSync } from "node:fs";
import fs from "node:fs/promises";
import path from "node:path";
import { safeModelId } from "../ids";
import { parseResults, type Results } from "../v1format";

const ROOT = process.env.THORIUM_WEB_ROOT ?? process.cwd();
const BIN = process.env.THORIUM2_BIN ?? path.join(ROOT, "bin", "thorium2");
const DATA = path.join(ROOT, "data");
const MODELS = path.join(DATA, "models");
const RUNS = path.join(DATA, "runs");
const EXAMPLES = process.env.THORIUM2_EXAMPLES ?? path.join(ROOT, "..", "thorium2", "tests", "data", "v1_examples");
const MAX_FILE = 50 * 1024 * 1024;

export class UserError extends Error {}

export interface ModelInfo {
  id: string;
  name: string;
  example: boolean;
  created: string;
}

const ID = /^[A-Za-z0-9_-]{1,64}$/;

function modelDir(id: string): string {
  if (!ID.test(id)) throw new UserError("bad model name");
  const dir = path.join(MODELS, id);
  if (!existsSync(path.join(dir, "model.bdf"))) throw new UserError(`model ${id} not found`);
  return dir;
}

let examplesReady: Promise<void> | null = null;
function ensureExamples(): Promise<void> {
  examplesReady ??= (async () => {
    await fs.mkdir(MODELS, { recursive: true });
    if (!existsSync(/*turbopackIgnore: true*/ EXAMPLES)) return;
    for (const name of await fs.readdir(/*turbopackIgnore: true*/ EXAMPLES)) {
      const src = path.join(EXAMPLES, name);
      const id = safeModelId(name);
      const dir = path.join(MODELS, id);
      if (!id || existsSync(dir) || !existsSync(path.join(src, `${name}.bdf`))) continue;
      await fs.mkdir(dir, { recursive: true });
      await fs.copyFile(path.join(src, `${name}.bdf`), path.join(dir, "model.bdf"));
      await fs.copyFile(path.join(src, `${name}.top`), path.join(dir, "model.top"));
      const meta: ModelInfo = { id, name, example: true, created: new Date().toISOString() };
      await fs.writeFile(path.join(dir, "meta.json"), JSON.stringify(meta));
    }
  })();
  return examplesReady;
}

export async function listModels(): Promise<ModelInfo[]> {
  await ensureExamples();
  const out: ModelInfo[] = [];
  for (const id of await fs.readdir(MODELS)) {
    try {
      out.push(JSON.parse(await fs.readFile(path.join(MODELS, id, "meta.json"), "utf8")));
    } catch {
      /* not a model folder */
    }
  }
  return out.sort((a, b) => Number(b.example) - Number(a.example) || a.name.localeCompare(b.name));
}

export async function saveModel(name: string, bdf: Buffer, top: Buffer): Promise<ModelInfo> {
  await ensureExamples();
  if (bdf.length > MAX_FILE || top.length > MAX_FILE) throw new UserError("file is larger than 50 MB");
  const base = safeModelId(name) || "model";
  let id = base;
  for (let k = 2; existsSync(path.join(MODELS, id)); k++) id = `${base}_${k}`;
  const dir = path.join(MODELS, id);
  await fs.mkdir(dir, { recursive: true });
  await fs.writeFile(path.join(dir, "model.bdf"), bdf);
  await fs.writeFile(path.join(dir, "model.top"), top);
  const meta: ModelInfo = { id, name: name.replace(/\.(bdf|dat)$/i, ""), example: false, created: new Date().toISOString() };
  await fs.writeFile(path.join(dir, "meta.json"), JSON.stringify(meta));
  return meta;
}

function execThorium(args: string[]): Promise<{ code: number; stdout: string; stderr: string }> {
  return new Promise((resolve, reject) => {
    const child = spawn(/*turbopackIgnore: true*/ BIN, args, { stdio: ["ignore", "pipe", "pipe"] });
    let stdout = "",
      stderr = "";
    child.stdout.on("data", (d) => (stdout += d));
    child.stderr.on("data", (d) => (stderr += d));
    child.on("error", reject);
    child.on("close", (code) => resolve({ code: code ?? -1, stdout, stderr }));
  });
}

/** Geometry for the viewer (cached next to the model) and the summary printed by thorium2 --check. */
export async function getMesh(id: string): Promise<{ mesh: unknown; summary: string }> {
  const dir = modelDir(id);
  const meshFile = path.join(dir, "mesh.json");
  const r = await execThorium([path.join(dir, "model.bdf"), "--top", path.join(dir, "model.top"), "--check", "--mesh", meshFile]);
  if (r.code !== 0) throw new UserError((r.stderr || r.stdout).replaceAll(dir + "/", "").trim());
  return { mesh: JSON.parse(await fs.readFile(meshFile, "utf8")), summary: r.stdout.replaceAll(dir + "/", "").trim() };
}

// ------------------------------------------------------------------ runs

export interface RunStatus {
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

interface Run extends RunStatus {
  child?: ChildProcess;
}

const g = globalThis as unknown as { thoriumRuns?: Map<string, Run> };
const runs = (g.thoriumRuns ??= new Map<string, Run>());

export interface RunOptions {
  end?: number;
  photons?: number;
}

export async function startRun(model: string, opts: RunOptions): Promise<RunStatus> {
  const dir = modelDir(model);
  for (const r of runs.values()) if (r.state === "running") throw new UserError("another analysis is running; wait or stop it");
  const { mesh } = (await getMesh(model)) as { mesh: { time: { start: number; end: number; step: number } } };
  const tStart = mesh.time.start;
  const tEnd = opts.end ?? mesh.time.end;
  if (!(tEnd > tStart)) throw new UserError("end time must be after the start time");
  const steps = Math.ceil((tEnd - tStart) / mesh.time.step);
  const every = Math.max(1, Math.ceil(steps / 1000)); // at most ~1000 time points for the browser
  const id = `${new Date().toISOString().replace(/[-:T]/g, "").slice(0, 14)}_${randomBytes(3).toString("hex")}`;
  const runDir = path.join(RUNS, id);
  await fs.mkdir(runDir, { recursive: true });
  const args = [path.join(dir, "model.bdf"), "--top", path.join(dir, "model.top"), "--out", path.join(runDir, "results.txt"),
    "--balance", path.join(runDir, "balance.txt"), "--every", String(every), "--end", String(tEnd)];
  if (opts.photons) args.push("--photons", String(Math.round(opts.photons)));
  const run: Run = { id, model, state: "running", started: new Date().toISOString(), tStart, tEnd, t: tStart, step: 0, log: [] };
  runs.set(id, run);
  const child = spawn(/*turbopackIgnore: true*/ BIN, args, { stdio: ["ignore", "pipe", "pipe"] });
  run.child = child;
  let buffer = "";
  const onText = (d: Buffer) => {
    buffer += d.toString();
    let nl: number;
    while ((nl = buffer.indexOf("\n")) >= 0) {
      const line = buffer.slice(0, nl).replaceAll(dir + "/", "").replaceAll(runDir + "/", "");
      buffer = buffer.slice(nl + 1);
      const m = /^t = (\S+)\s+step (\d+)/.exec(line);
      if (m) {
        run.t = Number(m[1]);
        run.step = Number(m[2]);
      } else if (line.trim()) {
        run.log.push(line);
        if (run.log.length > 200) run.log.shift();
      }
    }
  };
  child.stdout.on("data", onText);
  child.stderr.on("data", onText);
  child.on("error", (e) => {
    run.state = "failed";
    run.error = String(e);
  });
  child.on("close", async (code, signal) => {
    run.child = undefined;
    if (run.state === "running") {
      run.state = code === 0 ? "done" : signal ? "cancelled" : "failed";
      if (code !== 0 && !signal) run.error = run.log.filter((l) => l.startsWith("error")).join("\n") || `exit code ${code}`;
      if (code === 0) run.t = tEnd;
    }
    const { child: _c, ...status } = run;
    void _c;
    await fs.writeFile(path.join(runDir, "status.json"), JSON.stringify(status)).catch(() => {});
  });
  return publicStatus(run);
}

function publicStatus(r: Run): RunStatus {
  const { child: _c, ...s } = r;
  void _c;
  return s;
}

export async function getRun(id: string): Promise<RunStatus> {
  if (!ID.test(id)) throw new UserError("bad run id");
  const r = runs.get(id);
  if (r) return publicStatus(r);
  try {
    return JSON.parse(await fs.readFile(path.join(RUNS, id, "status.json"), "utf8"));
  } catch {
    throw new UserError(`run ${id} not found`);
  }
}

export async function listRuns(model: string): Promise<RunStatus[]> {
  await fs.mkdir(RUNS, { recursive: true });
  const out: RunStatus[] = [];
  for (const id of await fs.readdir(RUNS)) {
    try {
      const s = await getRun(id);
      if (s.model === model) out.push(s);
    } catch {
      /* skip */
    }
  }
  return out.sort((a, b) => b.started.localeCompare(a.started));
}

export function cancelRun(id: string): void {
  const r = runs.get(id);
  if (!r || r.state !== "running" || !r.child) throw new UserError("this analysis is not running");
  r.state = "cancelled";
  r.child.kill("SIGTERM");
}

export async function getResults(id: string): Promise<Results> {
  const s = await getRun(id);
  if (s.state !== "done") throw new UserError("the analysis has not finished");
  return parseResults(await fs.readFile(path.join(RUNS, id, "results.txt"), "utf8"));
}
