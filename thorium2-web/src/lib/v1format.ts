// Results file of T.H.O.R.I.U.M. 1 / Thorium 2 (option output_txt):
//   Element_number Time,s Temperature,K  Input_heat_power,W
//   <element> <time> <T> <Q>          lines grouped by element, then by time

export interface Results {
  ids: number[];
  times: number[];
  /** T[step][element] */
  T: number[][];
  /** Q[step][element] — net heat power into the element, W */
  Q: number[][];
}

export function parseResults(text: string): Results {
  const lines = text.split(/\r?\n/);
  if (!lines[0] || !lines[0].startsWith("Element_number")) throw new Error("not a results file (no Element_number header)");
  const ids: number[] = [];
  const perElement: { t: number[]; T: number[]; Q: number[] }[] = [];
  for (let i = 1; i < lines.length; i++) {
    const line = lines[i].trim();
    if (!line) continue;
    const f = line.split(/\s+/);
    if (f.length !== 4) throw new Error(`line ${i + 1}: expected 4 values`);
    const [id, t, T, Q] = f.map(Number);
    if (![id, t, T, Q].every(Number.isFinite)) throw new Error(`line ${i + 1}: not a number`);
    if (ids.length === 0 || ids[ids.length - 1] !== id) {
      ids.push(id);
      perElement.push({ t: [], T: [], Q: [] });
    }
    const e = perElement[perElement.length - 1];
    e.t.push(t);
    e.T.push(T);
    e.Q.push(Q);
  }
  if (ids.length === 0) throw new Error("the results file has no data");
  const times = perElement[0].t;
  for (let k = 1; k < perElement.length; k++)
    if (perElement[k].t.length !== times.length) throw new Error(`element ${ids[k]} has a different number of steps`);
  const T = times.map((_, s) => perElement.map((e) => e.T[s]));
  const Q = times.map((_, s) => perElement.map((e) => e.Q[s]));
  return { ids, times, T, Q };
}

/** Math.Round(v, 6) (half to even) + .NET Double.ToString(): as written by version 1 and Thorium 2. */
export function v1Number(v: number): string {
  const x = v * 1e6;
  const f = Math.floor(x);
  const d = x - f;
  const n = d > 0.5 ? f + 1 : d < 0.5 ? f : f % 2 === 0 ? f : f + 1;
  let r = n / 1e6;
  if (r === 0) r = 0; // no "-0"
  if (r === 0) return "0";
  const [mant, expText] = r.toExponential(14).split("e");
  const exp = Number(expText);
  const strip = (s: string) => (s.includes(".") ? s.replace(/0+$/, "").replace(/\.$/, "") : s);
  if (exp < -4 || exp >= 15) {
    const sign = exp < 0 ? "-" : "+";
    return `${strip(mant)}E${sign}${String(Math.abs(exp)).padStart(2, "0")}`;
  }
  return strip(r.toPrecision(15));
}

/** The results file text (CRLF line ends, as version 1 wrote it on Windows). */
export function formatResults(r: Results): string {
  const lines = ["Element_number Time,s Temperature,K  Input_heat_power,W"];
  r.ids.forEach((id, e) => {
    r.times.forEach((t, s) => lines.push(`${id} ${v1Number(t)} ${v1Number(r.T[s][e])} ${v1Number(r.Q[s][e])}`));
  });
  return lines.join("\r\n") + "\r\n";
}
