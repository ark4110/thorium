// Colour of a value, ported from TOP_solver.Field_color of T.H.O.R.I.U.M. 1.
// Returns [r, g, b] in 0..1.

function ramp(value: number, v0: number, v1: number, v2: number, v3: number, lo: number, hi: number, end: number): number {
  if (v0 <= value && value < v1) return ((hi - lo) / (v1 - v0)) * (value - v0) + lo;
  if (v1 <= value && value <= v2) return hi;
  if (v2 <= value && value <= v3) return ((end - hi) / (v3 - v2)) * (value - v2) + hi;
  return 0;
}

const clean = (x: number) => (Math.abs(x) < 1e-12 ? 0 : x);

export function fieldColor(value: number, min: number, max: number): [number, number, number] {
  if (!(max > min)) return fieldColor(0.5, 0, 1); // constant field: middle of the scale
  const d = max - min;
  const at = (k: number) => min + k * d;
  const blue = ramp(value, at(0), at(0.3), at(0.3), at(0.4), 0.3, 0.5, 0);
  const green = ramp(value, at(0.1), at(0.4), at(0.65), at(1), 0, 0.7, 0);
  const red = ramp(value, at(0.4), at(0.75), at(100), at(1000), 0, 1, 0);
  return [clean(red), clean(green), clean(blue)];
}
