import { describe, expect, it } from "vitest";
import { parseResults } from "../v1format";
import { fieldColor } from "../colors";
import { checkBasicAuth } from "../auth";
import { safeModelId } from "../ids";

describe("parseResults (output_txt of T.H.O.R.I.U.M. 1)", () => {
  it("reads lines grouped by element into a step x element table", () => {
    const text =
      "Element_number Time,s Temperature,K  Input_heat_power,W\r\n" +
      "1 0 300 0\r\n1 10 301.5 12\r\n2 0 300 0\r\n2 10 299.25 -3.5\r\n";
    const r = parseResults(text);
    expect(r.ids).toEqual([1, 2]);
    expect(r.times).toEqual([0, 10]);
    expect(r.T).toEqual([
      [300, 300],
      [301.5, 299.25],
    ]);
    expect(r.Q[1]).toEqual([12, -3.5]);
  });
  it("accepts exponents and rejects broken files", () => {
    const r = parseResults("Element_number Time,s Temperature,K  Input_heat_power,W\n7 1E-06 300 2.5E+20\n");
    expect(r.times[0]).toBe(1e-6);
    expect(r.Q[0][0]).toBe(2.5e20);
    expect(() => parseResults("1 0 300\n")).toThrow();
    expect(() =>
      parseResults("Element_number Time,s Temperature,K  Input_heat_power,W\n1 0 300 0\n1 5 300 0\n2 0 300 0\n"),
    ).toThrow(/steps/);
  });
});

describe("fieldColor (Field_color of version 1)", () => {
  it("goes from dark blue through green to red", () => {
    expect(fieldColor(0, 0, 1)).toEqual([0, 0, 0.3]);
    expect(fieldColor(1, 0, 1)).toEqual([1, 0, 0]);
    const [r, g, b] = fieldColor(0.5, 0, 1);
    expect(g).toBeCloseTo(0.7);
    expect(b).toBe(0);
    expect(r).toBeCloseTo(((1 - 0) / (0.75 - 0.4)) * (0.5 - 0.4));
  });
  it("handles a constant field", () => {
    expect(fieldColor(300, 300, 300)).toEqual(fieldColor(0.5, 0, 1));
  });
});

describe("checkBasicAuth", () => {
  const header = (u: string, p: string) => "Basic " + Buffer.from(`${u}:${p}`).toString("base64");
  it("accepts the password with any user name", () => {
    expect(checkBasicAuth(header("thorium", "s3cret"), "s3cret")).toBe(true);
    expect(checkBasicAuth(header("", "s3cret"), "s3cret")).toBe(true);
  });
  it("rejects everything else", () => {
    expect(checkBasicAuth(header("thorium", "wrong"), "s3cret")).toBe(false);
    expect(checkBasicAuth(null, "s3cret")).toBe(false);
    expect(checkBasicAuth("Bearer x", "s3cret")).toBe(false);
    expect(checkBasicAuth(header("a", ""), "")).toBe(false); // no password configured: nobody gets in
  });
});

describe("safeModelId", () => {
  it("keeps letters, digits, dash and underscore", () => {
    expect(safeModelId("Moon rover (v2).bdf")).toBe("Moon_rover_v2");
    expect(safeModelId("../../etc/passwd")).toBe("etc_passwd");
    expect(safeModelId("...")).toBe("");
  });
});

import { formatResults, v1Number } from "../v1format";

describe("v1Number / formatResults (same cases as the C++ writer)", () => {
  it("writes numbers as Math.Round(v, 6) + Double.ToString()", () => {
    expect(v1Number(300)).toBe("300");
    expect(v1Number(-0.0000001)).toBe("0");
    expect(v1Number(301.12345678)).toBe("301.123457");
    expect(v1Number(0.0000025)).toBe("2E-06");
    expect(v1Number(0.000123)).toBe("0.000123");
    expect(v1Number(0.0000123)).toBe("1.2E-05");
    expect(v1Number(1e7)).toBe("10000000");
    expect(v1Number(3.85e26)).toBe("3.85E+26");
  });
  it("round-trips a results table", () => {
    const text =
      "Element_number Time,s Temperature,K  Input_heat_power,W\r\n1 0 300 0\r\n1 10 301.5 12\r\n2 0 300 0\r\n2 10 299.25 -3.5\r\n";
    expect(formatResults(parseResults(text))).toBe(text);
  });
});
