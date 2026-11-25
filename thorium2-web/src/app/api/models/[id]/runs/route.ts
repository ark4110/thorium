import { connection, type NextRequest } from "next/server";
import { handle } from "@/lib/server/http";
import { listRuns, startRun, UserError } from "@/lib/server/store";

export async function GET(_req: NextRequest, ctx: RouteContext<"/api/models/[id]/runs">) {
  await connection();
  const { id } = await ctx.params;
  return handle(() => listRuns(id));
}

/** Starts "Analysis / Thermal analysis". Body: { end?: number, photons?: number }. */
export async function POST(req: NextRequest, ctx: RouteContext<"/api/models/[id]/runs">) {
  const { id } = await ctx.params;
  return handle(async () => {
    const body = (await req.json().catch(() => ({}))) as { end?: unknown; photons?: unknown };
    const num = (v: unknown) => (v === undefined || v === null || v === "" ? undefined : Number(v));
    const end = num(body.end), photons = num(body.photons);
    if (end !== undefined && !Number.isFinite(end)) throw new UserError("end time is not a number");
    if (photons !== undefined && !(Number.isInteger(photons) && photons > 0 && photons <= 10_000_000))
      throw new UserError("photons must be a whole number from 1 to 10 000 000");
    return startRun(id, { end, photons });
  });
}
