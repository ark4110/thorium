import { connection, type NextRequest } from "next/server";
import { handle } from "@/lib/server/http";
import { getResults } from "@/lib/server/store";

export async function GET(_req: NextRequest, ctx: RouteContext<"/api/runs/[id]/results">) {
  await connection();
  const { id } = await ctx.params;
  return handle(() => getResults(id));
}
