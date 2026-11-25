import { connection, type NextRequest } from "next/server";
import { handle } from "@/lib/server/http";
import { cancelRun, getRun } from "@/lib/server/store";

export async function GET(_req: NextRequest, ctx: RouteContext<"/api/runs/[id]">) {
  await connection();
  const { id } = await ctx.params;
  return handle(() => getRun(id));
}

export async function DELETE(_req: NextRequest, ctx: RouteContext<"/api/runs/[id]">) {
  const { id } = await ctx.params;
  return handle(async () => {
    cancelRun(id);
    return getRun(id);
  });
}
