import { connection, type NextRequest } from "next/server";
import { handle } from "@/lib/server/http";
import { getMesh } from "@/lib/server/store";

export async function GET(_req: NextRequest, ctx: RouteContext<"/api/models/[id]/mesh">) {
  await connection();
  const { id } = await ctx.params;
  return handle(() => getMesh(id));
}
