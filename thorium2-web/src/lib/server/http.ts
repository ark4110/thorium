import { UserError } from "./store";

/** Runs a handler and turns errors into JSON responses (400 for user errors, 500 otherwise). */
export async function handle(fn: () => Promise<unknown>): Promise<Response> {
  try {
    return Response.json(await fn());
  } catch (e) {
    if (e instanceof UserError) return Response.json({ error: e.message }, { status: 400 });
    console.error(e);
    return Response.json({ error: "internal error: " + String(e) }, { status: 500 });
  }
}
