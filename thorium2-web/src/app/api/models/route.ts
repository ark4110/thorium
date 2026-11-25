import { connection } from "next/server";
import { handle } from "@/lib/server/http";
import { listModels, saveModel, UserError } from "@/lib/server/store";

export async function GET() {
  await connection();
  return handle(() => listModels());
}

/** Upload of a *.bdf + *.top pair (multipart form fields "bdf" and "top"). */
export async function POST(request: Request) {
  return handle(async () => {
    const form = await request.formData();
    const bdf = form.get("bdf"), top = form.get("top");
    if (!(bdf instanceof File) || !(top instanceof File)) throw new UserError("choose both the *.bdf and the *.top file");
    return saveModel(bdf.name, Buffer.from(await bdf.arrayBuffer()), Buffer.from(await top.arrayBuffer()));
  });
}
