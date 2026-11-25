import { NextResponse, type NextRequest } from "next/server";
import { checkBasicAuth } from "@/lib/auth";

// The whole site is behind one password (THORIUM_WEB_PASSWORD), because it is reachable
// through a public tunnel. Without the variable nobody gets in.
export function proxy(request: NextRequest) {
  if (checkBasicAuth(request.headers.get("authorization"), process.env.THORIUM_WEB_PASSWORD ?? "")) return NextResponse.next();
  return new NextResponse("Password required", {
    status: 401,
    headers: { "WWW-Authenticate": 'Basic realm="Thorium 2", charset="UTF-8"' },
  });
}

export const config = { matcher: "/:path*" };
