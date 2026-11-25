// HTTP Basic authentication with one shared password (any user name).

function sameText(a: string, b: string): boolean {
  // Constant time for equal lengths; the length itself is not secret here.
  if (a.length !== b.length) return false;
  let diff = 0;
  for (let i = 0; i < a.length; i++) diff |= a.charCodeAt(i) ^ b.charCodeAt(i);
  return diff === 0;
}

export function checkBasicAuth(header: string | null, password: string): boolean {
  if (!password || !header || !header.startsWith("Basic ")) return false;
  let decoded: string;
  try {
    decoded = atob(header.slice(6).trim());
  } catch {
    return false;
  }
  const colon = decoded.indexOf(":");
  if (colon < 0) return false;
  return sameText(decoded.slice(colon + 1), password);
}
