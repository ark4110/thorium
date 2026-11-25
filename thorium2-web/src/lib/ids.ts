/** Model name usable as a folder name: letters, digits, '-' and '_' only. */
export function safeModelId(name: string): string {
  return name
    .replace(/\.(bdf|dat|top)$/i, "")
    .replace(/[^A-Za-z0-9_-]+/g, "_")
    .replace(/^_+|_+$/g, "")
    .slice(0, 64);
}
