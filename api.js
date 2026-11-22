// Backend location. Override with VITE_API_URL / VITE_WS_URL (e.g. in frontend/.env) when the API is not on port 8000
// of the machine that serves the page. Using the page's own hostname keeps http://127.0.0.1:5173 and LAN access working.
const host = window.location.hostname || 'localhost'
export const API = import.meta.env.VITE_API_URL || `http://${host}:8000`
export const WS = import.meta.env.VITE_WS_URL || API.replace(/^http/, 'ws') + '/ws/detection'

export async function getJSON(path, init) {
  const r = await fetch(API + path, init)
  let body = null
  try { body = await r.json() } catch { /* empty or non-JSON body */ }
  if (!r.ok) throw new Error((body && (body.detail || body.message)) || `HTTP ${r.status}`)
  return body
}
