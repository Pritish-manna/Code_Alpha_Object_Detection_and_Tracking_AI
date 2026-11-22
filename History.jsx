import {useEffect} from 'react'
import {RefreshCw} from 'lucide-react'
import {B, Spinner, BackButton} from './ui.jsx'

export default function History({hist, load, onBack}) {
  useEffect(() => { load() }, [load])
  return <div className="glass p-4 space-y-3">
    <div className="flex flex-wrap items-center justify-between gap-3"><h1 className="font-semibold text-cyan-300">Detection History</h1>
      <div className="flex gap-2"><B icon={RefreshCw} onClick={load} disabled={hist.loading}>Refresh</B><BackButton onClick={onBack}/></div></div>
    {hist.error && <div className="text-sm text-amber-300" role="alert">{hist.error}</div>}
    {hist.loading && !hist.rows.length ? <div className="text-sm text-slate-400"><Spinner/> Loading…</div>
      : hist.rows.length ? hist.rows.map((h, i) => <div key={i} className="text-sm py-1 border-b border-white/10">{h.t} · {h.cls} (ID {h.id})</div>)
      : !hist.error && <div className="text-sm text-slate-400">No detections yet.</div>}
  </div>
}
