import {useEffect, useState, useCallback} from 'react'
import {RefreshCw} from 'lucide-react'
import {B, Stat, Spinner, BackButton} from './ui.jsx'
import {getJSON} from './api.js'

// Every number comes from GET /api/detection/analytics (the engine's real session data). Nothing is computed from guesses here.
export default function Analytics({running, onBack}) {
  const [data, setData] = useState(null), [err, setErr] = useState(''), [loading, setLoading] = useState(true)
  const load = useCallback(() => {
    setLoading(true)
    return getJSON('/api/detection/analytics').then(d => { setData(d); setErr('') })
      .catch(e => setErr('Could not load analytics from the backend (' + e.message + ').')).finally(() => setLoading(false))
  }, [])
  useEffect(() => { load(); if (!running) return; const i = setInterval(load, 1000); return () => clearInterval(i) }, [load, running])

  const s = data?.stats || {}, classes = data?.classes || []
  const session = data ? (data.has_data ? (data.running ? (data.paused ? 'Paused' : 'Running') : 'Completed') + ' · ' + (data.mode === 'video' ? 'Uploaded video' : data.mode === 'webcam' ? 'Webcam' : 'Last session') : 'No session yet') : '–'
  return <div className="space-y-4">
    <div className="glass p-4 flex flex-wrap items-center justify-between gap-3">
      <div><h1 className="font-semibold text-cyan-300">Analytics</h1><div className="text-xs text-slate-400">Session: {session}{data?.has_data ? ` · ${data.active_seconds}s active` : ''}</div></div>
      <div className="flex gap-2"><B icon={RefreshCw} onClick={load} disabled={loading}>{loading ? 'Refreshing…' : 'Refresh'}</B><BackButton onClick={onBack}/></div>
    </div>
    {err && <div className="glass p-4 text-sm text-amber-300" role="alert">{err}</div>}
    {loading && !data && !err && <div className="glass p-4 text-sm text-slate-400"><Spinner/> Loading analytics…</div>}
    {data && !data.has_data && <div className="glass p-4 text-sm text-slate-300">No detection data yet. Start the webcam or process an uploaded video from the dashboard, then return here.</div>}
    {data && <>
      <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
        <Stat l="Total detections" v={s.total_detections ?? 0} title="Tracked objects drawn, summed over all frames"/>
        <Stat l="Currently tracked objects" v={s.tracked_now ?? 0}/><Stat l="Unique tracking IDs" v={s.unique_ids ?? 0}/>
        <Stat l="Average FPS" v={s.avg_fps ?? 0} sub={`current ${s.fps ?? 0}`} title="Frames per second of active processing, pauses excluded"/>
        <Stat l="Total processed frames" v={s.frames ?? 0} sub={s.frames_skipped ? `${s.frames_skipped} skipped by frame stride` : null}/>
        <Stat l="Average confidence" v={(s.avg_confidence ?? 0) + '%'}/>
        <Stat l="Device" v={s.device ? (s.device.startsWith('cuda') ? 'GPU' : 'CPU') : '–'} sub={[s.device_name, s.fp16 ? 'FP16' : null].filter(Boolean).join(' · ') || null}/>
        <Stat l="Model" v={s.model || '–'} sub={s.imgsz ? `${s.imgsz}px · ${s.latency_ms ?? 0} ms/frame` : null}/>
      </div>
      <div className="glass p-4 text-sm"><h2 className="mb-3 font-semibold text-cyan-300">Class-wise detection summary</h2>
        {classes.length ? <div className="space-y-3">
          <div className="hidden md:grid grid-cols-[10rem_1fr_6rem_6rem_7rem] gap-3 text-xs text-slate-400"><span>Class</span><span>Share of all detections</span><span className="text-right">Detections</span><span className="text-right">Unique IDs</span><span className="text-right">Avg conf.</span></div>
          {classes.map(c => <div key={c.cls} className="grid grid-cols-2 md:grid-cols-[10rem_1fr_6rem_6rem_7rem] gap-3 items-center">
            <span className="capitalize">{c.cls}</span>
            <div className="hidden md:flex items-center gap-2"><div className="h-2 flex-1 rounded bg-white/10 overflow-hidden"><div className="h-full bg-cyan-400" style={{width: c.share_pct + '%'}}/></div><span className="text-xs text-slate-300 w-12 text-right">{c.share_pct}%</span></div>
            <span className="text-right text-cyan-300">{c.detections}</span><span className="text-right">{c.unique_ids}</span><span className="text-right">{c.avg_confidence}%</span></div>)}
        </div> : <span className="text-slate-400">No classes detected yet.</span>}
      </div>
    </>}
  </div>
}
