import {Play, Square, Upload, Camera, RotateCcw, Pause, Settings as Cog, Cpu} from 'lucide-react'
import {B, Stat, Spinner, DevicePill} from './ui.jsx'

const TITLES = {
  dashboard: ['Dashboard', 'Webcam and uploaded video detection'],
  live: ['Live Detection', 'Detect and track objects from your webcam'],
  upload: ['Video Upload', 'Upload a video and process it frame by frame'],
}

export default function Dashboard({variant = 'dashboard', d, goSettings}) {
  const {phase, mode, online, loaded, idle, active, fid, fileInfo, uploading, stats, preview, progress, model} = d
  const [title, sub] = TITLES[variant] || TITLES.dashboard
  const webcamUI = variant !== 'upload', videoUI = variant !== 'live'
  const starting = phase === 'starting', stopping = phase === 'stopping'
  const canStart = idle && online && loaded && uploading == null
  const timing = stats.timing_ms || {}
  const stages = ['prep', 'detect', 'track', 'draw', 'encode'].filter(k => timing[k] != null)
  const empty = !preview && (variant === 'upload' ? 'Upload a video, then press Start Detection' : variant === 'live' ? 'Press Start Webcam' : 'Start the webcam or upload a video')

  return <div className="space-y-4">
    <div className="px-1"><h1 className="text-lg font-semibold text-cyan-300">{title}</h1><p className="text-xs text-slate-400">{sub}</p></div>

    <div className="glass p-3 relative min-h-[280px] md:min-h-[360px] flex items-center justify-center overflow-hidden">
      {preview ? <img src={preview} alt="Annotated detection preview" className="max-h-[60vh] max-w-full rounded-xl"/>
        : <span className="text-slate-400 text-center px-4">{starting ? <><Spinner/> Starting…</> : model.status === 'loading' ? <><Spinner/> Loading the detection model…</> : empty}</span>}
      {phase === 'paused' && <span className="absolute top-3 left-3 text-xs px-2 py-1 rounded-lg bg-black/50 border border-white/20">Paused</span>}
      {active && <span className="absolute top-3 right-3 text-xs px-2 py-1 rounded-lg bg-black/50 border border-white/20">{mode === 'webcam' ? 'Webcam' : 'Video'}</span>}
      {progress != null && <div className="absolute bottom-0 left-0 h-1 bg-cyan-400 transition-[width]" style={{width: progress + '%'}} role="progressbar" aria-valuenow={progress} aria-valuemin={0} aria-valuemax={100}/>}
    </div>

    <div className="glass p-4 flex flex-wrap gap-2" role="toolbar" aria-label="Detection controls">
      {webcamUI && <>
        <B icon={Camera} onClick={d.startWebcam} disabled={!canStart}>Start Webcam</B>
        <B icon={Square} onClick={d.stop} disabled={mode !== 'webcam' || !active || stopping}>Stop Webcam</B>
      </>}
      {videoUI && <>
        <label className={'lg-btn ' + (!idle || !online || uploading != null ? 'opacity-40 cursor-not-allowed' : '')} aria-disabled={!idle || !online}>
          {uploading != null ? <Spinner size={16}/> : <Upload size={16}/>}{uploading != null ? `Uploading ${uploading}%` : 'Upload Video'}
          <input type="file" accept=".mp4,.avi,.mov" hidden disabled={!idle || !online || uploading != null} onChange={e => { d.upload(e.target.files[0]); e.target.value = '' }}/>
        </label>
        <B icon={Play} onClick={d.startVideo} disabled={!canStart || !fid}>Start Detection</B>
        <B icon={Square} onClick={d.stop} disabled={mode !== 'video' || !active || stopping}>Stop Detection</B>
      </>}
      <B icon={Pause} on={phase === 'paused'} onClick={d.togglePause} disabled={phase !== 'running' && phase !== 'paused'}>{phase === 'paused' ? 'Resume' : 'Pause'}</B>
      <B icon={RotateCcw} onClick={d.reset} disabled={stopping}>Reset</B>
      {videoUI && fileInfo && <span className="text-xs text-slate-400 self-center ml-1">{fileInfo.name}{fileInfo.frames ? ` · ${fileInfo.frames} frames` : ''}{fileInfo.width ? ` · ${fileInfo.width}×${fileInfo.height}` : ''}</span>}
    </div>

    <div className="grid grid-cols-2 md:grid-cols-6 gap-3">
      <Stat l="Total detections" v={stats.total_detections ?? 0}/><Stat l="Tracked now" v={stats.tracked_now ?? 0}/><Stat l="Unique IDs" v={stats.unique_ids ?? 0}/>
      <Stat l="FPS" v={stats.fps ?? 0} sub={stats.avg_fps ? `avg ${stats.avg_fps}` : null} title="Current frames per second (last 30 frames)"/>
      <Stat l="Frames" v={stats.frames ?? 0} sub={stats.frames_skipped ? `${stats.frames_skipped} skipped` : null}/><Stat l="Avg confidence" v={(stats.avg_confidence ?? 0) + '%'}/>
    </div>

    <div className="glass p-3 text-xs text-slate-300 flex flex-wrap items-center gap-x-5 gap-y-1">
      <Cpu size={14} className="text-slate-400" aria-hidden="true"/>
      <DevicePill model={model}/>
      <span>Model: <b>{model.active || '–'}</b>{model.imgsz_effective ? ` · ${model.imgsz_effective}px` : ''}</span>
      {stats.latency_ms ? <span title="Time spent inside the model per frame">Inference: <b>{stats.latency_ms} ms</b></span> : null}
      {stages.length ? <span title="Average milliseconds per pipeline stage">{stages.map(k => `${k} ${timing[k]}`).join(' · ')} ms</span> : null}
      <button type="button" className="ml-auto underline text-cyan-300 inline-flex items-center gap-1" onClick={goSettings}><Cog size={12}/>Settings</button>
    </div>
    {model.hint && <div className="glass p-3 text-xs text-amber-300">{model.hint}</div>}
  </div>
}
