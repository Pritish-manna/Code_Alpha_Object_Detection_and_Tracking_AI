import {RotateCcw} from 'lucide-react'

const QUICK = ['person', 'car', 'bicycle', 'bus', 'motorcycle', 'dog', 'cat', 'bottle', 'chair', 'laptop', 'cell phone', 'keyboard', 'mouse', 'book']
const SIZES = [[320, '320 px: fastest, small objects may be missed'], [416, '416 px: fast'], [640, '640 px: balanced'], [960, '960 px: most accurate']]

export default function SettingsPanel({model, conf, setConf, imgsz, setImgsz, onPickModel, classNames, onFlip, onClearClasses, onReset, frameStride, setFrameStride, embedder, setEmbedder, locked}) {
  const avail = model.available || [], names = model.classes || []
  const selected = model.loading || model.active || ''
  return <div className="glass p-4 space-y-4 text-sm">
    <div className="flex items-center justify-between">
      <h2 className="font-semibold text-cyan-300">Detection settings</h2>
      <button type="button" className="lg-btn" onClick={onReset}><RotateCcw size={14}/>Defaults</button>
    </div>
    <div className="grid md:grid-cols-3 gap-4">
      <label className="block space-y-1"><span className="text-slate-300">Model</span>
        <select className="lg-select" value={selected} disabled={!avail.length || !!model.loading} onChange={e => onPickModel(e.target.value)}>
          {!avail.length && <option value="">Loading…</option>}
          {avail.map(m => <option key={m.name} value={m.name}>{m.label}{m.downloaded ? '' : ' (downloads on first use)'}</option>)}
        </select>
        {model.loading && <span className="text-xs text-cyan-300">Loading {model.loading}… the current model keeps running meanwhile.</span>}
      </label>
      <label className="block space-y-1"><span className="text-slate-300">Inference size</span>
        <select className="lg-select" value={imgsz} onChange={e => setImgsz(e.target.value === 'auto' ? 'auto' : +e.target.value)}>
          <option value="auto">Auto{model.auto_imgsz ? ` (${model.auto_imgsz}px on this machine)` : ''}</option>
          {SIZES.map(([v, l]) => <option key={v} value={v}>{l}</option>)}
        </select>
        {model.latency_ms ? <span className="text-xs text-slate-400">Measured {model.latency_ms} ms per frame at 960 px.</span> : null}
      </label>
      <label className="block space-y-1"><span className="text-slate-300">Confidence threshold: <b className="text-cyan-300">{Math.round(conf * 100)}%</b></span>
        <input type="range" min=".1" max=".95" step=".05" value={conf} onChange={e => setConf(+e.target.value)} className="lg-range w-full"/>
        <span className="text-xs text-slate-400">Detections below this are never shown. Higher = fewer false positives, more misses.</span>
      </label>
      <label className="block space-y-1"><span className="text-slate-300">Video frame stride: <b className="text-cyan-300">{frameStride === 1 ? 'every frame' : `every ${frameStride}rd/th frame`}</b></span>
        <input type="range" min="1" max="6" step="1" value={frameStride} onChange={e => setFrameStride(+e.target.value)} className="lg-range w-full"/>
        <span className="text-xs text-slate-400">Uploaded video only. Skipping frames finishes sooner; 1 processes everything.</span>
      </label>
      <label className="block space-y-1"><span className="text-slate-300">Tracking appearance matching</span>
        <select className="lg-select" value={embedder} disabled={locked} onChange={e => setEmbedder(e.target.value)}>
          <option value="mobilenet">On: steadier IDs (slower on CPU)</option><option value="none">Off: motion only (faster)</option>
        </select>
        <span className="text-xs text-slate-400">{locked ? 'Stop detection to change this.' : 'Applies from the next start.'}</span>
      </label>
    </div>
    {model.last_error && <div className="text-xs text-amber-300">{model.last_error}</div>}
    <div>Class filter (none selected = all):
      <div className="flex flex-wrap gap-2 mt-2">
        {QUICK.filter(n => names.includes(n)).map(n =>
          <button type="button" key={n} onClick={() => onFlip(n)} aria-pressed={classNames.includes(n)} className={'lg-btn ' + (classNames.includes(n) ? 'on' : '')}>{n}</button>)}
        {classNames.length ? <button type="button" className="lg-btn" onClick={onClearClasses}>Clear filter</button> : null}
      </div>
    </div>
  </div>
}
