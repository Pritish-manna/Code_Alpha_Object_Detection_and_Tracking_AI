import {useCallback, useEffect, useState} from 'react'
import {LayoutDashboard, Video, Upload, History as HistoryIcon, BarChart3, Settings as Cog, Info} from 'lucide-react'
import useDetection from './useDetection.js'
import Dashboard from './Dashboard.jsx'
import Analytics from './Analytics.jsx'
import SettingsPanel from './SettingsPanel.jsx'
import SystemInfo from './SystemInfo.jsx'
import History from './History.jsx'
import About from './About.jsx'
import {BackButton, DevicePill, StatusLine} from './ui.jsx'

const ROUTES = [['dashboard', 'Dashboard', LayoutDashboard], ['live', 'Live Detection', Video], ['upload', 'Video Upload', Upload],
  ['history', 'Detection History', HistoryIcon], ['analytics', 'Analytics', BarChart3], ['settings', 'Settings', Cog], ['about', 'About Project', Info]]
const parse = () => { const h = window.location.hash.replace(/^#\/?/, ''); return ROUTES.some(r => r[0] === h) ? h : 'dashboard' }

// The route lives in the URL hash: it survives reloads and hot reloads, the browser Back button works, and nothing in the
// app ever changes it except an explicit click, so no effect can bounce the user back to the Dashboard.
function useRoute() {
  const [route, setRoute] = useState(parse)
  useEffect(() => { const f = () => setRoute(parse()); window.addEventListener('hashchange', f); return () => window.removeEventListener('hashchange', f) }, [])
  const go = useCallback(r => { if (window.location.hash === '#/' + r) setRoute(r); else window.location.hash = '/' + r }, [])
  return [route, go]
}

export default function App() {
  const [route, go] = useRoute()
  const d = useDetection()
  const {model, conn, status} = d
  const toDash = () => go('dashboard'), toSettings = () => go('settings')
  const connLabel = {online: ['Online', 'text-emerald-400'], connecting: ['Connecting…', 'text-amber-300'], offline: ['Offline, reconnecting…', 'text-red-400']}[conn]

  return <div className="flex flex-col md:flex-row min-h-screen gap-4 p-3 md:p-4">
    <aside className="glass p-3 md:p-4 md:w-56 shrink-0 md:space-y-1">
      <div className="font-bold mb-2 md:mb-4 text-cyan-300">Object Detection<br className="hidden md:block"/> & Tracking AI</div>
      <nav aria-label="Main" className="flex md:block gap-1 overflow-x-auto md:overflow-visible">
        {ROUTES.map(([r, n, I]) => <a key={r} href={'#/' + r} aria-current={route === r ? 'page' : undefined}
          className={'flex gap-2 items-center px-3 py-2 rounded-xl text-sm whitespace-nowrap hover:bg-white/10 focus-visible:outline focus-visible:outline-2 focus-visible:outline-cyan-300 ' + (route === r ? 'bg-white/15 shadow-[0_0_12px_rgba(34,211,238,.4)]' : '')}>
          <I size={16} aria-hidden="true"/>{n}</a>)}
      </nav>
    </aside>
    <main className="flex-1 min-w-0 space-y-4">
      <div className="glass p-4 flex flex-wrap gap-x-6 gap-y-1 text-sm">
        <span>System: <b className={connLabel[1]}>{connLabel[0]}</b></span>
        <span>Model: <b className="capitalize">{model.unreachable && model.status !== 'loaded' ? 'unreachable' : model.status}</b>{model.active ? <> · <b>{model.active}</b></> : null}{model.imgsz_effective ? ' · ' + model.imgsz_effective + 'px' : ''}</span>
        <DevicePill model={model}/>
        <span>Source: <b>{d.mode === 'webcam' ? 'Webcam' : d.mode === 'video' || d.fid ? 'Uploaded Video' : '–'}</b></span>
        <StatusLine status={status}/>
      </div>
      {model.status === 'error' && !model.unreachable && <div className="glass p-3 text-sm text-red-400" role="alert">The model could not be loaded: {model.error}</div>}
      {conn === 'offline' && <div className="glass p-3 text-sm text-red-400" role="alert">Cannot reach the backend. Start it with <code>uvicorn app.main:app --port 8000</code>; this page reconnects automatically.</div>}

      {['dashboard', 'live', 'upload'].includes(route) && <Dashboard variant={route} d={d} goSettings={toSettings}/>}
      {route === 'history' && <History hist={d.hist} load={d.loadHistory} onBack={toDash}/>}
      {route === 'analytics' && <Analytics running={d.active} onBack={toDash}/>}
      {route === 'settings' && <div className="space-y-4">
        <div className="glass p-4 flex items-center justify-between gap-3"><h1 className="font-semibold text-cyan-300">Settings</h1><BackButton onClick={toDash}/></div>
        <SystemInfo model={model}/>
        <SettingsPanel model={model} conf={d.conf} setConf={d.setConf} imgsz={d.imgsz} setImgsz={d.setImgsz} onPickModel={d.pickModel}
          classNames={d.classNames} onFlip={d.flip} onClearClasses={() => d.classNames.forEach(d.flip)} onReset={d.resetSettings}
          frameStride={d.frameStride} setFrameStride={d.setFrameStride} embedder={d.embedder} setEmbedder={d.setEmbedder} locked={d.active}/>
      </div>}
      {route === 'about' && <About onBack={toDash}/>}
    </main>
    <video ref={d.vid} className="hidden" muted playsInline aria-hidden="true"/>
  </div>
}
