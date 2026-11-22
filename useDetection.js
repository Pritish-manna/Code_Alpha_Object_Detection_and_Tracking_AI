import {useCallback, useEffect, useRef, useState} from 'react'
import {API, WS, getJSON} from './api.js'

// ---- constants -------------------------------------------------------------------------------------------------
export const DEFAULTS = {model: 'yolo11s.pt', conf: 0.4, imgsz: 'auto', frameStride: 1, embedder: 'mobilenet'}
const CAM = {width: {ideal: 1280}, height: {ideal: 720}}
const JPEG_Q = 0.85
const MAX_INFLIGHT = 2          // frames sent but not yet answered; the server always processes only the newest
const TICK_MS = 15
const SAVED_KEY = 'odt.settings.v1'

const loadSaved = () => { try { return JSON.parse(localStorage.getItem(SAVED_KEY)) } catch { return null } }
const camError = e => ({
  NotAllowedError: 'Camera permission was denied. Allow camera access for this site and try again.',
  NotFoundError: 'No camera was found on this computer.',
  NotReadableError: 'The camera is in use by another application. Close it and try again.',
  OverconstrainedError: 'The camera does not support the requested resolution.',
}[e?.name] || 'Camera error: ' + (e?.message || e))

/**
 * Everything that talks to the backend: WebSocket, webcam capture, uploads, model status and settings.
 * phase: idle | starting | running | paused | stopping     mode: webcam | video | null
 * The server is the source of truth for phase/mode (it broadcasts {type:'state'}); the UI never guesses.
 */
export default function useDetection() {
  const saved = useRef(loadSaved()).current
  const [conn, setConn] = useState('connecting')                 // connecting | online | offline
  const [phase, setPhaseState] = useState('idle'), [mode, setModeState] = useState(null)
  const [status, setStatusState] = useState({kind: 'info', text: 'Idle'})
  const [stats, setStats] = useState({}), [preview, setPreview] = useState(''), [progress, setProgress] = useState(null)
  const [model, setModel] = useState({status: 'loading'}), [hist, setHist] = useState({rows: [], loading: false, error: ''})
  const [fid, setFid] = useState(null), [fileInfo, setFileInfo] = useState(null), [uploading, setUploading] = useState(null)
  const [conf, setConf] = useState(saved?.conf ?? DEFAULTS.conf), [imgsz, setImgsz] = useState(saved?.imgsz ?? DEFAULTS.imgsz)
  const [classNames, setClassNames] = useState(saved?.classNames ?? [])
  const [frameStride, setFrameStride] = useState(saved?.frameStride ?? DEFAULTS.frameStride)
  const [embedder, setEmbedder] = useState(saved?.embedder ?? DEFAULTS.embedder)
  const [ready, setReady] = useState(!!saved)

  const vid = useRef(null), cvs = useRef(null), ws = useRef(null), stream = useRef(null), alive = useRef(false)
  const timer = useRef(null), reconnect = useRef(null), stopTimer = useRef(null)
  const inflight = useRef(0), lastSend = useRef(0), pendingMeta = useRef(null), urlRef = useRef(''), targetW = useRef(960)
  const phaseRef = useRef('idle'), modeRef = useRef(null), inited = useRef(!!saved), wasLoading = useRef(false)

  const setPhase = p => { phaseRef.current = p; setPhaseState(p) }
  const setMode = m => { modeRef.current = m; setModeState(m) }
  const setStatus = (text, kind = 'info') => setStatusState({text, kind})
  const send = o => { const s = ws.current; if (s?.readyState === 1) { s.send(JSON.stringify(o)); return true } return false }

  // ---- preview image (object URLs, revoked after the next frame replaces them) ---------------------------------
  const showFrame = blob => {
    const url = URL.createObjectURL(blob), old = urlRef.current
    urlRef.current = url; setPreview(url)
    if (old) setTimeout(() => URL.revokeObjectURL(old), 1000)
  }
  const clearPreview = () => { const old = urlRef.current; urlRef.current = ''; setPreview(''); if (old) URL.revokeObjectURL(old) }

  // ---- camera --------------------------------------------------------------------------------------------------
  const releaseCamera = () => {
    clearTimeout(timer.current); timer.current = null
    stream.current?.getTracks().forEach(t => t.stop()); stream.current = null
    if (vid.current) vid.current.srcObject = null
    inflight.current = 0; pendingMeta.current = null
  }
  const cleanup = (text, kind) => {          // back to a clean idle state, whatever state we were in
    clearTimeout(stopTimer.current); releaseCamera(); setPhase('idle'); setMode(null)
    if (text) setStatus(text, kind || 'info')
  }

  const capture = () => {
    const v = vid.current, s = ws.current, c = cvs.current
    if (!v?.videoWidth || s?.readyState !== 1 || !c) return
    const w = Math.min(v.videoWidth, targetW.current), h = Math.round(v.videoHeight * w / v.videoWidth)
    if (c.width !== w || c.height !== h) { c.width = w; c.height = h }
    c.getContext('2d').drawImage(v, 0, 0, w, h)
    inflight.current++; lastSend.current = Date.now()
    c.toBlob(b => {
      if (b && ws.current?.readyState === 1) ws.current.send(b)
      else inflight.current = Math.max(0, inflight.current - 1)
    }, 'image/jpeg', JPEG_Q)
  }
  const startCaptureLoop = () => {
    clearTimeout(timer.current); inflight.current = 0
    const tick = () => {
      const p = phaseRef.current
      if ((p !== 'running' && p !== 'paused') || modeRef.current !== 'webcam') return    // the loop ends by itself
      timer.current = setTimeout(tick, TICK_MS)
      if (p === 'paused') return                                                         // nothing is sent while paused
      if (inflight.current >= MAX_INFLIGHT) {
        if (Date.now() - lastSend.current > 4000) inflight.current = 0                   // watchdog: never wait forever for a reply
        return
      }
      capture()
    }
    tick()
  }

  // ---- messages from the server --------------------------------------------------------------------------------
  const onMessage = e => {
    if (typeof e.data !== 'string') {                                  // binary = annotated JPEG belonging to the last 'frame' message
      const meta = pendingMeta.current; pendingMeta.current = null
      inflight.current = Math.max(0, inflight.current - 1)
      if (phaseRef.current === 'idle' || phaseRef.current === 'stopping') return        // late frame after Stop/Reset: ignore
      showFrame(e.data)
      if (meta) { setStats(meta.stats); if (meta.progress != null) setProgress(meta.progress) }
      return
    }
    let d; try { d = JSON.parse(e.data) } catch { return }
    switch (d.type) {
      case 'frame': pendingMeta.current = d; break
      case 'skip': inflight.current = Math.max(0, inflight.current - 1); break
      case 'frame_error': inflight.current = Math.max(0, inflight.current - 1); setStatus('Frame error: ' + d.message, 'error'); break
      case 'settings_error': setStatus('Setting rejected: ' + d.message, 'error'); break
      case 'state':
        if (d.state === 'idle') { if (phaseRef.current !== 'starting') cleanup() }
        else { setMode(d.mode); setPhase(d.state); if (d.state === 'running') setStatus(d.mode === 'webcam' ? 'Detecting from webcam' : 'Processing video', 'busy') ; else setStatus('Paused', 'info') }
        break
      case 'ack':
        if (d.stats) setStats(d.stats)
        if (d.action === 'start' && d.ok && modeRef.current === 'webcam') startCaptureLoop()
        if (d.action === 'stop') { cleanup('Stopped'); }
        break
      case 'error':
        setStatus(d.message, 'error')
        if (d.action === 'start' || phaseRef.current === 'starting') cleanup(d.message, 'error')
        break
      case 'done':
        if (d.stats) setStats(d.stats)
        setProgress(100); cleanup(`Finished: ${d.stats?.frames ?? 0} frames processed`, 'ok')
        break
      default: break
    }
  }

  // ---- connection ----------------------------------------------------------------------------------------------
  const connect = useCallback(() => {
    if (!alive.current) return
    const s = new WebSocket(WS); s.binaryType = 'blob'; ws.current = s
    s.onopen = () => { if (ws.current === s) setConn('online') }
    s.onmessage = e => { if (ws.current === s) onMessage(e) }
    s.onclose = () => {
      if (ws.current !== s || !alive.current) return
      setConn('offline')
      if (phaseRef.current !== 'idle') cleanup('Lost the connection to the backend. Reconnecting…', 'error')
      reconnect.current = setTimeout(connect, 1500)
    }
    s.onerror = () => {}                                                // onclose follows and handles it
  }, [])                                                                // eslint-disable-line react-hooks/exhaustive-deps
  useEffect(() => {
    alive.current = true; cvs.current = document.createElement('canvas'); connect()
    return () => {
      alive.current = false; clearTimeout(reconnect.current); clearTimeout(stopTimer.current); releaseCamera()
      const s = ws.current; ws.current = null; s?.close()
      if (urlRef.current) URL.revokeObjectURL(urlRef.current)
    }
  }, [connect])

  // ---- model status (polled; faster while a model loads) -------------------------------------------------------
  const loadingNow = !!model.loading
  useEffect(() => {
    let live = true
    const poll = () => getJSON('/api/model/status').then(d => {
      if (!live) return
      setModel(d)
      if (wasLoading.current && !d.loading) setStatus(d.last_error ? 'Model switch failed: see Settings' : 'Model ready: ' + d.active, d.last_error ? 'error' : 'ok')
      wasLoading.current = !!d.loading
      if (!inited.current && d.conf != null) {
        inited.current = true; setConf(d.conf); setImgsz(d.imgsz ?? 'auto'); setFrameStride(d.frame_stride ?? 1); setEmbedder(d.tracker_embedder ?? 'mobilenet'); setReady(true)
      }
    }).catch(() => live && setModel(m => ({...m, status: m.status === 'loaded' ? 'loaded' : 'error', unreachable: true})))
    poll(); const i = setInterval(poll, loadingNow ? 1500 : 4000)
    return () => { live = false; clearInterval(i) }
  }, [loadingNow])
  useEffect(() => { if (model.imgsz_effective) targetW.current = Math.min(1280, Math.max(640, model.imgsz_effective)) }, [model.imgsz_effective])

  // ---- settings: persisted locally, pushed to the backend on change and on every (re)connect -------------------
  useEffect(() => {
    if (!ready) return
    try { localStorage.setItem(SAVED_KEY, JSON.stringify({conf, imgsz, classNames, frameStride, embedder})) } catch { /* private mode */ }
    send({action: 'settings', conf, imgsz, class_names: classNames, frame_stride: frameStride, tracker_embedder: embedder})
  }, [conf, imgsz, classNames, frameStride, embedder, conn, ready])

  // ---- actions -------------------------------------------------------------------------------------------------
  const online = conn === 'online', loaded = model.status === 'loaded'
  const idle = phase === 'idle', active = phase === 'running' || phase === 'paused' || phase === 'starting'

  const startWebcam = async () => {
    if (phaseRef.current !== 'idle') return
    if (!ws.current || ws.current.readyState !== 1) return setStatus('Not connected to the backend.', 'error')
    if (!navigator.mediaDevices?.getUserMedia) return setStatus('The browser blocks camera access on this address. Open the app at http://localhost:5173.', 'error')
    setPhase('starting'); setMode('webcam'); setProgress(null); setStatus('Requesting camera…', 'busy')
    try {
      const s = await navigator.mediaDevices.getUserMedia({video: CAM, audio: false})
      if (phaseRef.current !== 'starting') { s.getTracks().forEach(t => t.stop()); return }      // cancelled while the permission prompt was open
      stream.current = s; vid.current.srcObject = s; await vid.current.play()
      if (phaseRef.current !== 'starting') return
      setStatus('Starting detection…', 'busy')
      if (!send({action: 'start', mode: 'webcam'})) cleanup('Not connected to the backend.', 'error')
    } catch (e) { cleanup(camError(e), 'error') }
  }
  const startVideo = () => {
    if (phaseRef.current !== 'idle') return
    if (!fid) return setStatus('Upload a video first.', 'error')
    setPhase('starting'); setMode('video'); setProgress(0); setStatus('Starting video detection…', 'busy')
    if (!send({action: 'start', mode: 'video', file_id: fid})) cleanup('Not connected to the backend.', 'error')
  }
  const stop = () => {
    if (phaseRef.current === 'idle' || phaseRef.current === 'stopping') return
    clearTimeout(timer.current)
    if (!send({action: 'stop'})) return cleanup('Stopped')
    setPhase('stopping'); setStatus('Stopping…', 'busy')
    clearTimeout(stopTimer.current); stopTimer.current = setTimeout(() => cleanup('Stopped'), 4000)     // buttons can never stay stuck
  }
  const reset = () => {
    send({action: 'reset'}); cleanup(); setStats({}); setProgress(null); clearPreview(); setStatus('Reset: session data cleared')
  }
  const togglePause = () => {
    if (phaseRef.current === 'running') send({action: 'pause'})
    else if (phaseRef.current === 'paused') send({action: 'resume'})
  }

  const upload = file => {
    if (!file || phaseRef.current !== 'idle') return
    if (!/\.(mp4|avi|mov)$/i.test(file.name)) return setStatus('Unsupported format. Use MP4, AVI or MOV.', 'error')
    const fd = new FormData(); fd.append('file', file)
    const x = new XMLHttpRequest(); x.open('POST', API + '/api/video/upload'); setUploading(0); setStatus('Uploading ' + file.name + '…', 'busy')
    x.upload.onprogress = e => e.lengthComputable && setUploading(Math.round(100 * e.loaded / e.total))
    x.onerror = () => { setUploading(null); setStatus('Upload failed: cannot reach the backend.', 'error') }
    x.onload = () => {
      setUploading(null)
      let d = {}; try { d = JSON.parse(x.responseText) } catch { /* keep {} */ }
      if (x.status < 200 || x.status >= 300) return setStatus('Upload error: ' + (d.detail || 'HTTP ' + x.status), 'error')
      setFid(d.file_id); setFileInfo({...d, name: file.name}); clearPreview(); setStats({}); setProgress(null)
      setStatus(`Video ready: ${file.name}${d.frames ? ` (${d.frames} frames)` : ''}. Press Start Detection.`, 'ok')
    }
    x.send(fd)
  }

  const pickModel = async name => {
    setStatus('Switching model…', 'busy')
    try {
      const d = await getJSON('/api/model/select', {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({model: name})})
      setModel(m => ({...m, loading: d.loading || m.loading})); setStatus(d.changed ? 'Loading ' + name + '…' : 'Model already active', d.changed ? 'busy' : 'info')
    } catch (x) { setStatus('Model error: ' + x.message, 'error') }
  }
  const flip = n => setClassNames(c => c.includes(n) ? c.filter(x => x !== n) : [...c, n])
  const resetSettings = () => {
    setConf(DEFAULTS.conf); setImgsz(DEFAULTS.imgsz); setClassNames([]); setFrameStride(DEFAULTS.frameStride); setEmbedder(DEFAULTS.embedder)
    if (model.active && model.active !== DEFAULTS.model) pickModel(DEFAULTS.model)
  }
  const loadHistory = useCallback(async () => {
    setHist(h => ({...h, loading: true, error: ''}))
    try { setHist({rows: await getJSON('/api/detection/history'), loading: false, error: ''}) }
    catch (e) { setHist(h => ({...h, loading: false, error: 'Could not load history (' + e.message + ').'})) }
  }, [])

  return {
    vid, conn, online, loaded, phase, mode, idle, active, status, stats, preview, progress, model, hist, loadHistory,
    fid, fileInfo, uploading, conf, setConf, imgsz, setImgsz, classNames, flip, frameStride, setFrameStride, embedder, setEmbedder,
    startWebcam, startVideo, stop, reset, togglePause, upload, pickModel, resetSettings,
  }
}
