import {useEffect, useState, useCallback} from 'react'
import {RefreshCw} from 'lucide-react'
import {B, Spinner} from './ui.jsx'
import {getJSON} from './api.js'

const Row = ({k, v}) => <div className="flex justify-between gap-4 py-1 border-b border-white/10"><span className="text-slate-400">{k}</span><span className="text-right">{v ?? '–'}</span></div>

// Hardware facts queried from the backend (/api/system). The GPU is only reported as active if the model weights are on CUDA.
export default function SystemInfo({model}) {
  const [s, setS] = useState(null), [err, setErr] = useState('')
  const load = useCallback(() => getJSON('/api/system').then(d => { setS(d); setErr('') }).catch(e => setErr('Could not read system info (' + e.message + ').')), [])
  useEffect(() => { load() }, [load, model.active])
  if (err) return <div className="glass p-4 text-sm text-amber-300" role="alert">{err}</div>
  if (!s) return <div className="glass p-4 text-sm text-slate-400"><Spinner/> Reading system info…</div>
  const gpu = s.gpu_active
  return <div className="glass p-4 text-sm space-y-3">
    <div className="flex items-center justify-between"><h2 className="font-semibold text-cyan-300">Hardware acceleration</h2><B icon={RefreshCw} onClick={load}>Refresh</B></div>
    <div className={'rounded-xl p-3 border ' + (gpu ? 'border-emerald-400/40 text-emerald-300' : 'border-amber-300/40 text-amber-300')}>
      {s.device ? <b>{gpu ? `Running on GPU: ${s.device_name}${s.fp16 ? ' (FP16)' : ''}` : 'Running on CPU'}</b> : <b>Model still loading…</b>}
      {s.device_note && <div className="text-xs mt-1 text-slate-200 break-words">{s.device_note}</div>}
    </div>
    {!gpu && s.device && <div className="text-xs text-slate-300">To use an NVIDIA GPU: <code className="block mt-1 p-2 rounded bg-black/40 break-all select-all">{s.cuda_install_hint}</code></div>}
    <div className="grid md:grid-cols-2 gap-x-8">
      <Row k="PyTorch" v={s.torch_version}/><Row k="Built for CUDA" v={s.torch_cuda || 'none (CPU-only build)'}/>
      <Row k="CUDA usable" v={s.cuda_available ? 'yes' : 'no'}/><Row k="GPU(s)" v={s.gpus?.length ? s.gpus.map(g => `${g.name} ${g.memory_gb} GB`).join(', ') : 'none detected'}/>
      <Row k="Requested device" v={s.requested_device}/><Row k="CPU threads" v={s.cpu_threads}/><Row k="OS" v={s.os}/><Row k="Python" v={s.python}/>
    </div>
  </div>
}
