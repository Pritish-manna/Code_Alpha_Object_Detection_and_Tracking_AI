import {Loader2, CheckCircle2, AlertTriangle, Info as InfoIcon, ArrowLeft as ArrowLeftIcon} from 'lucide-react'

export const B = ({icon: I, children, on, ...p}) =>
  <button type="button" className={'lg-btn ' + (on ? 'on' : '')} {...p}><I size={16}/>{children}</button>

export const Stat = ({l, v, sub, title}) =>
  <div className="glass p-4" title={title}><div className="text-xs text-slate-400">{l}</div>
    <div className="text-2xl font-semibold text-cyan-300">{v}</div>{sub && <div className="text-xs text-slate-500 mt-1">{sub}</div>}</div>

export const Spinner = ({size = 14}) => <Loader2 size={size} className="animate-spin inline-block" aria-hidden="true"/>

const KIND = {
  info: ['text-slate-300', InfoIcon], busy: ['text-cyan-300', null], ok: ['text-emerald-400', CheckCircle2], error: ['text-red-400', AlertTriangle],
}
export const StatusLine = ({status}) => {
  const [cls, Icon] = KIND[status.kind] || KIND.info
  return <span className={'inline-flex items-center gap-1.5 ' + cls} role="status" aria-live="polite">
    {status.kind === 'busy' ? <Spinner/> : Icon && <Icon size={14} aria-hidden="true"/>}{status.text}</span>
}

export const BackButton = ({onClick}) => <B icon={ArrowLeftIcon} onClick={onClick}>Back to Dashboard</B>

// Where the model really runs. Never shows a GPU unless the backend confirmed the weights are on CUDA.
export const DevicePill = ({model}) => {
  if (!model.device) return <span className="text-slate-400">Device: <b>{model.status === 'loading' ? 'detecting…' : '–'}</b></span>
  const gpu = model.gpu_active
  return <span title={model.device_note || ''}>Device: <b className={gpu ? 'text-emerald-400' : 'text-amber-300'}>
    {gpu ? 'GPU' : 'CPU'}</b>{model.device_name ? <span className="text-slate-300"> · {model.device_name}</span> : null}{gpu && model.fp16 ? ' · FP16' : ''}</span>
}
