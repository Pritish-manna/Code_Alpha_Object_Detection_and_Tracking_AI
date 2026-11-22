import {BackButton} from './ui.jsx'

export default function About({onBack}) {
  return <div className="glass p-6 text-sm space-y-4 max-w-3xl">
    <div className="flex items-center justify-between gap-3"><h1 className="text-lg font-semibold text-cyan-300">About Project</h1><BackButton onClick={onBack}/></div>
    <p>YOLO11 detection + DeepSORT tracking, FastAPI backend, React frontend, WebSocket streaming.</p>
    <ul className="list-disc pl-5 space-y-1 text-slate-300">
      <li>Webcam frames are sent to the backend, detected, tracked and returned annotated; only the newest frame is processed, so the preview never lags behind.</li>
      <li>Uploaded videos are processed completely, frame by frame, with progress; Stop, Pause, Resume and Reset release the video immediately.</li>
      <li>Class labels come only from the model; tracks keep a stable class through a short vote, so one bad frame cannot relabel an object.</li>
      <li>Analytics are computed from the real detection session on the backend, for both webcam and video.</li>
    </ul>
  </div>
}
