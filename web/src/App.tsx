import { useEffect, useState } from 'react'
import Conditions from './Conditions'
import SunCanvas, { type SunView } from './SunCanvas'

const BASE = import.meta.env.BASE_URL

interface Meta {
  observation_time: string
  coronal_hole_pct: number
  active_region_pct: number
}

const WAVELENGTH_STOPS = [
  { at: 0, label: '171 Å', temp: '0.6 MK · quiet corona' },
  { at: 1, label: '193 Å', temp: '1.2 MK · corona & holes' },
  { at: 2, label: '304 Å', temp: '0.05 MK · chromosphere' },
]

export default function App() {
  const [meta, setMeta] = useState<Meta | null>(null)
  const [mix, setMix] = useState(1) // start on 193A, where the story is
  const [wipe, setWipe] = useState(0)
  const [showCH, setShowCH] = useState(true)
  const [showAR, setShowAR] = useState(true)

  useEffect(() => {
    const load = () =>
      fetch(`${BASE}live/meta.json`)
        .then((r) => r.json())
        .then(setMeta)
        .catch(() => {})
    load()
    const timer = setInterval(load, 5 * 60 * 1000)
    return () => clearInterval(timer)
  }, [])

  const view: SunView = { mix, wipe, showCH, showAR }
  const obs = meta?.observation_time
    ? meta.observation_time.slice(0, 16).replace('T', ' ') + ' UT'
    : '—'

  return (
    <div className="stage">
      <header className="masthead">
        <h1>
          <span className="masthead-el">Ο Ήλιος</span>
          <span className="masthead-en">The Living Sun</span>
        </h1>
        <nav className="masthead-links">
          <a href="https://github.com/CaptainJimbo/o-ilios">source & methodology</a>
        </nav>
      </header>

      <aside className="telemetry" aria-label="Observation telemetry">
        <dl>
          <dt>DATE-OBS</dt>
          <dd>{obs}</dd>
          <dt>TELESCOP</dt>
          <dd>SDO / AIA</dd>
          <dt>SEGMENTS</dt>
          <dd>
            <label className="seg-toggle">
              <input type="checkbox" checked={showCH}
                onChange={(e) => setShowCH(e.target.checked)} />
              <span className="swatch swatch-ch" />
              coronal holes {meta ? `${meta.coronal_hole_pct.toFixed(1)}%` : ''}
            </label>
            <label className="seg-toggle">
              <input type="checkbox" checked={showAR}
                onChange={(e) => setShowAR(e.target.checked)} />
              <span className="swatch swatch-ar" />
              active regions {meta ? `${meta.active_region_pct.toFixed(1)}%` : ''}
            </label>
          </dd>
          <dt>MODEL</dt>
          <dd>U-Net · AIA 171+193+304</dd>
          <dt>TEST IOU</dt>
          <dd>CH 0.60 · AR 0.41</dd>
        </dl>
      </aside>

      <main className="sun-stage">
        <div className="sun-halo" aria-hidden="true" />
        <SunCanvas view={view} base={BASE} />
      </main>

      <section className="instruments">
        <div className="spectrum">
          <input
            type="range" min={0} max={2} step={0.01} value={mix}
            onChange={(e) => setMix(parseFloat(e.target.value))}
            aria-label="Wavelength"
          />
          <div className="spectrum-scale" aria-hidden="true">
            {WAVELENGTH_STOPS.map((s) => (
              <button key={s.label} className="spectrum-stop"
                data-active={Math.abs(mix - s.at) < 0.34}
                onClick={() => setMix(s.at)} tabIndex={-1}>
                <span className="stop-label">{s.label}</span>
                <span className="stop-temp">{s.temp}</span>
              </button>
            ))}
          </div>
        </div>
        <label className="wipe-control">
          <span className="cond-key">WIPE&nbsp;RAW</span>
          <input
            type="range" min={0} max={1} step={0.01} value={wipe}
            onChange={(e) => setWipe(parseFloat(e.target.value))}
            aria-label="Reveal raw image from the left"
          />
        </label>
      </section>

      <Conditions />
    </div>
  )
}
