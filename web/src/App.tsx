import { useEffect, useState } from 'react'
import Conditions from './Conditions'
import FlareWatch, { type DiskGeometry, type FlareData } from './FlareWatch'
import SunCanvas, { type SunView } from './SunCanvas'
import Timelapse from './Timelapse'

const BASE = import.meta.env.BASE_URL

interface Meta {
  observation_time: string
  coronal_hole_pct: number
  active_region_pct: number
  disk?: DiskGeometry
}

const WAVELENGTH_STOPS = [
  { at: 0, label: '171 Å', temp: '0.6 MK · quiet corona' },
  { at: 1, label: '193 Å', temp: '1.2 MK · corona & holes' },
  { at: 2, label: '304 Å', temp: '0.05 MK · chromosphere' },
]

export default function App() {
  const [meta, setMeta] = useState<Meta | null>(null)
  const [flares, setFlares] = useState<FlareData | null>(null)
  const [mode, setMode] = useState<'live' | 'timelapse'>('live')
  const [mix, setMix] = useState(1) // start on 193A, where the story is
  const [wipe, setWipe] = useState(0)
  const [showCH, setShowCH] = useState(true)
  const [showAR, setShowAR] = useState(true)
  const [showFlares, setShowFlares] = useState(true)

  useEffect(() => {
    const load = () => {
      // Cache-bust: Pages serves max-age=600, which can outlive a redeploy
      // and pin users to a stale (or degraded) artifact for 10 minutes.
      const bust = `?t=${Date.now()}`
      fetch(`${BASE}live/meta.json${bust}`)
        .then((r) => r.json()).then(setMeta).catch(() => {})
      fetch(`${BASE}live/flares.json${bust}`)
        .then((r) => r.json())
        // Schema-strict: a soft-failed worker can publish a degraded file
        // ({regions: [], error: ...}); bad data must never unmount the app.
        .then((d: FlareData) =>
          setFlares(
            d && Array.isArray(d.regions) && d.full_disk && d.model ? d : null,
          ))
        .catch(() => setFlares(null))
    }
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
          {flares && (
            <>
              <dt>FLARE WATCH · 24H</dt>
              <dd>
                <label className="seg-toggle">
                  <input type="checkbox" checked={showFlares}
                    onChange={(e) => setShowFlares(e.target.checked)} />
                  M+ anywhere{' '}
                  {flares.full_disk.p_m24_any < 0.005
                    ? '<1%' : `${Math.round(flares.full_disk.p_m24_any * 100)}%`}
                </label>
                <span className="flare-noaa-line">
                  NOAA says{' '}
                  {flares.full_disk.noaa_p_m24_any === null
                    ? '—'
                    : `${Math.round(flares.full_disk.noaa_p_m24_any * 100)}%`}
                  {flares.staleness_min !== null
                    && flares.staleness_min > 240 && ' · STALE'}
                </span>
                <span className="flare-noaa-line">
                  LightGBM · TSS {flares.model.test_tss_p5.toFixed(2)} · BSS{' '}
                  {flares.model.test_bss_p5.toFixed(2)}
                </span>
              </dd>
            </>
          )}
        </dl>
      </aside>

      <main className="sun-stage">
        <div className="sun-halo" aria-hidden="true" />
        {mode === 'live' ? (
          <div className="sun-frame">
            <SunCanvas view={view} base={BASE} />
            {showFlares && flares && meta?.disk && flares.regions.length > 0 && (
              <FlareWatch data={flares} disk={meta.disk} />
            )}
          </div>
        ) : (
          <Timelapse base={BASE} />
        )}
      </main>

      <section className="instruments">
        <div className="mode-switch" role="tablist" aria-label="View mode">
          <button role="tab" aria-selected={mode === 'live'}
            data-active={mode === 'live'} onClick={() => setMode('live')}>
            LIVE
          </button>
          <button role="tab" aria-selected={mode === 'timelapse'}
            data-active={mode === 'timelapse'}
            onClick={() => setMode('timelapse')}>
            TIME-LAPSE
          </button>
        </div>
        {mode === 'live' && (
          <>
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
          </>
        )}
      </section>

      <Conditions />
    </div>
  )
}
