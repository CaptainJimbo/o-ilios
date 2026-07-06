import { useEffect, useMemo, useState } from 'react'

/** Time-lapse mode: a curated historical sequence (AR 13664, May 2024 — the
 * Gannon storm region) segmented by the U-Net frame by frame. Plain stacked
 * images instead of the WebGL canvas: one wavelength, no morph needed. */

interface Frame {
  id: string
  time: string
  coronal_hole_pct: number
  active_region_pct: number
}

interface Index {
  title: string
  subtitle: string
  frames: Frame[]
}

export default function Timelapse({ base }: { base: string }) {
  const [index, setIndex] = useState<Index | null>(null)
  const [at, setAt] = useState(0)

  useEffect(() => {
    fetch(`${base}timelapse/index.json`)
      .then((r) => r.json())
      .then((idx: Index) => {
        setIndex(idx)
        // Warm the cache so scrubbing is instant.
        for (const f of idx.frames) {
          new Image().src = `${base}timelapse/sun_${f.id}.jpg`
          new Image().src = `${base}timelapse/mask_${f.id}.png`
        }
      })
      .catch(() => {})
  }, [base])

  const frame = index?.frames[at]
  const when = useMemo(
    () => (frame ? frame.time.slice(0, 16).replace('T', ' ') + ' UT' : ''),
    [frame],
  )

  if (!index || !frame) return <div className="timelapse-loading">loading sequence…</div>

  return (
    <>
      <div className="timelapse-view">
        <img src={`${base}timelapse/sun_${frame.id}.jpg`} alt="" draggable={false} />
        <img src={`${base}timelapse/mask_${frame.id}.png`} alt=""
          className="timelapse-mask" draggable={false} />
      </div>
      <div className="timelapse-controls">
        <div className="timelapse-caption">
          <span className="cond-key">{index.title.toUpperCase()}</span>
          <span className="timelapse-when">{when}</span>
          <span className="timelapse-stats">
            CH {frame.coronal_hole_pct.toFixed(1)}% · AR{' '}
            {frame.active_region_pct.toFixed(1)}%
          </span>
        </div>
        <input
          type="range" min={0} max={index.frames.length - 1} step={1}
          value={at} onChange={(e) => setAt(parseInt(e.target.value, 10))}
          aria-label={`Time-lapse position, ${when}`}
        />
      </div>
    </>
  )
}
