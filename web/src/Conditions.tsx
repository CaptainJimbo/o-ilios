import { useEffect, useState } from 'react'

/** "Conditions now" strip: live space weather fetched client-side from
 * NOAA SWPC's public JSON feeds (CORS-enabled, no auth). Independent of the
 * ML worker — stays live even if segmentation artifacts go stale. */

const SWPC = 'https://services.swpc.noaa.gov'

interface Conditions {
  kp: number | null
  windSpeed: number | null // km/s
  windDensity: number | null // p/cm3
  bz: number | null // nT
}

function stormLabel(kp: number | null): { label: string; alert: boolean } {
  if (kp === null) return { label: 'NO DATA', alert: false }
  if (kp < 4) return { label: 'QUIET', alert: false }
  if (kp < 5) return { label: 'UNSETTLED', alert: false }
  const g = Math.min(5, Math.floor(kp) - 4)
  return { label: `G${g} STORM — AURORA LIKELY`, alert: true }
}

/** Newest non-null value of `field` in an RTSW feed (array of objects with
 * time_tag; not reliably sorted, so scan for the max timestamp). */
const newest = (rows: Record<string, unknown>[] | null, field: string) => {
  let bestTime = ''
  let best: number | null = null
  for (const row of rows ?? []) {
    const v = row[field]
    const t = String(row.time_tag ?? '')
    if (typeof v === 'number' && t > bestTime) {
      bestTime = t
      best = v
    }
  }
  return best
}

export default function Conditions() {
  const [data, setData] = useState<Conditions | null>(null)

  useEffect(() => {
    const load = async () => {
      const grab = (path: string) =>
        fetch(`${SWPC}${path}`).then((r) => r.json()).catch(() => null)
      const [kp, wind, mag] = await Promise.all([
        grab('/products/noaa-planetary-k-index.json'),
        grab('/json/rtsw/rtsw_wind_1m.json'),
        grab('/json/rtsw/rtsw_mag_1m.json'),
      ])
      setData({
        kp: newest(kp, 'Kp'),
        windSpeed: newest(wind, 'proton_speed'),
        windDensity: newest(wind, 'proton_density'),
        bz: newest(mag, 'bz_gsm'),
      })
    }
    load()
    const timer = setInterval(load, 5 * 60 * 1000)
    return () => clearInterval(timer)
  }, [])

  const { label, alert } = stormLabel(data?.kp ?? null)
  const value = (v: number | null, digits = 0, unit = '') =>
    v === null ? '—' : `${v.toFixed(digits)}${unit}`

  return (
    <footer className="conditions" aria-label="Space weather conditions now">
      <span className="cond-item">
        <span className="cond-key">KP INDEX</span>
        <span className="cond-val">{value(data?.kp ?? null, 1)}</span>
      </span>
      <span className="cond-item">
        <span className="cond-key">SOLAR WIND</span>
        <span className="cond-val">{value(data?.windSpeed ?? null, 0, ' km/s')}</span>
      </span>
      <span className="cond-item">
        <span className="cond-key">DENSITY</span>
        <span className="cond-val">{value(data?.windDensity ?? null, 1, ' p/cm³')}</span>
      </span>
      <span className="cond-item">
        <span className="cond-key">BZ</span>
        <span className="cond-val">{value(data?.bz ?? null, 1, ' nT')}</span>
      </span>
      <span className={`cond-item cond-status${alert ? ' cond-alert' : ''}`}>
        <span className="cond-val">{label}</span>
      </span>
      <span className="cond-item cond-credit">
        <span className="cond-key">NOAA SWPC REAL-TIME SOLAR WIND</span>
      </span>
    </footer>
  )
}
