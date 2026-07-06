/** Flare Watch: per-region P(M+ flare within 24 h) badges positioned on the
 * live disk, from the worker's flares.json + meta.json disk geometry.
 * Badge position (north-up image fractions):
 *   left = cx + fx * r,  top = cy - fy * r.  */

export interface DiskGeometry {
  cx_frac: number
  cy_frac: number
  r_frac: number
}

export interface FlareRegion {
  harpnum: number
  noaa_ar: number | null
  fx: number
  fy: number
  p_m24: number
  alert: boolean
  noaa_p_m?: number | null
}

export interface FlareData {
  t_rec_utc: string | null
  staleness_min: number | null
  regions: FlareRegion[]
  full_disk: { p_m24_any: number; noaa_p_m24_any: number | null }
  model: { test_tss_p5: number; test_bss_p5: number }
}

const pct = (p: number) => (p < 0.005 ? '<1%' : `${Math.round(p * 100)}%`)

export default function FlareWatch({ data, disk }: {
  data: FlareData
  disk: DiskGeometry
}) {
  return (
    <div className="flare-layer" aria-label="Flare Watch badges">
      {data.regions.map((r) => (
        <div
          key={r.harpnum}
          className={`flare-badge${r.alert ? ' flare-alert' : ''}`}
          style={{
            left: `${(disk.cx_frac + r.fx * disk.r_frac) * 100}%`,
            top: `${(disk.cy_frac - r.fy * disk.r_frac) * 100}%`,
          }}
        >
          <span className="flare-ring" aria-hidden="true" />
          <span className="flare-label">
            {r.noaa_ar ? `AR ${r.noaa_ar % 10000}` : `HARP ${r.harpnum}`}
            <b> {pct(r.p_m24)}</b>
          </span>
        </div>
      ))}
    </div>
  )
}
