import { useEffect, useRef, useState } from 'react'

/** WebGL view of the sun: three wavelength textures morphed by `mix`,
 * segmentation mask composited additively (luminous), with a raw/annotated
 * wipe. All imagery comes from the worker's artifacts in /live/. */

const VERT = `#version 300 es
in vec2 aPos;
out vec2 vUv;
void main() {
  vUv = aPos * 0.5 + 0.5;
  gl_Position = vec4(aPos, 0.0, 1.0);
}`

const FRAG = `#version 300 es
precision highp float;
uniform sampler2D uSun171, uSun193, uSun304, uMask;
uniform float uMix;   // 0 -> 171A, 1 -> 193A, 2 -> 304A
uniform float uWipe;  // overlays drawn where uv.x > uWipe
uniform float uCH, uAR;
in vec2 vUv;
out vec4 outColor;
void main() {
  vec2 uv = vec2(vUv.x, 1.0 - vUv.y);
  vec3 a = texture(uSun171, uv).rgb;
  vec3 b = texture(uSun193, uv).rgb;
  vec3 c = texture(uSun304, uv).rgb;
  vec3 sun = uMix < 1.0 ? mix(a, b, uMix) : mix(b, c, uMix - 1.0);

  vec4 m = texture(uMask, uv);
  float isCH = step(m.r, m.b);               // cyan has blue > red
  float gate = mix(uAR, uCH, isCH) * m.a;
  float show = step(uWipe, vUv.x);
  vec3 overlay = m.rgb * gate * show * 0.85; // additive => luminous
  outColor = vec4(sun + overlay, 1.0);
}`

export interface SunView {
  mix: number
  wipe: number
  showCH: boolean
  showAR: boolean
}

export default function SunCanvas({ view, base, obsTime }: {
  view: SunView
  base: string
  /** meta.json observation_time — a change means fresh artifacts exist. */
  obsTime?: string
}) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const uniformsRef = useRef<{
    gl: WebGL2RenderingContext
    set: (v: SunView) => void
    reload: () => void
  } | null>(null)
  // Bumped on GPU context restore to re-run the whole GL setup.
  const [glEpoch, setGlEpoch] = useState(0)

  // Unmount-only concerns: context-loss listeners + explicit release.
  // (Separate from the setup effect: its cleanup also runs on glEpoch
  // re-runs, and calling loseContext there would kill the context we are
  // in the middle of restoring.)
  useEffect(() => {
    const canvas = canvasRef.current!
    // Recover from GPU resets / mobile background eviction — otherwise the
    // canvas stays permanently black (we have no rAF loop to notice).
    const onLost = (e: Event) => e.preventDefault()
    const onRestored = () => setGlEpoch((n) => n + 1)
    canvas.addEventListener('webglcontextlost', onLost)
    canvas.addEventListener('webglcontextrestored', onRestored)
    return () => {
      canvas.removeEventListener('webglcontextlost', onLost)
      canvas.removeEventListener('webglcontextrestored', onRestored)
      // NB: no eager loseContext() here — the canvas DOM node outlives a
      // StrictMode/HMR remount cycle, and poisoning its context breaks the
      // next setup. Browsers evict surplus contexts oldest-first anyway.
    }
  }, [])

  useEffect(() => {
    const canvas = canvasRef.current!
    const gl = canvas.getContext('webgl2')
    if (!gl) return

    const compile = (type: number, src: string) => {
      const s = gl.createShader(type)!
      gl.shaderSource(s, src)
      gl.compileShader(s)
      if (!gl.getShaderParameter(s, gl.COMPILE_STATUS))
        throw new Error(gl.getShaderInfoLog(s) ?? 'shader error')
      return s
    }
    const program = gl.createProgram()!
    gl.attachShader(program, compile(gl.VERTEX_SHADER, VERT))
    gl.attachShader(program, compile(gl.FRAGMENT_SHADER, FRAG))
    gl.linkProgram(program)
    gl.useProgram(program)

    const quad = gl.createBuffer()
    gl.bindBuffer(gl.ARRAY_BUFFER, quad)
    gl.bufferData(gl.ARRAY_BUFFER,
      new Float32Array([-1, -1, 1, -1, -1, 1, 1, 1]), gl.STATIC_DRAW)
    const loc = gl.getAttribLocation(program, 'aPos')
    gl.enableVertexAttribArray(loc)
    gl.vertexAttribPointer(loc, 2, gl.FLOAT, false, 0, 0)

    const loadTexture = (unit: number, url: string, uniform: string) => {
      const tex = gl.createTexture()
      gl.activeTexture(gl.TEXTURE0 + unit)
      gl.bindTexture(gl.TEXTURE_2D, tex)
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, 1, 1, 0, gl.RGBA,
        gl.UNSIGNED_BYTE, new Uint8Array([0, 0, 0, 0]))
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR)
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE)
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE)
      gl.uniform1i(gl.getUniformLocation(program, uniform), unit)
      const img = new Image()
      img.onload = () => {
        gl.activeTexture(gl.TEXTURE0 + unit)
        gl.bindTexture(gl.TEXTURE_2D, tex)
        gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, img)
        draw()
      }
      img.src = url
    }

    const u = (name: string) => gl.getUniformLocation(program, name)
    let current = view
    const draw = () => {
      const size = canvas.clientWidth * devicePixelRatio
      if (canvas.width !== size) {
        canvas.width = canvas.height = size
        gl.viewport(0, 0, size, size)
      }
      gl.uniform1f(u('uMix'), current.mix)
      gl.uniform1f(u('uWipe'), current.wipe)
      gl.uniform1f(u('uCH'), current.showCH ? 1 : 0)
      gl.uniform1f(u('uAR'), current.showAR ? 1 : 0)
      gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4)
    }

    // Cache-bust texture loads: Pages pins assets for 600 s, and the whole
    // point of this canvas is that the sun is CURRENT.
    const loadAll = () => {
      const bust = `?t=${Date.now()}`
      loadTexture(0, `${base}live/sun_171.png${bust}`, 'uSun171')
      loadTexture(1, `${base}live/sun_193.png${bust}`, 'uSun193')
      loadTexture(2, `${base}live/sun_304.png${bust}`, 'uSun304')
      loadTexture(3, `${base}live/mask.png${bust}`, 'uMask')
    }
    loadAll()

    uniformsRef.current = {
      gl,
      set: (v) => {
        current = v
        draw()
      },
      reload: loadAll,
    }
    const observer = new ResizeObserver(draw)
    observer.observe(canvas)
    return () => observer.disconnect()
  }, [base, glEpoch])

  useEffect(() => {
    uniformsRef.current?.set(view)
  }, [view])

  // New observation time -> the worker published fresh artifacts. Reload
  // textures in place (the tab stays "living" without a page reload).
  useEffect(() => {
    if (obsTime) uniformsRef.current?.reload()
  }, [obsTime])

  return <canvas ref={canvasRef} className="sun-canvas" aria-label="Live sun" />
}
