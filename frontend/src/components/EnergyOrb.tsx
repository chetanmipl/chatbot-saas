// ============================================================
// EnergyOrb — Pure CSS/Canvas orb with:
// - Fresnel glow rings
// - Animated energy lines
// - Pulsing core
// - Orbiting particles
// No Spline dependency — full control over look
// ============================================================

import { useEffect, useRef } from 'react'

interface EnergyOrbProps {
  size?: number
}

export default function EnergyOrb({ size = 480 }: EnergyOrbProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null)

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const ctx = canvas.getContext('2d')
    if (!ctx) return

    canvas.width  = size * 2   // retina
    canvas.height = size * 2
    canvas.style.width  = `${size}px`
    canvas.style.height = `${size}px`

    const cx = canvas.width  / 2
    const cy = canvas.height / 2
    const r  = canvas.width  * 0.32   // core radius

    let frame = 0
    let animId: number

    // Particles orbiting the orb
    const particles = Array.from({ length: 60 }, (_, i) => ({
      angle:  (i / 60) * Math.PI * 2,
      speed:  0.002 + Math.random() * 0.004,
      radius: r * (1.1 + Math.random() * 0.6),
      size:   1 + Math.random() * 3,
      opacity: 0.3 + Math.random() * 0.7,
      color:   Math.random() > 0.5 ? '#ea580c' : '#fb923c',
    }))

    // Energy arc lines
    const arcs = Array.from({ length: 8 }, (_, i) => ({
      startAngle: (i / 8) * Math.PI * 2,
      length: 0.3 + Math.random() * 0.8,
      radius: r * (0.95 + Math.random() * 0.15),
      speed:  0.008 + Math.random() * 0.015,
      opacity: 0.4 + Math.random() * 0.5,
      width: 1 + Math.random() * 2,
    }))

    function draw() {
      if (!ctx || !canvas) return
      ctx.clearRect(0, 0, canvas.width, canvas.height)
      frame++

      // ── Deep core glow (innermost) ──────────────────────
      const coreGrad = ctx.createRadialGradient(cx, cy, 0, cx, cy, r * 0.5)
      coreGrad.addColorStop(0,   'rgba(251,146,60,0.95)')
      coreGrad.addColorStop(0.3, 'rgba(234,88,12,0.7)')
      coreGrad.addColorStop(1,   'rgba(120,53,15,0)')
      ctx.beginPath()
      ctx.arc(cx, cy, r * 0.5, 0, Math.PI * 2)
      ctx.fillStyle = coreGrad
      ctx.fill()

      // ── Main sphere body ────────────────────────────────
      const sphereGrad = ctx.createRadialGradient(
        cx - r * 0.25, cy - r * 0.25, 0,
        cx, cy, r
      )
      sphereGrad.addColorStop(0,    'rgba(251,146,60,0.45)')
      sphereGrad.addColorStop(0.4,  'rgba(234,88,12,0.25)')
      sphereGrad.addColorStop(0.75, 'rgba(120,53,15,0.15)')
      sphereGrad.addColorStop(1,    'rgba(12,10,9,0)')
      ctx.beginPath()
      ctx.arc(cx, cy, r, 0, Math.PI * 2)
      ctx.fillStyle = sphereGrad
      ctx.fill()

      // ── Fresnel rim light (edge glow) ───────────────────
      const fresnelGrad = ctx.createRadialGradient(cx, cy, r * 0.75, cx, cy, r * 1.05)
      fresnelGrad.addColorStop(0,   'rgba(251,146,60,0)')
      fresnelGrad.addColorStop(0.6, 'rgba(234,88,12,0.2)')
      fresnelGrad.addColorStop(0.9, 'rgba(251,146,60,0.55)')
      fresnelGrad.addColorStop(1,   'rgba(234,88,12,0)')
      ctx.beginPath()
      ctx.arc(cx, cy, r * 1.05, 0, Math.PI * 2)
      ctx.fillStyle = fresnelGrad
      ctx.fill()

      // ── Outer atmospheric glow (multiple rings) ─────────
      ;[1.3, 1.6, 2.0, 2.6].forEach((mult, i) => {
        const pulse = Math.sin(frame * 0.018 + i * 0.8) * 0.12
        const alpha = (0.18 - i * 0.035) + pulse
        const atmGrad = ctx.createRadialGradient(cx, cy, r * (mult - 0.25), cx, cy, r * mult)
        atmGrad.addColorStop(0,   `rgba(234,88,12,${alpha})`)
        atmGrad.addColorStop(0.5, `rgba(251,146,60,${alpha * 0.5})`)
        atmGrad.addColorStop(1,   'rgba(120,53,15,0)')
        ctx.beginPath()
        ctx.arc(cx, cy, r * mult, 0, Math.PI * 2)
        ctx.fillStyle = atmGrad
        ctx.fill()
      })

      // ── Rotating energy ring ────────────────────────────
      const ringAngle = frame * 0.012
      ctx.save()
      ctx.translate(cx, cy)
      ctx.rotate(ringAngle)
      ctx.beginPath()
      ctx.ellipse(0, 0, r * 1.2, r * 0.3, 0, 0, Math.PI * 2)
      ctx.strokeStyle = 'rgba(251,146,60,0.35)'
      ctx.lineWidth = 2
      ctx.stroke()
      ctx.restore()

      // Second ring at different tilt
      ctx.save()
      ctx.translate(cx, cy)
      ctx.rotate(-ringAngle * 0.7)
      ctx.beginPath()
      ctx.ellipse(0, 0, r * 1.15, r * 0.2, Math.PI / 4, 0, Math.PI * 2)
      ctx.strokeStyle = 'rgba(234,88,12,0.25)'
      ctx.lineWidth = 1.5
      ctx.stroke()
      ctx.restore()

      // ── Energy arc lines (lightning-like) ───────────────
      arcs.forEach((arc) => {
        arc.startAngle += arc.speed
        const pulse = 0.5 + 0.5 * Math.sin(frame * 0.04 + arc.startAngle)

        ctx.beginPath()
        ctx.arc(
          cx, cy,
          arc.radius,
          arc.startAngle,
          arc.startAngle + arc.length
        )
        ctx.strokeStyle = `rgba(251,146,60,${arc.opacity * pulse})`
        ctx.lineWidth = arc.width
        ctx.lineCap = 'round'
        ctx.stroke()
      })

      // ── Orbiting particles ───────────────────────────────
      particles.forEach((p) => {
        p.angle += p.speed
        const px = cx + Math.cos(p.angle) * p.radius
        const py = cy + Math.sin(p.angle) * p.radius * 0.4 // flatten to ellipse
        const pulse = 0.6 + 0.4 * Math.sin(frame * 0.05 + p.angle)

        ctx.beginPath()
        ctx.arc(px, py, p.size, 0, Math.PI * 2)
        ctx.fillStyle = p.color
        ctx.globalAlpha = p.opacity * pulse
        ctx.fill()

        // Glow behind particle
        const gGrad = ctx.createRadialGradient(px, py, 0, px, py, p.size * 3)
        gGrad.addColorStop(0, p.color.replace(')', ',0.4)').replace('rgb', 'rgba'))
        gGrad.addColorStop(1, 'rgba(0,0,0,0)')
        ctx.beginPath()
        ctx.arc(px, py, p.size * 3, 0, Math.PI * 2)
        ctx.fillStyle = gGrad
        ctx.fill()

        ctx.globalAlpha = 1
      })

      // ── Core highlight (specular) ────────────────────────
      const specGrad = ctx.createRadialGradient(
        cx - r * 0.28, cy - r * 0.28, 0,
        cx - r * 0.28, cy - r * 0.28, r * 0.45
      )
      specGrad.addColorStop(0,   'rgba(255,247,237,0.5)')
      specGrad.addColorStop(0.4, 'rgba(255,247,237,0.15)')
      specGrad.addColorStop(1,   'rgba(255,247,237,0)')
      ctx.beginPath()
      ctx.arc(cx, cy, r, 0, Math.PI * 2)
      ctx.fillStyle = specGrad
      ctx.fill()

      // ── Pulse breathing effect ────────────────────────────
      const breathe = 0.97 + 0.03 * Math.sin(frame * 0.025)
      canvas.style.transform = `scale(${breathe})`

      animId = requestAnimationFrame(draw)
    }

    draw()
    return () => cancelAnimationFrame(animId)
  }, [size])

  return (
    <canvas
      ref={canvasRef}
      style={{
        borderRadius: '50%',
        cursor: 'none',
      }}
    />
  )
}