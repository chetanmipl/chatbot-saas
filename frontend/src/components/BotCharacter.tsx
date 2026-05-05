// ============================================================
// BotCharacter — Clean redesign
// ORBACK: Energy orb (tight fresnel, fits sphere exactly)
// BOTFACE: Simple cute face matching Spline design
//          (circle head + warm ellipse eyes)
// FLIP: GSAP scroll scrub drives rotateY 0→180
//       Scroll down = flip to face
//       Scroll up = flip back to orb
// ============================================================

import { useEffect, useRef, forwardRef } from 'react'
import { useCursor } from '../hooks/useCursor'

interface BotCharacterProps {
  scrollProgress: number   // 0→1 driven by parent GSAP scrub
  size?: number
}

export default function BotCharacter({
  scrollProgress,
  size = 480,
}: BotCharacterProps) {
  const cursor = useCursor()
  const cardRef = useRef<HTMLDivElement>(null)
  const containerRef = useRef<HTMLDivElement>(null)

  // Flip angle: 0 = orb, 180 = face
  // Only flips during scroll 0.25→0.65
  const flipProgress = Math.max(0, Math.min(1,
    (scrollProgress - 0.25) / 0.4
  ))
  const rotateY = flipProgress * 180

  // Subtle cursor tilt on top of flip
  const tiltX = cursor.y * 7
  const tiltY = -cursor.x * 7

  return (
    <div
      ref={containerRef}
      style={{
        width: size,
        height: size,
        maxWidth: '88vw',
        maxHeight: '88vw',
        position: 'relative',
        perspective: '1100px',
      }}
    >
      {/* Ambient glow — stays behind whole thing */}
      <div style={{
        position: 'absolute',
        inset: '-18%',
        borderRadius: '50%',
        background: 'radial-gradient(ellipse, rgba(120,53,15,0.38) 0%, rgba(234,88,12,0.1) 55%, transparent 75%)',
        animation: 'pulse-glow 4s ease-in-out infinite',
        pointerEvents: 'none',
        zIndex: 0,
      }} />

      {/* Flip card */}
      <div
        ref={cardRef}
        style={{
          width: '100%',
          height: '100%',
          position: 'relative',
          transformStyle: 'preserve-3d',
          // Combine flip + cursor tilt
          transform: `rotateX(${tiltX}deg) rotateY(${tiltY + rotateY}deg)`,
          transition: 'transform 0.08s ease-out',
          zIndex: 1,
        }}
      >
        {/* BACK: Energy Orb */}
        <div style={{
          position: 'absolute', inset: 0,
          backfaceVisibility: 'hidden',
          WebkitBackfaceVisibility: 'hidden',
        }}>
          <OrbBack size={size} />
        </div>

        {/* FRONT: Cute bot face */}
        <div style={{
          position: 'absolute', inset: 0,
          backfaceVisibility: 'hidden',
          WebkitBackfaceVisibility: 'hidden',
          transform: 'rotateY(180deg)',
        }}>
          <BotFace size={size} cursor={cursor} />
        </div>
      </div>
    </div>
  )
}

// ─────────────────────────────────────────────────────────────
// ORB BACK
// Tight fresnel = thin bright ring exactly at sphere edge
// Neural lines, rotating rings, orbiting particles inside sphere
// ─────────────────────────────────────────────────────────────
function OrbBack({ size }: { size: number }) {
  const canvasRef = useRef<HTMLCanvasElement>(null)

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const ctx = canvas.getContext('2d')
    if (!ctx) return

    const dpr = Math.min(window.devicePixelRatio || 1, 2)
    canvas.width = size * dpr
    canvas.height = size * dpr
    canvas.style.width = `${size}px`
    canvas.style.height = `${size}px`
    ctx.scale(dpr, dpr)

    const cx = size / 2
    const cy = size / 2
    const R = size * 0.44   // sphere radius — fills the box well

    let frame = 0
    let animId: number

    // Orbital particles — INSIDE and AROUND the sphere
    const particles = Array.from({ length: 70 }, (_, i) => ({
      angle: (i / 70) * Math.PI * 2 + Math.random() * 0.3,
      speed: 0.004 + Math.random() * 0.006,
      orbitRx: R * (0.65 + Math.random() * 0.45),
      orbitRy: R * (0.12 + Math.random() * 0.35),
      tilt: Math.random() * Math.PI,
      sz: 1.2 + Math.random() * 2.2,
      opacity: 0.35 + Math.random() * 0.65,
      color: ['#ea580c', '#fb923c', '#fdba74', '#c2410c'][Math.floor(Math.random() * 4)],
    }))

    // 3 rotating rings at different tilts
    const rings = [
      { rx: R * 0.88, ry: R * 0.22, rot: 0, speed: 0.009, alpha: 0.45 },
      { rx: R * 0.82, ry: R * 0.17, rot: Math.PI / 3, speed: -0.007, alpha: 0.32 },
      { rx: R * 0.76, ry: R * 0.14, rot: Math.PI * 0.7, speed: 0.011, alpha: 0.25 },
    ]

    // Energy arcs (spark-like)
    const arcs = Array.from({ length: 10 }, (_, i) => ({
      a: (i / 10) * Math.PI * 2,
      len: 0.18 + Math.random() * 0.55,
      rad: R * (0.78 + Math.random() * 0.18),
      spd: (0.007 + Math.random() * 0.011) * (Math.random() > 0.5 ? 1 : -1),
      op: 0.3 + Math.random() * 0.45,
      w: 0.8 + Math.random() * 1.8,
    }))

    function draw() {
      if (!ctx || !canvas) return
      ctx.clearRect(0, 0, size, size)
      frame++

      // ── Clip everything inside sphere ──────────────────
      ctx.save()
      ctx.beginPath()
      ctx.arc(cx, cy, R, 0, Math.PI * 2)
      ctx.clip()

      // Body gradient — rich dark brown/obsidian
      const bodyG = ctx.createRadialGradient(
        cx - R * 0.22, cy - R * 0.22, R * 0.04,
        cx, cy, R
      )
      bodyG.addColorStop(0, 'rgba(92,40,10,1)')
      bodyG.addColorStop(0.3, 'rgba(44,18,5,1)')
      bodyG.addColorStop(0.65, 'rgba(22,10,4,1)')
      bodyG.addColorStop(1, 'rgba(12,10,9,1)')
      ctx.fillStyle = bodyG
      ctx.fillRect(0, 0, size, size)

      // Neural lines (inside sphere)
      for (let i = 0; i < 9; i++) {
        const a1 = (i / 9) * Math.PI * 2 + frame * 0.004
        const a2 = ((i + 3.5) / 9) * Math.PI * 2 + frame * 0.004
        const x1 = cx + Math.cos(a1) * R * 0.58
        const y1 = cy + Math.sin(a1) * R * 0.58
        const x2 = cx + Math.cos(a2) * R * 0.58
        const y2 = cy + Math.sin(a2) * R * 0.58
        const pulse = 0.3 + 0.7 * Math.abs(Math.sin(frame * 0.035 + i * 0.7))

        ctx.beginPath()
        ctx.moveTo(x1, y1)
        ctx.quadraticCurveTo(cx, cy, x2, y2)
        ctx.strokeStyle = `rgba(251,146,60,${0.12 * pulse})`
        ctx.lineWidth = 0.8
        ctx.stroke()

        // Node
        ctx.beginPath()
        ctx.arc(x1, y1, 2, 0, Math.PI * 2)
        ctx.fillStyle = `rgba(251,146,60,${0.5 * pulse})`
        ctx.fill()
      }

      // Rotating rings (inside clip)
      rings.forEach((rg) => {
        rg.rot += rg.speed
        ctx.save()
        ctx.translate(cx, cy)
        ctx.rotate(rg.rot)
        ctx.beginPath()
        ctx.ellipse(0, 0, rg.rx, rg.ry, 0, 0, Math.PI * 2)
        const p = 0.6 + 0.4 * Math.sin(frame * 0.02)
        ctx.strokeStyle = `rgba(251,146,60,${rg.alpha * p})`
        ctx.lineWidth = 1.4
        ctx.stroke()
        ctx.restore()
      })

      // Energy arcs
      arcs.forEach((arc) => {
        arc.a += arc.spd
        const p = 0.4 + 0.6 * Math.abs(Math.sin(frame * 0.045 + arc.a))
        ctx.beginPath()
        ctx.arc(cx, cy, arc.rad, arc.a, arc.a + arc.len)
        ctx.strokeStyle = `rgba(251,146,60,${arc.op * p})`
        ctx.lineWidth = arc.w
        ctx.lineCap = 'round'
        ctx.stroke()
      })

      // Particles
      particles.forEach((p) => {
        p.angle += p.speed
        const px = cx + Math.cos(p.angle + p.tilt) * p.orbitRx
        const py = cy + Math.sin(p.angle) * p.orbitRy
        const pulse = 0.5 + 0.5 * Math.sin(frame * 0.07 + p.angle * 2)

        ctx.beginPath()
        ctx.arc(px, py, p.sz, 0, Math.PI * 2)
        ctx.fillStyle = p.color
        ctx.globalAlpha = p.opacity * pulse
        ctx.fill()
        ctx.globalAlpha = 1
      })

      // Glowing nucleus
      const ns = R * 0.14 + Math.sin(frame * 0.05) * R * 0.025
      const nG = ctx.createRadialGradient(cx, cy, 0, cx, cy, ns)
      nG.addColorStop(0, 'rgba(255,247,237,1)')
      nG.addColorStop(0.3, 'rgba(251,146,60,0.9)')
      nG.addColorStop(0.7, 'rgba(234,88,12,0.5)')
      nG.addColorStop(1, 'rgba(120,53,15,0)')
      ctx.beginPath()
      ctx.arc(cx, cy, ns, 0, Math.PI * 2)
      ctx.fillStyle = nG
      ctx.fill()

      // Specular highlight (top-left)
      const specG = ctx.createRadialGradient(
        cx - R * 0.28, cy - R * 0.28, 0,
        cx - R * 0.28, cy - R * 0.28, R * 0.38
      )
      specG.addColorStop(0, 'rgba(255,247,237,0.28)')
      specG.addColorStop(0.5, 'rgba(255,247,237,0.07)')
      specG.addColorStop(1, 'rgba(255,247,237,0)')
      ctx.fillStyle = specG
      ctx.fillRect(0, 0, size, size)

      ctx.restore() // end sphere clip

      // ── Fresnel rim — TIGHT to sphere edge ────────────
      // Only a thin ring right at the boundary — NOT outside
      const fresnelG = ctx.createRadialGradient(
        cx, cy, R * 0.84,   // inner edge of fresnel
        cx, cy, R           // outer = exactly sphere edge
      )
      fresnelG.addColorStop(0, 'rgba(234,88,12,0)')
      fresnelG.addColorStop(0.5, 'rgba(251,146,60,0.22)')
      fresnelG.addColorStop(0.82, 'rgba(251,146,60,0.60)')
      fresnelG.addColorStop(1, 'rgba(253,186,116,0.80)')

      ctx.beginPath()
      ctx.arc(cx, cy, R, 0, Math.PI * 2)
      ctx.fillStyle = fresnelG
      ctx.fill()

      animId = requestAnimationFrame(draw)
    }

    draw()
    return () => cancelAnimationFrame(animId)
  }, [size])

  return (
    <canvas
      ref={canvasRef}
      style={{ display: 'block', borderRadius: '50%' }}
    />
  )
}

// ─────────────────────────────────────────────────────────────
// BOT FACE — matches your Spline design:
// Dark circle head + warm ellipse eyes + ellipse circle body
// Clean, cute, minimal — NOT scary
// ─────────────────────────────────────────────────────────────
function BotFace({
  size,
  cursor,
}: {
  size: number
  cursor: { x: number; y: number }
}) {
  const svgRef = useRef<SVGSVGElement>(null)
  const leftEyeRef = useRef<SVGGElement>(null)
  const rightEyeRef = useRef<SVGGElement>(null)
  const blinkRef = useRef<ReturnType<typeof setTimeout> | undefined>(undefined)

  // Eye blink loop
  useEffect(() => {
    const eyeEls = [leftEyeRef.current, rightEyeRef.current]

    const scheduleBlink = () => {
      const delay = 2500 + Math.random() * 2000
      blinkRef.current = setTimeout(() => {
        // Squish eyes to blink
        eyeEls.forEach(el => {
          if (el) {
            el.style.transform = 'scaleY(0.08)'
            el.style.transformBox = 'fill-box'
            el.style.transformOrigin = 'center'
            el.style.transition = 'transform 0.08s ease'
          }
        })
        setTimeout(() => {
          eyeEls.forEach(el => {
            if (el) {
              el.style.transform = 'scaleY(1)'
              el.style.transition = 'transform 0.1s ease'
            }
          })
          scheduleBlink()
        }, 130)
      }, delay)
    }

    scheduleBlink()
    return () => clearTimeout(blinkRef.current)
  }, [])

  // Look-at — pupils follow cursor
  const pupilOffset = {
    x: cursor.x * 5,
    y: -cursor.y * 4,
  }

  // Eye Y position — look up slightly when cursor is high
  const eyeLookY = -cursor.y * 3

  // Scale: 200 viewBox units = size px
  const vb = 200

  return (
    <svg
      ref={svgRef}
      width={size}
      height={size}
      viewBox={`0 0 ${vb} ${vb}`}
      style={{ display: 'block', overflow: 'visible' }}
    >
      <defs>
        {/* Main head - dark sphere gradient */}
        <radialGradient id="faceBody" cx="40%" cy="35%" r="65%">
          <stop offset="0%" stopColor="#2d1a0e" />
          <stop offset="45%" stopColor="#1a0e06" />
          <stop offset="100%" stopColor="#0c0a09" />
        </radialGradient>

        {/* Fresnel rim — tight to face edge */}
        <radialGradient id="faceRim" cx="50%" cy="50%" r="50%">
          <stop offset="0%" stopColor="rgba(0,0,0,0)" />
          <stop offset="75%" stopColor="rgba(234,88,12,0)" />
          <stop offset="88%" stopColor="rgba(234,88,12,0.18)" />
          <stop offset="95%" stopColor="rgba(251,146,60,0.55)" />
          <stop offset="100%" stopColor="rgba(253,186,116,0.7)" />
        </radialGradient>

        {/* Eye fill — warm orange glow */}
        <radialGradient id="eyeFill" cx="40%" cy="35%" r="60%">
          <stop offset="0%" stopColor="#fdba74" />
          <stop offset="45%" stopColor="#fb923c" />
          <stop offset="100%" stopColor="#ea580c" />
        </radialGradient>

        {/* Eye glow filter */}
        <filter id="eyeGlowF" x="-40%" y="-40%" width="180%" height="180%">
          <feGaussianBlur stdDeviation="2.5" result="blur" />
          <feMerge>
            <feMergeNode in="blur" />
            <feMergeNode in="SourceGraphic" />
          </feMerge>
        </filter>

        {/* Soft outer glow */}
        <filter id="faceGlowF" x="-20%" y="-20%" width="140%" height="140%">
          <feGaussianBlur stdDeviation="4" result="blur" />
          <feMerge>
            <feMergeNode in="blur" />
            <feMergeNode in="SourceGraphic" />
          </feMerge>
        </filter>

        {/* Clip to head circle */}
        <clipPath id="headCircle">
          <circle cx="100" cy="105" r="70" />
        </clipPath>

        {/* Specular highlight */}
        <radialGradient id="faceSpec" cx="35%" cy="30%" r="38%">
          <stop offset="0%" stopColor="rgba(255,247,237,0.20)" />
          <stop offset="100%" stopColor="rgba(255,247,237,0)" />
        </radialGradient>
      </defs>

      {/* ── Outer atmospheric haze ── */}
      <circle cx="100" cy="105" r="82"
        fill="none"
        stroke="rgba(234,88,12,0.1)"
        strokeWidth="18" />

      {/* ── Head body ── */}
      <circle cx="100" cy="105" r="70"
        fill="url(#faceBody)" />

      {/* ── Fresnel rim (tight) ── */}
      <circle cx="100" cy="105" r="70"
        fill="url(#faceRim)" />

      {/* ── Subtle circuit/texture lines ── */}
      <g clipPath="url(#headCircle)" opacity="0.07">
        <line x1="60" y1="80" x2="140" y2="80" stroke="#fb923c" strokeWidth="0.6" />
        <line x1="60" y1="95" x2="140" y2="95" stroke="#fb923c" strokeWidth="0.6" />
        <line x1="80" y1="60" x2="80" y2="150" stroke="#fb923c" strokeWidth="0.6" />
        <line x1="120" y1="60" x2="120" y2="150" stroke="#fb923c" strokeWidth="0.6" />
      </g>

      {/* ── LEFT EYE — ellipse shape like your Spline ── */}
      <g
        ref={leftEyeRef}
        style={{ transformBox: 'fill-box', transformOrigin: 'center' }}
      >
        {/* Eye glow halo */}
        <ellipse
          cx={72} cy={100 + eyeLookY}
          rx="18" ry="13"
          fill="rgba(234,88,12,0.15)"
          filter="url(#eyeGlowF)"
        />
        {/* Eye socket (dark background) */}
        <ellipse
          cx={72} cy={100 + eyeLookY}
          rx="13" ry="9.5"
          fill="#0c0a09"
        />
        {/* Iris — warm orange ellipse (matching your Spline) */}
        <ellipse
          cx={72 + pupilOffset.x * 0.5}
          cy={100 + eyeLookY + pupilOffset.y * 0.5}
          rx="10" ry="7.5"
          fill="url(#eyeFill)"
          filter="url(#eyeGlowF)"
        />
        {/* Pupil */}
        <ellipse
          cx={72 + pupilOffset.x}
          cy={100 + eyeLookY + pupilOffset.y}
          rx="4.5" ry="3.5"
          fill="#0c0a09"
        />
        {/* Specular dot */}
        <ellipse
          cx={69 + pupilOffset.x * 0.3}
          cy={97.5 + eyeLookY + pupilOffset.y * 0.3}
          rx="2.2" ry="1.6"
          fill="rgba(255,247,237,0.85)"
        />
      </g>

      {/* ── RIGHT EYE ── */}
      <g
        ref={rightEyeRef}
        style={{ transformBox: 'fill-box', transformOrigin: 'center' }}
      >
        <ellipse
          cx={128} cy={100 + eyeLookY}
          rx="18" ry="13"
          fill="rgba(234,88,12,0.15)"
          filter="url(#eyeGlowF)"
        />
        <ellipse
          cx={128} cy={100 + eyeLookY}
          rx="13" ry="9.5"
          fill="#0c0a09"
        />
        <ellipse
          cx={128 + pupilOffset.x * 0.5}
          cy={100 + eyeLookY + pupilOffset.y * 0.5}
          rx="10" ry="7.5"
          fill="url(#eyeFill)"
          filter="url(#eyeGlowF)"
        />
        <ellipse
          cx={128 + pupilOffset.x}
          cy={100 + eyeLookY + pupilOffset.y}
          rx="4.5" ry="3.5"
          fill="#0c0a09"
        />
        <ellipse
          cx={125 + pupilOffset.x * 0.3}
          cy={97.5 + eyeLookY + pupilOffset.y * 0.3}
          rx="2.2" ry="1.6"
          fill="rgba(255,247,237,0.85)"
        />
      </g>

      {/* ── Face specular highlight ── */}
      <circle cx="100" cy="105" r="70"
        fill="url(#faceSpec)" />

      {/* ── Small antenna ── */}
      <line x1="100" y1="35" x2="100" y2="22"
        stroke="#fb923c" strokeWidth="2"
        strokeLinecap="round" opacity="0.7" />
      <circle cx="100" cy="19" r="4.5"
        fill="#ea580c"
        filter="url(#eyeGlowF)" />
      <circle cx="100" cy="19" r="2"
        fill="rgba(255,247,237,0.9)" />

      {/* ── Ear nubs (simple rounded rects) ── */}
      <rect x="24" y="92" width="8" height="18" rx="4"
        fill="#1c1917"
        stroke="rgba(234,88,12,0.35)" strokeWidth="1" />
      <rect x="168" y="92" width="8" height="18" rx="4"
        fill="#1c1917"
        stroke="rgba(234,88,12,0.35)" strokeWidth="1" />

      {/* ── Status light (green pulse) ── */}
      <circle cx="138" cy="76" r="3" fill="#4ade80" opacity="0.85">
        <animate attributeName="opacity"
          values="0.85;0.25;0.85" dur="2.1s" repeatCount="indefinite" />
      </circle>

      {/* ── Cheek blush ── */}
      <ellipse cx="58" cy="118" rx="11" ry="6"
        fill="rgba(251,146,60,0.14)" />
      <ellipse cx="142" cy="118" rx="11" ry="6"
        fill="rgba(251,146,60,0.14)" />

      {/* ── Subtle smile line (resting expression) ── */}
      <path
        d="M 83 126 Q 100 133 117 126"
        fill="none"
        stroke="rgba(251,146,60,0.45)"
        strokeWidth="2.2"
        strokeLinecap="round"
      />
    </svg>
  )
}