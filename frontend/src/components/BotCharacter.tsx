// ============================================================
// BotCharacter — v3
// ORB:  Proper 3D sphere with specular highlight, subsurface
//       scatter, animated fresnel energy bands
// FACE: Image 2 reference — pill eyes, glassy sphere, soft
// THEME: dark / light prop
// FLOAT: Drop shadow underneath for heavy-object feel
// ============================================================

import { useEffect, useRef, useState } from 'react'
import { useCursor } from '../hooks/useCursor'

type Theme = 'dark' | 'light'

interface BotCharacterProps {
  scrollProgress: number
  size?: number
  theme?: Theme
}

export default function BotCharacter({ scrollProgress, size = 480, theme = 'dark' }: BotCharacterProps) {
  const cursor = useCursor()

  const flipT   = Math.max(0, Math.min(1, (scrollProgress - 0.25) / 0.4))
  const rotateY = flipT * 180
  const tiltX =  cursor.y * 5   // much less body tilt
  const tiltY = -cursor.x * 5
  const isDark  = theme === 'dark'

  // Show face when more than halfway through flip
  const showFace = rotateY > 90

  return (
    <div style={{
      width: size, height: size,
      maxWidth: '88vw', maxHeight: '88vw',
      position: 'relative',
      perspective: '1200px',
      filter: 'drop-shadow(0 40px 60px rgba(234,88,12,0.35)) drop-shadow(0 60px 80px rgba(0,0,0,0.5))',
    }}>

      {/* Ambient glow */}
      <div style={{
        position: 'absolute', inset: '-15%', borderRadius: '50%',
        background: 'radial-gradient(ellipse, rgba(120,53,15,0.32) 0%, rgba(234,88,12,0.08) 55%, transparent 72%)',
        animation: 'pulse-glow 4s ease-in-out infinite',
        pointerEvents: 'none',
      }} />

      {/* Rotating wrapper */}
      <div style={{
        width: '100%', height: '100%',
        position: 'relative',
        transformStyle: 'preserve-3d',
        transform: `rotateX(${tiltX}deg) rotateY(${tiltY + rotateY}deg)`,
        // No transition here to keep scroll syncing perfectly
      }}>

        {/* BACK: Orb — visible when rotateY 0-90 */}
        <div style={{
          position: 'absolute', inset: 0,
          opacity: showFace ? 0 : 1,
          transition: 'opacity 0.15s ease-out',
          zIndex: showFace ? 0 : 1,
        }}>
          <EnergyOrb size={size} theme={theme} />
        </div>

        {/* FRONT: Face — rotated 180° to start */}
        <div style={{
          position: 'absolute', inset: 0,
          transform: 'rotateY(180deg)',
          opacity: showFace ? 1 : 0,
          transition: 'opacity 0.15s ease-out',
          zIndex: showFace ? 1 : 0,
        }}>
          <BotFace size={size} cursor={cursor} theme={theme} />
        </div>
      </div>
    </div>
  )
}

// ─────────────────────────────────────────────────────────────
// ENERGY ORB — proper 3D sphere
// Key: specular highlight + subsurface + animated fresnel bands
// ─────────────────────────────────────────────────────────────

function EnergyOrb({ size, theme }: { size: number; theme: Theme }) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const isDark    = theme === 'dark'

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const ctx = canvas.getContext('2d')
    if (!ctx) return

    const dpr = Math.min(window.devicePixelRatio || 1, 2)
    canvas.width  = size * dpr
    canvas.height = size * dpr
    canvas.style.width  = `${size}px`
    canvas.style.height = `${size}px`
    ctx.scale(dpr, dpr)

    const cx = size / 2
    const cy = size / 2
    const R  = size * 0.42

    let frame = 0
    let animId: number

    // Floating dust particles — scattered, no orbit path
    const dustParticles = Array.from({ length: 80 }, () => ({
      x:    cx + (Math.random() - 0.5) * R * 1.8,
      y:    cy + (Math.random() - 0.5) * R * 1.8,
      vx:   (Math.random() - 0.5) * 0.3,
      vy:   (Math.random() - 0.5) * 0.3,
      sz:   0.8 + Math.random() * 2.2,
      op:   0.2 + Math.random() * 0.6,
      hue:  18 + Math.random() * 22,
    }))

    // Fresnel energy bands — sweep across surface
    const bands = Array.from({ length: 6 }, (_, i) => ({
      angle: (i / 6) * Math.PI * 2,
      speed: (0.002 + Math.random() * 0.004) * (Math.random() > 0.5 ? 1 : -1),
      width: 0.06 + Math.random() * 0.12,
      bright: 0.25 + Math.random() * 0.4,
    }))

    function draw() {
      if (!ctx || !canvas) return
      ctx.clearRect(0, 0, size, size)
      frame++

      // Light source — top left
      const lx = cx - R * 0.30
      const ly = cy - R * 0.30

      // ── Sphere base — 3D shading with deeper shadows ────────
      const baseG = ctx.createRadialGradient(lx, ly, R * 0.05, cx, cy, R)
      if (isDark) {
        baseG.addColorStop(0,    'rgba(160,70,20,1)')   // Light center
        baseG.addColorStop(0.3,  'rgba(90,35,10,1)')
        baseG.addColorStop(0.6,  'rgba(40,15,5,1)')
        baseG.addColorStop(0.85, 'rgba(15,8,2,1)')
        baseG.addColorStop(1,    'rgba(8,5,2,1)')       // Dark edge
      } else {
        baseG.addColorStop(0,    '#ffd4a0')
        baseG.addColorStop(0.4,  '#f0a850')
        baseG.addColorStop(0.8,  '#d07020')
        baseG.addColorStop(1,    '#a04810')
      }
      ctx.beginPath()
      ctx.arc(cx, cy, R, 0, Math.PI * 2)
      ctx.fillStyle = baseG
      ctx.fill()

      // ── Subsurface scatter ──────────────────────────────
      const ssG = ctx.createRadialGradient(cx, cy + R * 0.1, 0, cx, cy, R * 0.8)
      ssG.addColorStop(0,   'rgba(220,70,5,0.20)')
      ssG.addColorStop(0.4, 'rgba(180,50,0,0.10)')
      ssG.addColorStop(1,   'rgba(0,0,0,0)')
      ctx.beginPath()
      ctx.arc(cx, cy, R * 0.8, 0, Math.PI * 2)
      ctx.fillStyle = ssG
      ctx.fill()

      // ── Clip for surface effects ────────────────────────
      ctx.save()
      ctx.beginPath()
      ctx.arc(cx, cy, R * 0.995, 0, Math.PI * 2)
      ctx.clip()

      // Moving energy bands on surface
      bands.forEach((band) => {
        band.angle += band.speed
        const pulse = 0.5 + 0.5 * Math.sin(frame * 0.035 + band.angle * 2)
        const a1 = band.angle - band.width
        const a2 = band.angle + band.width

        // Multi-layer band for soft glow
        ;[0.97, 0.88, 0.78].forEach((r, d) => {
          const alp = band.bright * pulse * (1 - d * 0.32)
          ctx.beginPath()
          ctx.arc(cx, cy, R * r, a1, a2)
          ctx.strokeStyle = `rgba(251,146,60,${alp})`
          ctx.lineWidth = 5 - d * 1.2
          ctx.lineCap = 'round'
          ctx.stroke()
        })
      })

      // Subtle horizontal energy lines
      for (let i = -10; i <= 10; i++) {
        const y  = cy + (i / 10) * R * 0.88
        const hw = Math.sqrt(Math.max(0, R * R - (y - cy) ** 2)) * 0.93
        if (hw < 2) continue
        const pulse = 0.2 + 0.8 * Math.abs(Math.sin(frame * 0.02 + i * 0.5))
        ctx.beginPath()
        ctx.moveTo(cx - hw, y)
        ctx.lineTo(cx + hw, y)
        ctx.strokeStyle = `rgba(251,146,60,${0.04 * pulse})`
        ctx.lineWidth = 0.6
        ctx.stroke()
      }

      // Dust particles floating inside sphere
      dustParticles.forEach((p) => {
        // Drift
        p.x += p.vx
        p.y += p.vy
        // Bounce off sphere boundary
        const dist = Math.sqrt((p.x - cx) ** 2 + (p.y - cy) ** 2)
        if (dist > R * 0.88) {
          p.vx *= -0.8
          p.vy *= -0.8
          p.x = cx + ((p.x - cx) / dist) * R * 0.86
          p.y = cy + ((p.y - cy) / dist) * R * 0.86
        }
        const pulse = 0.5 + 0.5 * Math.sin(frame * 0.05 + p.x * 0.05)
        ctx.beginPath()
        ctx.arc(p.x, p.y, p.sz, 0, Math.PI * 2)
        ctx.fillStyle = `hsla(${p.hue},90%,62%,${p.op * pulse})`
        ctx.fill()
      })

      ctx.restore() // end sphere clip

      // ── Fresnel rim — tight bright edge only ───────────
      const rimG = ctx.createRadialGradient(cx, cy, R * 0.82, cx, cy, R)
      rimG.addColorStop(0,    'rgba(234,88,12,0)')
      rimG.addColorStop(0.50, 'rgba(234,88,12,0.05)')
      rimG.addColorStop(0.75, 'rgba(251,146,60,0.35)')
      rimG.addColorStop(0.90, 'rgba(253,186,116,0.70)')
      rimG.addColorStop(1,    'rgba(255,220,160,0.55)')
      ctx.beginPath()
      ctx.arc(cx, cy, R, 0, Math.PI * 2)
      ctx.fillStyle = rimG
      ctx.fill()

      // ── Glowing core ────────────────────────────────────
      const pulse = 0.85 + 0.15 * Math.sin(frame * 0.04)
      const ns = R * 0.16 * pulse
      const nG = ctx.createRadialGradient(cx, cy, 0, cx, cy, ns)
      nG.addColorStop(0,   'rgba(255,252,245,1)')
      nG.addColorStop(0.3, 'rgba(253,186,116,0.95)')
      nG.addColorStop(0.65,'rgba(234,88,12,0.6)')
      nG.addColorStop(1,   'rgba(120,53,15,0)')
      ctx.beginPath()
      ctx.arc(cx, cy, ns, 0, Math.PI * 2)
      ctx.fillStyle = nG
      ctx.fill()

      // Inner corona around core
      const coronaG = ctx.createRadialGradient(cx, cy, ns, cx, cy, ns * 2.5)
      coronaG.addColorStop(0,   'rgba(251,146,60,0.25)')
      coronaG.addColorStop(0.5, 'rgba(234,88,12,0.10)')
      coronaG.addColorStop(1,   'rgba(0,0,0,0)')
      ctx.beginPath()
      ctx.arc(cx, cy, ns * 2.5, 0, Math.PI * 2)
      ctx.fillStyle = coronaG
      ctx.fill()

      // ── Large specular (3D illusion) ────────────────────
      const bigSpec = ctx.createRadialGradient(lx, ly, 0, lx, ly, R * 0.52)
      bigSpec.addColorStop(0,   'rgba(255,210,150,0.32)')
      bigSpec.addColorStop(0.4, 'rgba(255,170,80,0.12)')
      bigSpec.addColorStop(1,   'rgba(255,255,255,0)')
      ctx.beginPath()
      ctx.arc(cx, cy, R, 0, Math.PI * 2)
      ctx.fillStyle = bigSpec
      ctx.fill()

      // Small sharp specular dot (Glint)
      const dotX = lx + R * 0.08
      const dotY = ly + R * 0.08
      const dotG = ctx.createRadialGradient(dotX, dotY, 0, dotX, dotY, R * 0.12)
      dotG.addColorStop(0,   'rgba(255,255,255,1)')
      dotG.addColorStop(0.4, 'rgba(255,240,200,0.6)')
      dotG.addColorStop(1,   'rgba(255,255,255,0)')
      ctx.beginPath()
      ctx.arc(cx, cy, R, 0, Math.PI * 2)
      ctx.fillStyle = dotG
      ctx.fill()

      // ── Ground shadow ───────────────────────────────────
      const shadowG = ctx.createRadialGradient(
        cx, cy + R * 1.20, 0,
        cx, cy + R * 1.20, R * 0.60
      )
      shadowG.addColorStop(0,   'rgba(234,88,12,0.25)')
      shadowG.addColorStop(0.5, 'rgba(120,40,5,0.10)')
      shadowG.addColorStop(1,   'rgba(0,0,0,0)')
      ctx.beginPath()
      ctx.ellipse(cx, cy + R * 1.18, R * 0.58, R * 0.08, 0, 0, Math.PI * 2)
      ctx.fillStyle = shadowG
      ctx.fill()

      animId = requestAnimationFrame(draw)
    }

    draw()
    return () => cancelAnimationFrame(animId)
  }, [size, isDark])

  return (
    <canvas
      ref={canvasRef}
      style={{ display: 'block', borderRadius: '50%' }}
    />
  )
}

// ─────────────────────────────────────────────────────────────
// BOT FACE — Reference Image 2 style
// Glassy sphere + PILL eyes (rounded rectangle, tall, white)
// Soft iridescent sheen, cute minimal expression
// ─────────────────────────────────────────────────────────────

function BotFace({
  size,
  cursor,
  theme,
}: {
  size: number
  cursor: { x: number; y: number }
  theme: Theme
}) {
  const leftEyeRef  = useRef<SVGRectElement>(null)
  const rightEyeRef = useRef<SVGRectElement>(null)
  const blinkRef    = useRef<ReturnType<typeof setTimeout>>(undefined)
  const isDark      = theme === 'dark'

  // Blink loop
  useEffect(() => {
    const blink = () => {
      const delay = 2800 + Math.random() * 2200
      blinkRef.current = setTimeout(() => {
        ;[leftEyeRef.current, rightEyeRef.current].forEach((el) => {
          if (!el) return
          el.style.transform     = 'scaleY(0.06)'
          el.style.transformBox  = 'fill-box'
          el.style.transformOrigin = 'center'
          el.style.transition    = 'transform 0.08s ease'
        })
        setTimeout(() => {
          ;[leftEyeRef.current, rightEyeRef.current].forEach((el) => {
            if (!el) return
            el.style.transform  = 'scaleY(1)'
            el.style.transition = 'transform 0.11s ease'
          })
          blink()
        }, 130)
      }, delay)
    }
    blink()
    return () => clearTimeout(blinkRef.current)
  }, [])

  // Eye look-at offset
  const lx = cursor.x * 9      // eyes move more
  const ly = -cursor.y * 7

  // Pill eye dimensions
  const eyeW = 22     // pill width
  const eyeH = 38     // pill height (tall)
  const eyeR = 11     // pill corner radius

  // Eye positions
  const lEyeX = 74
  const rEyeX = 126
  const eyeY  = 96

  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 200 200"
      style={{ display: 'block', overflow: 'visible' }}
    >
      <defs>
        {/* Main sphere — iridescent/glassy */}
        <radialGradient id="glassSphere" cx="38%" cy="32%" r="68%">
          {isDark ? <>
            <stop offset="0%"   stopColor="#5a2e15" />
            <stop offset="30%"  stopColor="#3a1a0b" />
            <stop offset="65%"  stopColor="#251208" />
            <stop offset="100%" stopColor="#150a05" />
          </> : <>
            <stop offset="0%"   stopColor="#ffe8c8" />
            <stop offset="25%"  stopColor="#ffd4a0" />
            <stop offset="55%"  stopColor="#f0a850" />
            <stop offset="85%"  stopColor="#d07020" />
            <stop offset="100%" stopColor="#a04810" />
          </>}
        </radialGradient>

        {/* Iridescent color shift — top band */}
        <linearGradient id="iridescentBand" x1="0%" y1="0%" x2="100%" y2="100%">
          <stop offset="0%"   stopColor={isDark ? 'rgba(251,146,60,0.25)'  : 'rgba(255,220,120,0.35)'} />
          <stop offset="35%"  stopColor={isDark ? 'rgba(234,88,12,0.12)'   : 'rgba(255,180,60,0.20)'} />
          <stop offset="65%"  stopColor={isDark ? 'rgba(180,50,200,0.08)'  : 'rgba(255,160,80,0.15)'} />
          <stop offset="100%" stopColor="rgba(0,0,0,0)" />
        </linearGradient>

        {/* Fresnel rim */}
        <radialGradient id="glassRim" cx="50%" cy="50%" r="50%">
          <stop offset="0%"   stopColor="rgba(0,0,0,0)" />
          <stop offset="74%"  stopColor="rgba(0,0,0,0)" />
          <stop offset="85%"  stopColor={isDark ? 'rgba(234,88,12,0.15)'  : 'rgba(255,200,100,0.20)'} />
          <stop offset="93%"  stopColor={isDark ? 'rgba(251,146,60,0.55)' : 'rgba(255,220,140,0.65)'} />
          <stop offset="100%" stopColor={isDark ? 'rgba(255,210,140,0.80)': 'rgba(255,240,180,0.85)'} />
        </radialGradient>

        {/* Eye fill — white/cream pill (Image 2 style) */}
        <radialGradient id="eyePill" cx="35%" cy="25%" r="65%">
          <stop offset="0%"   stopColor={isDark ? '#fff7ed' : '#ffffff'} />
          <stop offset="50%"  stopColor={isDark ? '#fed7aa' : '#fef9f0'} />
          <stop offset="100%" stopColor={isDark ? '#fdba74' : '#fde8c0'} />
        </radialGradient>

        {/* Eye glow */}
        <filter id="pillGlow" x="-40%" y="-25%" width="180%" height="150%">
          <feGaussianBlur stdDeviation="2" result="blur"/>
          <feMerge>
            <feMergeNode in="blur"/>
            <feMergeNode in="SourceGraphic"/>
          </feMerge>
        </filter>

        {/* Large specular highlight */}
        <radialGradient id="bigSpec" cx="35%" cy="28%" r="42%">
          <stop offset="0%"   stopColor={isDark ? 'rgba(255,220,160,0.30)' : 'rgba(255,255,240,0.55)'} />
          <stop offset="50%"  stopColor={isDark ? 'rgba(255,180,80,0.10)'  : 'rgba(255,240,200,0.20)'} />
          <stop offset="100%" stopColor="rgba(255,255,255,0)" />
        </radialGradient>

        {/* Small sharp specular */}
        <radialGradient id="sharpSpec" cx="32%" cy="25%" r="14%">
          <stop offset="0%"   stopColor="rgba(255,255,255,0.90)" />
          <stop offset="60%"  stopColor="rgba(255,245,220,0.40)" />
          <stop offset="100%" stopColor="rgba(255,255,255,0)" />
        </radialGradient>

        {/* Clip to circle */}
        <clipPath id="sphereClip">
          <circle cx="100" cy="100" r="72" />
        </clipPath>

        {/* Antenna glow */}
        <filter id="antGlow" x="-100%" y="-100%" width="300%" height="300%">
          <feGaussianBlur stdDeviation="2.5"/>
        </filter>
      </defs>

      {/* ── Outer orbit ring ── */}
      <circle cx="100" cy="100" r="80"
        fill="none"
        stroke={isDark ? 'rgba(234,88,12,0.12)' : 'rgba(234,88,12,0.18)'}
        strokeWidth="1"
        strokeDasharray="3 9"
      />

      {/* ── Sphere body ── */}
      <circle cx="100" cy="100" r="72"
        fill="url(#glassSphere)" />

      {/* ── Iridescent color band (image 2 feel) ── */}
      <circle cx="100" cy="100" r="72"
        fill="url(#iridescentBand)"
        opacity="0.7" />

      {/* ── Subtle grid texture ── */}
      <g clipPath="url(#sphereClip)" opacity={isDark ? 0.05 : 0.04}>
        {Array.from({ length: 8 }, (_, i) => (
          <line key={`h${i}`}
            x1="28" y1={45 + i * 16}
            x2="172" y2={45 + i * 16}
            stroke={isDark ? '#fb923c' : '#ea580c'}
            strokeWidth="0.5"
          />
        ))}
      </g>

      {/* ── LEFT EYE — tall pill shape ── */}
      <g transform={`translate(${lx * 0.6},${ly * 0.6})`}>
        {/* Glow halo behind pill */}
        <rect
          x={lEyeX - eyeW / 2 - 5} y={eyeY - eyeH / 2 - 5}
          width={eyeW + 10} height={eyeH + 10}
          rx={eyeR + 5}
          fill={isDark ? 'rgba(251,146,60,0.18)' : 'rgba(255,200,80,0.22)'}
          filter="url(#pillGlow)"
        />
        {/* Pill body */}
        <rect
          ref={leftEyeRef}
          x={lEyeX - eyeW / 2 + lx * 0.4}
          y={eyeY - eyeH / 2 + ly * 0.4}
          width={eyeW}
          height={eyeH}
          rx={eyeR}
          fill="url(#eyePill)"
          filter="url(#pillGlow)"
          style={{
            transformBox: 'fill-box',
            transformOrigin: 'center',
          }}
        />
        {/* Pill inner shadow (depth) */}
        <rect
          x={lEyeX - eyeW / 2 + lx * 0.4 + 4}
          y={eyeY - eyeH / 2 + ly * 0.4 + 6}
          width={eyeW - 8}
          height={eyeH - 10}
          rx={eyeR - 3}
          fill="rgba(200,90,10,0.12)"
        />
        {/* Specular on pill */}
        <ellipse
          cx={lEyeX - 4 + lx * 0.4}
          cy={eyeY - 10 + ly * 0.4}
          rx={5} ry={3.5}
          fill="rgba(255,255,255,0.55)"
        />
      </g>

      {/* ── RIGHT EYE ── */}
      <g transform={`translate(${lx * 0.6},${ly * 0.6})`}>
        <rect
          x={rEyeX - eyeW / 2 - 5} y={eyeY - eyeH / 2 - 5}
          width={eyeW + 10} height={eyeH + 10}
          rx={eyeR + 5}
          fill={isDark ? 'rgba(251,146,60,0.18)' : 'rgba(255,200,80,0.22)'}
          filter="url(#pillGlow)"
        />
        <rect
          ref={rightEyeRef}
          x={rEyeX - eyeW / 2 + lx * 0.4}
          y={eyeY - eyeH / 2 + ly * 0.4}
          width={eyeW}
          height={eyeH}
          rx={eyeR}
          fill="url(#eyePill)"
          filter="url(#pillGlow)"
          style={{
            transformBox: 'fill-box',
            transformOrigin: 'center',
          }}
        />
        <rect
          x={rEyeX - eyeW / 2 + lx * 0.4 + 4}
          y={eyeY - eyeH / 2 + ly * 0.4 + 6}
          width={eyeW - 8}
          height={eyeH - 10}
          rx={eyeR - 3}
          fill="rgba(200,90,10,0.12)"
        />
        <ellipse
          cx={rEyeX - 4 + lx * 0.4}
          cy={eyeY - 10 + ly * 0.4}
          rx={5} ry={3.5}
          fill="rgba(255,255,255,0.55)"
        />
      </g>

      {/* ── Fresnel rim ── */}
      <circle cx="100" cy="100" r="72"
        fill="url(#glassRim)" />

      {/* ── Large specular (main 3D highlight) ── */}
      <circle cx="100" cy="100" r="72"
        fill="url(#bigSpec)" />

      {/* ── Small sharp specular dot ── */}
      <circle cx="100" cy="100" r="72"
        fill="url(#sharpSpec)" />

      {/* ── Ground shadow ellipse (floating weight) ── */}
      <ellipse cx="100" cy="185" rx="45" ry="6"
        fill={isDark ? 'rgba(234,88,12,0.18)' : 'rgba(180,80,10,0.15)'}
        style={{ filter: 'blur(4px)' }}
      />

      {/* ── Antenna ── */}
      <line x1="100" y1="28" x2="100" y2="15"
        stroke={isDark ? '#fb923c' : '#ea580c'}
        strokeWidth="2" strokeLinecap="round" opacity="0.75" />
      <circle cx="100" cy="12" r="5"
        fill={isDark ? '#ea580c' : '#f97316'}
        filter="url(#pillGlow)" />
      <circle cx="100" cy="12" r="2.2"
        fill="rgba(255,250,240,0.92)" />

      {/* ── Ear nubs ── */}
      {/* <rect x="22" y="88" width="9" height="20" rx="4.5"
        fill={isDark ? '#1c1917' : '#f5e8d5'}
        stroke={isDark ? 'rgba(234,88,12,0.4)' : 'rgba(234,88,12,0.5)'}
        strokeWidth="1.2" />
      <rect x="169" y="88" width="9" height="20" rx="4.5"
        fill={isDark ? '#1c1917' : '#f5e8d5'}
        stroke={isDark ? 'rgba(234,88,12,0.4)' : 'rgba(234,88,12,0.5)'}
        strokeWidth="1.2" /> */}

      {/* ── Status light ── */}
      <circle cx="136" cy="72" r="3.2"
        fill="#4ade80" opacity="0.85">
        <animate attributeName="opacity"
          values="0.85;0.25;0.85" dur="2.2s" repeatCount="indefinite"/>
      </circle>
    </svg>
  )
}