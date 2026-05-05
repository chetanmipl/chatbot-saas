// ============================================================
// SplineScene — embeds your Spline scene and handles:
// 1. Cursor tilt effect (CSS transform on container)
// 2. Floating animation wrapper
// 3. Scroll-driven position (left / right / center)
// 4. Orb vs Bot opacity overlay (since both are in same scene)
// ============================================================

import { useRef, useEffect, Suspense } from 'react'
import Spline from '@splinetool/react-spline'
import type { ObjectPosition, SplineState } from '../types'
import { useCursor } from '../hooks/useCursor'

interface SplineSceneProps {
  state: SplineState          // 'orb' | 'bot'
  position: ObjectPosition    // 'left' | 'right' | 'center'
  isHero?: boolean            // hero gets special sizing
}

export default function SplineScene({
  state,
  position,
  isHero = false,
}: SplineSceneProps) {
  const cursor = useCursor()
  const wrapperRef = useRef<HTMLDivElement>(null)
  const splineRef = useRef<any>(null)

  // Apply subtle cursor-driven tilt to the wrapper
  useEffect(() => {
    if (!wrapperRef.current) return
    const tiltX = cursor.y * 8   // degrees
    const tiltY = cursor.x * -8
    wrapperRef.current.style.transform =
      `rotateX(${tiltX}deg) rotateY(${tiltY}deg)`
  }, [cursor])

  // When state changes, trigger Spline named events
  // Your scene has hover timelines — we fire them on state change
  useEffect(() => {
    if (!splineRef.current) return
    try {
      if (state === 'bot') {
        splineRef.current.emitEvent('mouseDown', 'Ellipse')
      } else {
        splineRef.current.emitEvent('mouseDown', 'Sphere')
      }
    } catch {
      // Spline event not found — fine, scene still shows
    }
  }, [state])

  const positionStyles: Record<ObjectPosition, React.CSSProperties> = {
    center: { margin: '0 auto' },
    left:   { marginRight: 'auto' },
    right:  { marginLeft: 'auto' },
  }

  return (
    <div
      style={{
        position: 'relative',
        width: isHero ? '560px' : '480px',
        height: isHero ? '560px' : '480px',
        maxWidth: '90vw',
        maxHeight: '90vw',
        ...positionStyles[position],
        // Perspective needed for tilt to look 3D
        perspective: '800px',
      }}
    >
      {/* Ambient glow behind the scene */}
      <div
        style={{
          position: 'absolute',
          inset: '-20%',
          borderRadius: '50%',
          background:
            'radial-gradient(ellipse, rgba(234,88,12,0.25) 0%, transparent 70%)',
          animation: 'pulse-glow 3s ease-in-out infinite',
          pointerEvents: 'none',
          zIndex: 0,
        }}
      />

      {/* Tilt wrapper — cursor moves this */}
      <div
        ref={wrapperRef}
        style={{
          width: '100%',
          height: '100%',
          position: 'relative',
          zIndex: 1,
          transition: 'transform 0.12s ease-out',
          transformStyle: 'preserve-3d',
          // Floating animation
          animation: 'float-slow 5s ease-in-out infinite',
        }}
      >
        {/* Spline scene embed */}
        <Suspense fallback={<SplineFallback />}>
          <Spline
            scene="https://prod.spline.design/BOFgLvJ8lcC67okz/scene.splinecode"
            onLoad={(spline) => {
              splineRef.current = spline
            }}
            style={{
              width: '100%',
              height: '100%',
              borderRadius: '50%',
            }}
          />
        </Suspense>

        {/* Bot expression overlay — shown when state === 'bot'
            This adds a mouth since your scene only has eyes */}
        <BotExpressionOverlay visible={state === 'bot'} cursor={cursor} />
      </div>

      {/* Particle ring decoration */}
      <ParticleRing active={state === 'orb'} />
    </div>
  )
}

// ── Bot expression overlay ────────────────────────────────
// Adds SVG mouth expressions on top of your Spline eyes
// because your scene doesn't have a mouth yet

function BotExpressionOverlay({
  visible,
  cursor,
}: {
  visible: boolean
  cursor: { x: number; y: number }
}) {
  // Mouth curves based on cursor Y position
  // cursor.y > 0 = upper half = happy, < 0 = lower half = thinking
  const smileAmount = Math.max(0, cursor.y) * 20
  const d = `M 30 50 Q 50 ${50 + smileAmount} 70 50`

  return (
    <div
      style={{
        position: 'absolute',
        inset: 0,
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        pointerEvents: 'none',
        opacity: visible ? 1 : 0,
        transition: 'opacity 0.6s ease',
        zIndex: 2,
        // Push mouth below center where bot face is
        paddingTop: '55%',
      }}
    >
      <svg
        width="100"
        height="60"
        viewBox="0 0 100 80"
        style={{ filter: 'drop-shadow(0 0 6px rgba(251,146,60,0.8))' }}
      >
        {/* Mouth */}
        <path
          d={d}
          fill="none"
          stroke="#fb923c"
          strokeWidth="3.5"
          strokeLinecap="round"
          style={{ transition: 'd 0.3s ease' }}
        />
        {/* Cute cheek blush dots */}
        <circle cx="18" cy="52" r="5" fill="rgba(251,146,60,0.3)" />
        <circle cx="82" cy="52" r="5" fill="rgba(251,146,60,0.3)" />
      </svg>
    </div>
  )
}

// ── Particle ring ─────────────────────────────────────────
// Orbiting dots shown in orb state — CSS only, no library needed

function ParticleRing({ active }: { active: boolean }) {
  const particles = Array.from({ length: 8 }, (_, i) => i)

  return (
    <div
      style={{
        position: 'absolute',
        inset: '-15%',
        borderRadius: '50%',
        pointerEvents: 'none',
        opacity: active ? 1 : 0,
        transition: 'opacity 0.8s ease',
        animation: 'spin-slow 12s linear infinite',
      }}
    >
      {particles.map((i) => {
        const angle = (i / particles.length) * 360
        const size = i % 2 === 0 ? 6 : 4
        return (
          <div
            key={i}
            style={{
              position: 'absolute',
              width: `${size}px`,
              height: `${size}px`,
              borderRadius: '50%',
              background: i % 3 === 0 ? '#ea580c' : '#fb923c',
              top: '50%',
              left: '50%',
              transform: `rotate(${angle}deg) translateX(calc(50% + 80px)) translateY(-50%)`,
              boxShadow: `0 0 ${size * 2}px rgba(234,88,12,0.8)`,
            }}
          />
        )
      })}
    </div>
  )
}

// ── Loading fallback ──────────────────────────────────────
function SplineFallback() {
  return (
    <div
      style={{
        width: '100%',
        height: '100%',
        borderRadius: '50%',
        background:
          'radial-gradient(ellipse at center, #78350f 0%, #1c1917 60%, transparent 100%)',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        animation: 'pulse-glow 2s ease-in-out infinite',
      }}
    >
      <div
        style={{
          width: 48,
          height: 48,
          borderRadius: '50%',
          border: '3px solid #ea580c',
          borderTopColor: 'transparent',
          animation: 'spin-slow 1s linear infinite',
        }}
      />
    </div>
  )
}