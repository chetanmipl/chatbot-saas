// ============================================================
// BotScene — React Three Fiber
// FIXED:
// - Dark volumetric sphere (not flat orange)
// - Orb shows FIRST, face after 180° spin
// - Face is on BACK of sphere, counter-rotated to be readable
// - Pointer look-at works on face group only
// - Correct drei imports, no unused three
// ============================================================

import { useRef } from 'react'
import { Canvas, useFrame } from '@react-three/fiber'
import { Float, Sparkles, Line } from '@react-three/drei'
import * as THREE from 'three'

interface BotSceneProps {
  scrollProgress: number
  size: number
  theme: 'dark' | 'light'
}

export default function BotScene({ scrollProgress, size, theme }: BotSceneProps) {
  return (
    <div style={{ width: size, height: size, position: 'relative' }}>
      {/* Ground shadow */}
      <div style={{
        position: 'absolute',
        bottom: -24,
        left: '50%',
        transform: 'translateX(-50%)',
        width: '55%',
        height: 16,
        background: 'radial-gradient(ellipse, rgba(234,88,12,0.28) 0%, transparent 70%)',
        filter: 'blur(10px)',
        pointerEvents: 'none',
        borderRadius: '50%',
      }} />

      <Canvas
        camera={{ position: [0, 0, 3.2], fov: 42 }}
        style={{ background: 'transparent' }}
        gl={{ alpha: true, antialias: true, powerPreference: 'high-performance' }}
      >
        {/* ── Lighting — key to non-flat look ── */}
        {/* Ambient: very dim so sphere stays dark */}
        <ambientLight intensity={0.08} />

        {/* Key light: orange from top-left — creates specular */}
        <pointLight
          position={[-2.5, 2.5, 2.5]}
          intensity={12}
          color="#fb923c"
          decay={2}
        />

        {/* Fill light: dim opposite side */}
        <pointLight
          position={[2, -1.5, 1]}
          intensity={3}
          color="#ea580c"
          decay={2}
        />

        {/* Back rim light: creates edge separation */}
        <pointLight
          position={[0, 0, -3]}
          intensity={6}
          color="#fdba74"
          decay={2}
        />

        {/* Inner core glow — very subtle */}
        <pointLight
          position={[0, 0, 0]}
          intensity={1.5}
          color="#ff6a00"
          decay={1}
        />

        <Float speed={1.6} rotationIntensity={0.1} floatIntensity={0.35}>
          <OrbGroup scrollProgress={scrollProgress} theme={theme} />
        </Float>
      </Canvas>
    </div>
  )
}

// ─────────────────────────────────────────────────────────────
// OrbGroup — the spinning group
// Back of sphere = face (counter-rotated to be readable)
// Front of sphere = energy orb effects
// ─────────────────────────────────────────────────────────────
function OrbGroup({
  scrollProgress,
  theme,
}: {
  scrollProgress: number
  theme: 'dark' | 'light'
}) {
  const groupRef = useRef<THREE.Group>(null)
  const isDark   = theme === 'dark'

  // Spin target: scroll 0.2→0.65 maps to 0→PI
  const spinT   = Math.max(0, Math.min(1, (scrollProgress - 0.20) / 0.45))
  const targetY = spinT * Math.PI

  // Track actual rotation to know which side is showing
  const actualRotY = useRef(0)

  useFrame(() => {
    if (!groupRef.current) return

    // Smooth lerp to target
    const newY = THREE.MathUtils.lerp(actualRotY.current, targetY, 0.055)
    actualRotY.current = newY
    groupRef.current.rotation.y = newY
  })

  // Face is visible when rotation > PI/2
  const faceVisible = targetY > Math.PI * 0.5

  return (
    <group ref={groupRef}>

      {/* ── Sphere body — dark with emissive rim ── */}
      <mesh>
        <sphereGeometry args={[1, 64, 64]} />
        <meshStandardMaterial
          color={isDark ? '#0d0602' : '#2a1005'}
          emissive={isDark ? '#c2410c' : '#ea580c'}
          emissiveIntensity={0.22}
          roughness={0.30}
          metalness={0.12}
        />
      </mesh>

      {/* ── Fresnel rim — BackSide sphere ── */}
      {/* Creates the bright orange edge that makes it look 3D */}
      <mesh scale={[1.008, 1.008, 1.008]}>
        <sphereGeometry args={[1, 32, 32]} />
        <meshBasicMaterial
          color={isDark ? '#fb923c' : '#f97316'}
          side={THREE.BackSide}
          transparent
          opacity={0.55}
          depthWrite={false}
        />
      </mesh>

      {/* ── Second softer rim for atmosphere ── */}
      <mesh scale={[1.018, 1.018, 1.018]}>
        <sphereGeometry args={[1, 24, 24]} />
        <meshBasicMaterial
          color={isDark ? '#ea580c' : '#fb923c'}
          side={THREE.BackSide}
          transparent
          opacity={0.18}
          depthWrite={false}
        />
      </mesh>

      {/* ── FRONT FACE (z positive): Energy orb effects ── */}
      {/* These are visible when sphere is at 0° rotation */}
      {/* They go to back after 180° spin */}
      <EnergyEffects isDark={isDark} visible={!faceVisible} />

      {/* ── BACK FACE (z negative): Bot face ── */}
      {/* Starts on back — hidden behind sphere initially */}
      {/* Counter-rotated [0, PI, 0] so it reads correctly when front */}
      <group
        position={[0, 0, -0.96]}
        rotation={[0, Math.PI, 0]}
      >
        <BotFace3D isDark={isDark} visible={faceVisible} />
      </group>

      {/* ── Sparkles — always present ── */}
      <Sparkles
        count={45}
        scale={2.6}
        size={1.0}
        speed={0.3}
        opacity={faceVisible ? 0.3 : 0.65}
        color={isDark ? '#fb923c' : '#f97316'}
      />

    </group>
  )
}

// ─────────────────────────────────────────────────────────────
// EnergyEffects — visible on FRONT (orb state)
// Glowing core + energy rings
// ─────────────────────────────────────────────────────────────
function EnergyEffects({
  isDark,
  visible,
}: {
  isDark: boolean
  visible: boolean
}) {
  const coreRef  = useRef<THREE.Mesh>(null)
  const ring1Ref = useRef<THREE.Mesh>(null)
  const ring2Ref = useRef<THREE.Mesh>(null)

  useFrame(({ clock }) => {
    const t = clock.elapsedTime

    // Pulsing core
    if (coreRef.current) {
      const s = 0.82 + 0.18 * Math.sin(t * 2.8)
      coreRef.current.scale.setScalar(s)
    }

    // Counter-rotate rings
    if (ring1Ref.current) ring1Ref.current.rotation.z += 0.008
    if (ring2Ref.current) ring2Ref.current.rotation.z -= 0.006
  })

  if (!visible) return null

  return (
    <group position={[0, 0, 0.02]}>

      {/* Glowing nucleus */}
      <mesh ref={coreRef}>
        <sphereGeometry args={[0.14, 16, 16]} />
        <meshBasicMaterial color="#fff7ed" />
      </mesh>

      {/* Core halo */}
      <mesh>
        <sphereGeometry args={[0.28, 16, 16]} />
        <meshBasicMaterial
          color="#fb923c"
          transparent
          opacity={0.25}
          depthWrite={false}
        />
      </mesh>

      {/* Rotating energy ring 1 */}
      <group ref={ring1Ref} rotation={[Math.PI * 0.25, 0, 0]}>
        <mesh>
          <torusGeometry args={[0.62, 0.012, 8, 60]} />
          <meshBasicMaterial
            color={isDark ? '#fb923c' : '#f97316'}
            transparent
            opacity={0.55}
          />
        </mesh>
      </group>

      {/* Rotating energy ring 2 */}
      <group ref={ring2Ref} rotation={[Math.PI * 0.6, Math.PI * 0.3, 0]}>
        <mesh>
          <torusGeometry args={[0.55, 0.009, 8, 60]} />
          <meshBasicMaterial
            color={isDark ? '#fdba74' : '#fb923c'}
            transparent
            opacity={0.40}
          />
        </mesh>
      </group>

      {/* Neural nodes */}
      {Array.from({ length: 6 }, (_, i) => {
        const angle = (i / 6) * Math.PI * 2
        const x = Math.cos(angle) * 0.58
        const y = Math.sin(angle) * 0.58
        return (
          <mesh key={i} position={[x, y, 0]}>
            <sphereGeometry args={[0.028, 8, 8]} />
            <meshBasicMaterial color="#fb923c" transparent opacity={0.7} />
          </mesh>
        )
      })}

    </group>
  )
}

// ─────────────────────────────────────────────────────────────
// BotFace3D — cute face on back of sphere
// Look-at via pointer (eyes only, not body)
// ─────────────────────────────────────────────────────────────
function BotFace3D({
  isDark,
  visible,
}: {
  isDark: boolean
  visible: boolean
}) {
  const eyeGroupRef  = useRef<THREE.Group>(null)   // only eyes move
  const blinkL       = useRef<THREE.Mesh>(null)
  const blinkR       = useRef<THREE.Mesh>(null)
  const blinkTimer   = useRef(0)
  const isBlinking   = useRef(false)
  const nextBlink    = useRef(3.0 + Math.random() * 2)

  // Eye material — white/cream pill
  const eyeMat = new THREE.MeshStandardMaterial({
    color:              isDark ? '#fff7ed' : '#ffffff',
    emissive:           isDark ? '#fdba74' : '#fde68a',
    emissiveIntensity:  0.5,
    roughness:          0.15,
    metalness:          0.05,
  })

  useFrame(({ pointer, clock }) => {
    const t = clock.elapsedTime

    // ── Look-at: only the eye group tracks pointer ──
    // Pointer is already -1→1 in R3F
    if (eyeGroupRef.current) {
      eyeGroupRef.current.rotation.x = THREE.MathUtils.lerp(
        eyeGroupRef.current.rotation.x,
        -pointer.y * 0.25,   // up/down — subtle
        0.07
      )
      eyeGroupRef.current.rotation.y = THREE.MathUtils.lerp(
        eyeGroupRef.current.rotation.y,
        pointer.x * 0.25,    // left/right — subtle
        0.07
      )
    }

    // ── Blink loop ──
    blinkTimer.current += 1 / 60
    if (!isBlinking.current && blinkTimer.current > nextBlink.current) {
      isBlinking.current = true
      // Close
      ;[blinkL.current, blinkR.current].forEach(m => {
        if (m) m.scale.y = 0.06
      })
      // Open after 130ms
      setTimeout(() => {
        ;[blinkL.current, blinkR.current].forEach(m => {
          if (m) m.scale.y = 1
        })
        isBlinking.current = false
        blinkTimer.current = 0
        nextBlink.current  = 2.5 + Math.random() * 2.5
      }, 130)
    }
  })

  if (!visible) return null

  return (
    <group>

      {/* ── Eye group — this tracks cursor ── */}
      <group ref={eyeGroupRef}>

        {/* Left pill eye */}
        <mesh
          ref={blinkL}
          position={[-0.26, 0.10, 0.01]}
        >
          {/* Capsule = pill shape */}
          <capsuleGeometry args={[0.07, 0.20, 8, 16]} />
          <primitive object={eyeMat} attach="material" />
        </mesh>

        {/* Right pill eye */}
        <mesh
          ref={blinkR}
          position={[0.26, 0.10, 0.01]}
        >
          <capsuleGeometry args={[0.07, 0.20, 8, 16]} />
          <primitive object={eyeMat} attach="material" />
        </mesh>

      </group>

      {/* ── Smile — fixed, doesn't track cursor ── */}
      <SmileLine isDark={isDark} />

      {/* ── Cheek blush ── */}
      <mesh position={[-0.42, -0.10, 0.005]}>
        <circleGeometry args={[0.09, 16]} />
        <meshBasicMaterial color="#fb923c" transparent opacity={0.18} />
      </mesh>
      <mesh position={[0.42, -0.10, 0.005]}>
        <circleGeometry args={[0.09, 16]} />
        <meshBasicMaterial color="#fb923c" transparent opacity={0.18} />
      </mesh>

      {/* ── Status dot (green pulse) ── */}
      <GreenDot />

    </group>
  )
}

// ── Smile curve using drei Line ───────────────────────────────
function SmileLine({ isDark }: { isDark: boolean }) {
  // Build smile points manually
  const smilePoints: [number, number, number][] = Array.from({ length: 20 }, (_, i) => {
    const t  = i / 19
    const x  = THREE.MathUtils.lerp(-0.22, 0.22, t)
    // Quadratic bezier: P0=(-0.22,-0.16) P1=(0,-0.06) P2=(0.22,-0.16)
    const bx = (1 - t) * (1 - t) * -0.22 + 2 * (1 - t) * t * 0.0 + t * t * 0.22
    const by = (1 - t) * (1 - t) * -0.16 + 2 * (1 - t) * t * -0.06 + t * t * -0.16
    return [bx, by, 0.01]
  })

  return (
    <Line
      points={smilePoints}
      color={isDark ? '#fb923c' : '#ea580c'}
      lineWidth={2.5}
      opacity={0.85}
      transparent
    />
  )
}

// ── Pulsing green status dot ──────────────────────────────────
function GreenDot() {
  const dotRef = useRef<THREE.Mesh>(null)

  useFrame(({ clock }) => {
    if (!dotRef.current) return
    const mat = dotRef.current.material as THREE.MeshBasicMaterial
    mat.opacity = 0.5 + 0.5 * Math.abs(Math.sin(clock.elapsedTime * 1.4))
  })

  return (
    <mesh ref={dotRef} position={[0.50, 0.50, 0.01]}>
      <circleGeometry args={[0.038, 16]} />
      <meshBasicMaterial color="#4ade80" transparent opacity={0.8} />
    </mesh>
  )
}