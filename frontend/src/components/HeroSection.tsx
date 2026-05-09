import { useEffect, useRef, useState } from 'react'
import { gsap } from 'gsap'
import { ScrollTrigger } from 'gsap/ScrollTrigger'
import { HERO, STATS } from '../data/content'
import BotScene from './BotScene'

gsap.registerPlugin(ScrollTrigger)

interface HeroSectionProps {
  theme: 'dark' | 'light'
  setTheme: React.Dispatch<React.SetStateAction<'dark' | 'light'>>
}

export default function HeroSection({ theme, setTheme }: HeroSectionProps) {
  const sectionRef = useRef<HTMLDivElement>(null)
  const charRef    = useRef<HTMLDivElement>(null)
  const textRef    = useRef<HTMLDivElement>(null)
  const hintRef    = useRef<HTMLDivElement>(null)
  const [scrollProgress, setScrollProgress] = useState(0)
  const isDark = theme === 'dark'

  useEffect(() => {
    if (!sectionRef.current) return

    const ctx = gsap.context(() => {
      // Set initial states
      gsap.set(textRef.current, { x: -80, opacity: 0 })
      gsap.set(charRef.current, { x: 0, scale: 1 })

      const tl = gsap.timeline({
        scrollTrigger: {
          trigger: sectionRef.current,
          start: 'top top',
          end: '+=130%',
          scrub: 1.4,
          pin: true,
          anticipatePin: 1,
          onUpdate: (self) => setScrollProgress(self.progress),
        },
      })

      // Phase 1 (0→0.25): Orb starts spinning (handled inside BotScene)
      // Phase 2 (0.25→0.65): Spin reveals face (handled inside BotScene)
      // Phase 3 (0.5→0.85): Move orb right + shrink
      tl.to(charRef.current, {
        x: '20vw',
        scale: 0.60,
        duration: 0.5,
        ease: 'power3.inOut',
      }, 0.45)

      // Phase 4 (0.6→0.95): Text fades in center-left
      tl.to(textRef.current, {
        x: '-2vw',
        opacity: 1,
        duration: 0.45,
        ease: 'power3.out',
      }, 0.58)

      // Hide scroll hint immediately
      tl.to(hintRef.current, {
        opacity: 0,
        y: 16,
        duration: 0.2,
      }, 0)

    }, sectionRef)

    return () => ctx.revert()
  }, [])

  return (
    <div
      ref={sectionRef}
      style={{
        height: '230vh',
        position: 'relative',
        background: isDark
          ? 'radial-gradient(ellipse at 50% 40%, #1c0f05 0%, #0c0a09 65%)'
          : 'radial-gradient(ellipse at 50% 40%, #fff1df 0%, #fff7ed 65%)',
      }}
    >
      <div style={{
        position: 'sticky',
        top: 0,
        height: '100vh',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        overflow: 'hidden',
      }}>

        {/* Large BG word */}
        <div aria-hidden style={{
          position: 'absolute',
          top: '50%', left: '50%',
          transform: 'translate(-50%, -50%)',
          fontSize: 'clamp(180px, 38vw, 540px)',
          fontFamily: 'Nunito, sans-serif',
          fontWeight: 900,
          color: isDark ? 'rgba(251,146,60,0.03)' : 'rgba(234,88,12,0.04)',
          pointerEvents: 'none',
          userSelect: 'none',
          letterSpacing: '-20px',
          whiteSpace: 'nowrap',
          zIndex: 0,
        }}>
          AI
        </div>

        {/* ── Tight two-column layout ── */}
        <div style={{
          position: 'relative',
          width: '100%',
          maxWidth: 1280,
          height: '100%',
          display: 'flex',
          alignItems: 'center',
          // Tight gap — text and bot close together
          justifyContent: 'center',
          gap: 0,
          padding: '0 4vw',
          zIndex: 2,
        }}>

          {/* LEFT: Text — hidden until scroll */}
          <div
            ref={textRef}
            style={{
              width: 'min(480px, 38vw)',
              flexShrink: 0,
              zIndex: 4,
              pointerEvents: scrollProgress > 0.6 ? 'auto' : 'none',
            }}
          >
            <TextContent isDark={isDark} />
          </div>

          {/* RIGHT: 3D Bot/Orb — starts centered via GSAP x offset */}
          <div
            ref={charRef}
            style={{
              flexShrink: 0,
              zIndex: 3,
            }}
          >
            <BotScene
              scrollProgress={scrollProgress}
              size={520}
              theme={theme}
            />
          </div>
        </div>

        {/* Scroll hint */}
        <div ref={hintRef} style={{
          position: 'absolute',
          bottom: '7vh',
          left: '50%',
          transform: 'translateX(-50%)',
          textAlign: 'center',
          zIndex: 5,
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          gap: 12,
          animation: 'fade-in-up 1s ease 0.8s both',
        }}>
          <p style={{
            color: isDark ? '#78716c' : '#a8826a',
            fontSize: 12,
            fontWeight: 700,
            fontFamily: 'Plus Jakarta Sans, sans-serif',
            textTransform: 'uppercase',
            letterSpacing: '0.18em',
            margin: 0,
          }}>
            Explore
          </p>
          {/* Animated line */}
          <div style={{
            width: 1.5,
            height: 48,
            background: `linear-gradient(to bottom, #ea580c, transparent)`,
            borderRadius: 1,
            animation: 'float 1.6s ease-in-out infinite',
          }} />
        </div>

      </div>
    </div>
  )
}

function TextContent({ isDark }: { isDark: boolean }) {
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 24 }}>

      {/* Badge — appears with text */}
      <div style={{
        display: 'inline-flex',
        alignItems: 'center',
        gap: 8,
        background: 'rgba(120,53,15,0.38)',
        border: '1px solid rgba(234,88,12,0.35)',
        borderRadius: 9999,
        padding: '6px 16px',
        fontSize: 12,
        fontWeight: 600,
        color: '#fb923c',
        fontFamily: 'Plus Jakarta Sans, sans-serif',
        width: 'fit-content',
      }}>
        <span style={{ animation: 'pulse-glow 2s infinite' }}>✦</span>
        Now with semantic search
      </div>

      <h1 style={{
        fontFamily: 'Nunito, sans-serif',
        fontWeight: 900,
        fontSize: 'clamp(40px, 4.5vw, 68px)',
        lineHeight: 1.04,
        color: isDark ? '#fff7ed' : '#1c1917',
        letterSpacing: '-2px',
        margin: 0,
      }}>
        {HERO.headline}
        <br />
        <span style={{
          background: 'linear-gradient(135deg, #fb923c 0%, #fdba74 60%)',
          WebkitBackgroundClip: 'text',
          WebkitTextFillColor: 'transparent',
          backgroundClip: 'text',
        }}>
          {HERO.headlineHighlight}
        </span>
      </h1>

      <p style={{
        fontSize: 'clamp(14px, 1.2vw, 17px)',
        color: isDark ? '#a8826a' : '#44403c',
        lineHeight: 1.7,
        fontFamily: 'Plus Jakarta Sans, sans-serif',
        margin: 0,
        maxWidth: 420,
      }}>
        {HERO.subline}
      </p>

      <div style={{ display: 'flex', gap: 14, flexWrap: 'wrap' as const }}>
        <a href={HERO.cta.primary.href} style={{
          background: '#ea580c',
          color: '#fff7ed',
          padding: '14px 32px',
          borderRadius: 9999,
          fontSize: 15,
          fontWeight: 800,
          textDecoration: 'none',
          fontFamily: 'Plus Jakarta Sans, sans-serif',
          boxShadow: '0 8px 28px rgba(234,88,12,0.4)',
          transition: 'all 0.25s ease',
          display: 'inline-block',
        }}>
          {HERO.cta.primary.label}
        </a>
        <a href={HERO.cta.secondary.href} style={{
          background: 'transparent',
          color: '#fb923c',
          padding: '13px 32px',
          borderRadius: 9999,
          fontSize: 15,
          fontWeight: 700,
          textDecoration: 'none',
          fontFamily: 'Plus Jakarta Sans, sans-serif',
          border: '2px solid rgba(234,88,12,0.4)',
          transition: 'all 0.25s ease',
          display: 'inline-block',
        }}>
          {HERO.cta.secondary.label}
        </a>
      </div>

      <div style={{
        display: 'flex',
        gap: 32,
        paddingTop: 8,
        borderTop: `1px solid ${isDark ? 'rgba(255,247,237,0.08)' : 'rgba(28,25,23,0.08)'}`,
      }}>
        {STATS.slice(0, 3).map((s) => (
          <div key={s.label}>
            <div style={{
              fontFamily: 'Nunito, sans-serif',
              fontWeight: 900,
              fontSize: 26,
              color: '#ea580c',
              lineHeight: 1,
            }}>
              {s.value}
            </div>
            <div style={{
              fontSize: 11,
              fontWeight: 700,
              color: isDark ? '#78716c' : '#57534e',
              fontFamily: 'Plus Jakarta Sans, sans-serif',
              marginTop: 4,
              textTransform: 'uppercase' as const,
              letterSpacing: '0.08em',
            }}>
              {s.label}
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}