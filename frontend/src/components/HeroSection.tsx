import { useEffect, useRef, useState } from 'react'
import { gsap } from 'gsap'
import { ScrollTrigger } from 'gsap/ScrollTrigger'
import { HERO, STATS } from '../data/content'
import BotCharacter from './BotCharacter'

gsap.registerPlugin(ScrollTrigger)

export default function HeroSection() {
  const sectionRef = useRef<HTMLDivElement>(null)
  const charRef    = useRef<HTMLDivElement>(null)
  const textRef    = useRef<HTMLDivElement>(null)
  const tagRef     = useRef<HTMLDivElement>(null)
  const pillRef    = useRef<HTMLDivElement>(null)

  // Scroll progress 0→1 drives the flip inside BotCharacter
  const [scrollProgress, setScrollProgress] = useState(0)

  useEffect(() => {
    if (!sectionRef.current) return

    const ctx = gsap.context(() => {
      const tl = gsap.timeline({
        scrollTrigger: {
          trigger: sectionRef.current,
          start: 'top top',
          end: '+=100%',
          scrub: 1.2,
          pin: true,
          anticipatePin: 1,
          // This updates every frame as user scrolls
          onUpdate: (self) => {
            setScrollProgress(self.progress)
          },
        },
      })

      // Character: center → right + shrink
      // Only moves AFTER flip is done (progress > 0.65)
      tl.to(charRef.current, {
        x: '21vw',
        scale: 0.58,
        duration: 0.35,
        ease: 'power2.inOut',
      }, 0.65)   // starts at 65% through scroll

      // Hide tagline early
      tl.to(tagRef.current, {
        opacity: 0, y: -20,
        duration: 0.25,
        ease: 'power2.in',
      }, 0.05)

      // Hide pill early
      tl.to(pillRef.current, {
        opacity: 0, duration: 0.2,
      }, 0.05)

      // Text slides in after flip + move
      tl.fromTo(textRef.current,
        { opacity: 0, x: -60 },
        { opacity: 1, x: 0, duration: 0.35, ease: 'power3.out' },
        0.72
      )

    }, sectionRef)

    return () => ctx.revert()
  }, [])

  return (
    <div ref={sectionRef} style={{ height: '200vh', position: 'relative' }}>

      <div style={{
        position: 'sticky',
        top: 0,
        height: '100vh',
        overflow: 'hidden',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
      }}>

        {/* BG letters */}
        <div aria-hidden style={{
          position: 'absolute',
          top: '50%', left: '50%',
          transform: 'translate(-50%, -50%)',
          fontSize: 'clamp(200px, 32vw, 420px)',
          fontFamily: 'Nunito, sans-serif',
          fontWeight: 900,
          color: 'rgba(41,37,36,0.09)',
          pointerEvents: 'none',
          userSelect: 'none',
          letterSpacing: '-14px',
          whiteSpace: 'nowrap',
          zIndex: 0,
        }}>
          AI
        </div>

        {/* Bot character — GSAP moves this div, BotCharacter handles flip */}
        <div
          ref={charRef}
          style={{
            position: 'absolute',
            zIndex: 2,
            animation: 'float-slow 5s ease-in-out infinite',
          }}
        >
          <BotCharacter
            scrollProgress={scrollProgress}
            size={480}
          />
        </div>

        {/* Tagline — fades out on first scroll */}
        <div ref={tagRef} style={{
          position: 'absolute',
          bottom: '11vh',
          left: '50%',
          transform: 'translateX(-50%)',
          textAlign: 'center',
          zIndex: 5,
          animation: 'fade-in-up 1s ease 0.5s both',
          whiteSpace: 'nowrap',
        }}>
          <div style={{
            display: 'inline-flex',
            alignItems: 'center',
            gap: 8,
            background: 'rgba(120,53,15,0.38)',
            border: '1px solid rgba(234,88,12,0.35)',
            borderRadius: 9999,
            padding: '7px 20px',
            fontSize: 13,
            fontWeight: 600,
            color: '#fb923c',
            fontFamily: 'Plus Jakarta Sans, sans-serif',
            marginBottom: 14,
            boxShadow: '0 0 18px rgba(234,88,12,0.12)',
          }}>
            <span style={{ animation: 'pulse-glow 2s infinite' }}>✦</span>
            {HERO.badge}
          </div>
          <p style={{
            color: '#78716c', fontSize: 14,
            fontFamily: 'Plus Jakarta Sans, sans-serif',
          }}>
            Scroll to meet your AI
          </p>
        </div>

        {/* Text content — fades in after flip */}
        <div ref={textRef} style={{
          position: 'absolute',
          left: '5vw',
          top: '50%',
          transform: 'translateY(-50%)',
          width: 'min(38vw, 450px)',
          opacity: 0,
          zIndex: 4,
        }}>
          <TextContent />
        </div>

        {/* Scroll pill */}
        <div ref={pillRef} style={{
          position: 'absolute',
          bottom: 32,
          left: '50%',
          transform: 'translateX(-50%)',
          zIndex: 6,
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          gap: 8,
          animation: 'fade-in-up 1s ease 1.5s both',
        }}>
          <div style={{
            width: 24, height: 40,
            border: '1.5px solid rgba(120,113,108,0.45)',
            borderRadius: 12,
            display: 'flex',
            justifyContent: 'center',
            paddingTop: 7,
          }}>
            <div style={{
              width: 4, height: 8,
              background: '#fb923c',
              borderRadius: 2,
              animation: 'float 1.4s ease-in-out infinite',
              boxShadow: '0 0 8px rgba(251,146,60,0.8)',
            }} />
          </div>
          <span style={{
            fontSize: 10,
            letterSpacing: '0.18em',
            textTransform: 'uppercase' as const,
            color: '#78716c',
            fontFamily: 'Plus Jakarta Sans, sans-serif',
          }}>
            Scroll
          </span>
        </div>

      </div>
    </div>
  )
}

function TextContent() {
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 22 }}>
      <h1 style={{
        fontFamily: 'Nunito, sans-serif',
        fontWeight: 900,
        fontSize: 'clamp(32px, 3.8vw, 60px)',
        lineHeight: 1.06,
        color: '#fff7ed',
        letterSpacing: '-1.5px',
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
        fontSize: 'clamp(13px, 1.15vw, 16px)',
        color: '#a8826a',
        lineHeight: 1.75,
        fontFamily: 'Plus Jakarta Sans, sans-serif',
        margin: 0,
        maxWidth: 390,
      }}>
        {HERO.subline}
      </p>

      <div style={{ display: 'flex', gap: 12, flexWrap: 'wrap' as const }}>
        {[
          { href: '/register', label: 'Start building free →', primary: true },
          { href: '#how-it-works', label: '▶  See demo', primary: false },
        ].map(({ href, label, primary }) => (
          <a key={label} href={href} style={{
            background: primary ? '#ea580c' : 'transparent',
            color:      primary ? '#fff7ed' : '#fb923c',
            padding: '11px 24px',
            borderRadius: 9999,
            fontSize: 14,
            fontWeight: 700,
            textDecoration: 'none',
            fontFamily: 'Nunito, sans-serif',
            border: primary ? 'none' : '1.5px solid rgba(234,88,12,0.45)',
            boxShadow: primary ? '0 0 24px rgba(234,88,12,0.4)' : 'none',
            display: 'inline-block',
            transition: 'all 0.2s ease',
          }}>
            {label}
          </a>
        ))}
      </div>

      <div style={{
        display: 'flex', gap: 18, paddingTop: 6,
        borderTop: '1px solid rgba(41,37,36,0.7)',
      }}>
        {STATS.slice(0, 3).map((s) => (
          <div key={s.label}>
            <div style={{
              fontFamily: 'Nunito, sans-serif',
              fontWeight: 800,
              fontSize: 20,
              color: '#fb923c',
              lineHeight: 1,
            }}>
              {s.value}
            </div>
            <div style={{
              fontSize: 11,
              color: '#78716c',
              fontFamily: 'Plus Jakarta Sans, sans-serif',
              marginTop: 3,
            }}>
              {s.label}
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}