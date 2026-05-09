// ============================================================
// Navbar — sticky, blurs on scroll, active link highlight
// ============================================================

import { useEffect, useState } from 'react'
import { NAV_LINKS } from '../data/content'

interface NavbarProps {
  theme: 'dark' | 'light'
  toggleTheme: () => void
}

export default function Navbar({
  theme,
  toggleTheme,
}: NavbarProps) {

  const isDark = theme === 'dark'
  const [scrolled, setScrolled] = useState(false)
  const [mobileOpen, setMobileOpen] = useState(false)

  useEffect(() => {
    const onScroll = () => setScrolled(window.scrollY > 40)
    window.addEventListener('scroll', onScroll, { passive: true })
    return () => window.removeEventListener('scroll', onScroll)
  }, [])

  return (
    <nav
      style={{
        position: 'fixed',
        top: 0,
        left: '50%',
        transform: 'translateX(-50%)',
        zIndex: 200,
        padding: '0 24px',
        height: 64,
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        // Frosted glass effect on scroll
        width: 'min(1280px, calc(100% - 32px))',
        margin: '16px auto 0',
        borderRadius: 9999,

        background: scrolled
          ? isDark
            ? 'rgba(28,25,23,0.72)'
            : 'rgba(255,247,237,0.72)'
          : isDark
            ? 'rgba(28,25,23,0.35)'
            : 'rgba(255,255,255,0.35)',

        backdropFilter: 'blur(24px)',

        border: isDark
          ? '1px solid rgba(255,255,255,0.06)'
          : '1px solid rgba(0,0,0,0.06)',

        boxShadow: scrolled
          ? isDark
            ? '0 10px 40px rgba(0,0,0,0.35)'
            : '0 10px 40px rgba(0,0,0,0.08)'
          : 'none',
        transition: 'all 0.3s ease',
      }}
    >
      {/* Logo */}
      <a
        href="/"
        style={{
          display: 'flex',
          alignItems: 'center',
          gap: 10,
          textDecoration: 'none',
        }}
      >
        {/* Logo mark */}
        <div
          style={{
            width: 32,
            height: 32,
            borderRadius: 9,
            background: 'linear-gradient(135deg, #ea580c 0%, #fb923c 100%)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            fontSize: 16,
            boxShadow: '0 0 16px rgba(234,88,12,0.5)',
          }}
        >
          ✦
        </div>
        <span
          style={{
            fontFamily: 'Nunito, sans-serif',
            fontWeight: 800,
            fontSize: 20,
            color: isDark ? '#fff7ed' : '#1c1917',
            letterSpacing: '-0.3px',
          }}
        >
          Botify
        </span>
      </a>

      {/* Desktop nav links */}
      <div
        style={{
          display: 'flex',
          alignItems: 'center',
          gap: 'clamp(20px, 2vw, 48px)',
        }}
        className="nav-links-desktop"
      >
        {NAV_LINKS.map((link) => (
          <a
            key={link.label}
            href={link.href}
            style={{
              color: isDark ? '#a8826a' : '#57534e',
              fontSize: 14,
              fontWeight: 500,
              textDecoration: 'none',
              transition: 'color 0.2s ease',
              fontFamily: 'Plus Jakarta Sans, sans-serif',
            }}
            onMouseEnter={(e) =>
              ((e.target as HTMLElement).style.color = '#fb923c')
            }
            onMouseLeave={(e) =>
              ((e.target as HTMLElement).style.color = '#a8826a')
            }
          >
            {link.label}
          </a>
        ))}
      </div>

      {/* Right side — CTA buttons */}
      <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
        <button
          onClick={() => toggleTheme()}
          style={{
            width: 42,
            height: 42,
            borderRadius: '50%',
            border: 'none',
            cursor: 'pointer',

            background: isDark
              ? 'rgba(255,255,255,0.06)'
              : 'rgba(0,0,0,0.05)',

            color: isDark ? '#fb923c' : '#ea580c',

            backdropFilter: 'blur(12px)',
          }}
        >
          {isDark ? '☀️' : '🌙'}
        </button>
        <a
          href="/login"
          style={{
            color: isDark ? '#a8826a' : '#57534e',
            fontSize: 14,
            fontWeight: 500,
            textDecoration: 'none',
            fontFamily: 'Plus Jakarta Sans, sans-serif',
          }}
          onMouseEnter={(e) =>
            ((e.target as HTMLElement).style.color = '#fff7ed')
          }
          onMouseLeave={(e) =>
            ((e.target as HTMLElement).style.color = '#a8826a')
          }
        >
          Sign in
        </a>

        <a
          href="/register"
          style={{
            background: '#ea580c',
            color: isDark ? '#fff7ed' : '#1c1917',
            padding: '8px 20px',
            borderRadius: 9999,
            fontSize: 14,
            fontWeight: 600,
            textDecoration: 'none',
            fontFamily: 'Plus Jakarta Sans, sans-serif',
            boxShadow: '0 0 20px rgba(234,88,12,0.35)',
            transition: 'all 0.2s ease',
            whiteSpace: 'nowrap',
          }}
          onMouseEnter={(e) => {
            const el = e.target as HTMLElement
            el.style.background = '#c2410c'
            el.style.boxShadow = '0 0 28px rgba(234,88,12,0.55)'
            el.style.transform = 'translateY(-1px)'
          }}
          onMouseLeave={(e) => {
            const el = e.target as HTMLElement
            el.style.background = '#ea580c'
            el.style.boxShadow = '0 0 20px rgba(234,88,12,0.35)'
            el.style.transform = 'translateY(0)'
          }}
        >
          Start free →
        </a>

        {/* Mobile hamburger */}
        <button
          onClick={() => setMobileOpen(!mobileOpen)}
          style={{
            display: 'none',
            background: 'none',
            border: 'none',
            color: isDark ? '#fff7ed' : '#1c1917',
            fontSize: 22,
            cursor: 'pointer',
            padding: 4,
          }}
          className="hamburger"
          aria-label="Toggle menu"
        >
          {mobileOpen ? '✕' : '☰'}
        </button>
      </div>

      {/* Mobile menu */}
      {mobileOpen && (
        <div
          style={{
            position: 'fixed',
            top: 64,
            left: 0,
            right: 0,
            background: isDark
            ? 'rgba(12,10,9,0.92)'
            : 'rgba(255,247,237,0.92)',
            backdropFilter: 'blur(20px)',
            borderBottom: '1px solid #292524',
            padding: '24px',
            display: 'flex',
            flexDirection: 'column',
            gap: 20,
            zIndex: 199,
          }}
        >
          {NAV_LINKS.map((link) => (
            <a
              key={link.label}
              href={link.href}
              onClick={() => setMobileOpen(false)}
              style={{
                color: isDark ? '#fff7ed' : '#1c1917',
                fontSize: 18,
                fontWeight: 600,
                textDecoration: 'none',
                fontFamily: 'Nunito, sans-serif',
              }}
            >
              {link.label}
            </a>
          ))}
          <a
            href="/register"
            style={{
              background: '#ea580c',
              color: isDark ? '#fff7ed' : '#1c1917',
              padding: '12px 24px',
              borderRadius: 9999,
              fontSize: 16,
              fontWeight: 700,
              textDecoration: 'none',
              textAlign: 'center',
              fontFamily: 'Nunito, sans-serif',
            }}
          >
            Start free →
          </a>
        </div>
      )}

      {/* Responsive CSS injected inline */}
      <style>{`
        @media (max-width: 768px) {
          .nav-links-desktop { display: none !important; }
          .hamburger { display: block !important; }
        }
      `}</style>
    </nav>
  )
}