// ============================================================
// Navbar — sticky, blurs on scroll, active link highlight
// ============================================================

import { useEffect, useState } from 'react'
import { NAV_LINKS } from '../data/content'

export default function Navbar() {
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
        left: 0,
        right: 0,
        zIndex: 200,
        padding: '0 24px',
        height: 64,
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        // Frosted glass effect on scroll
        background: scrolled
          ? 'rgba(12, 10, 9, 0.85)'
          : 'transparent',
        backdropFilter: scrolled ? 'blur(20px)' : 'none',
        borderBottom: scrolled
          ? '1px solid rgba(41, 37, 36, 0.8)'
          : '1px solid transparent',
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
            color: '#fff7ed',
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
          gap: 32,
        }}
        className="nav-links-desktop"
      >
        {NAV_LINKS.map((link) => (
          <a
            key={link.label}
            href={link.href}
            style={{
              color: '#a8826a',
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
        <a
          href="/login"
          style={{
            color: '#a8826a',
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
            color: '#fff7ed',
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
            color: '#fff7ed',
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
            background: 'rgba(12,10,9,0.97)',
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
                color: '#fff7ed',
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
              color: '#fff7ed',
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