import './styles/globals.css'
import Navbar from './components/Navbar'
import HeroSection from './components/HeroSection'

export default function App() {
  return (
    <div
      className="noise-overlay"
      style={{ minHeight: '100vh', background: '#0c0a09' }}
    >
      {/* Ambient background glows */}
      <div className="ambient-glow ambient-glow--top" />
      <div className="ambient-glow ambient-glow--bottom" />

      <Navbar />
      <HeroSection />

      {/* Placeholder so you can see the dark canvas */}
      <div
        style={{
          height: '100vh',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          flexDirection: 'column',
          gap: 16,
          paddingTop: 64,
        }}
      >
        <div
          style={{
            fontFamily: 'Nunito, sans-serif',
            fontSize: 48,
            fontWeight: 800,
            color: '#fff7ed',
          }}
        >
          Your AI,{' '}
          <span
            style={{
              background: 'linear-gradient(135deg, #fb923c, #fdba74)',
              WebkitBackgroundClip: 'text',
              WebkitTextFillColor: 'transparent',
            }}
          >
            Your Brand.
          </span>
        </div>
        <p style={{ color: '#a8826a', fontSize: 18 }}>
          Design system loaded ✓ — Hero coming in Part 3
        </p>
      </div>
    </div>
  )
}