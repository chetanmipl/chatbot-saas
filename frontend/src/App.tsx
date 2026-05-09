import './styles/globals.css'
import { useState } from 'react'
import Navbar from './components/Navbar'
import HeroSection from './components/HeroSection'

export default function App() {
  const [theme, setTheme] = useState<'dark' | 'light'>('dark')

  return (
    <div
      className="noise-overlay"
      style={{
        minHeight: '100vh',
        background: theme === 'dark' ? '#0c0a09' : '#fff7ed',
        transition: 'background 0.5s ease',
      }}
    >
      <div className="ambient-glow ambient-glow--top" />
      <div className="ambient-glow ambient-glow--bottom" />
      <Navbar
        theme={theme}
        toggleTheme={() => setTheme(t => t === 'dark' ? 'light' : 'dark')}
      />
      <HeroSection theme={theme} setTheme={setTheme} />
    </div>
  )
}