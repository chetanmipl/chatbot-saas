// ============================================================
// useCursor — tracks mouse position normalised to -1..1
// Used to make the Spline scene react to cursor movement
// ============================================================

import { useEffect, useRef, useState } from 'react'
import type { CursorPosition } from '../types'

export function useCursor(): CursorPosition {
  const [cursor, setCursor] = useState<CursorPosition>({ x: 0, y: 0 })
  const rafRef = useRef<number>(0)
  const targetRef = useRef<CursorPosition>({ x: 0, y: 0 })
  const currentRef = useRef<CursorPosition>({ x: 0, y: 0 })

  useEffect(() => {
    const handleMouseMove = (e: MouseEvent) => {
      // Normalise to -1..1
      targetRef.current = {
        x: (e.clientX / window.innerWidth) * 2 - 1,
        y: -(e.clientY / window.innerHeight) * 2 + 1,
      }
    }

    // Lerp loop — smooth the cursor movement
    const lerp = (a: number, b: number, t: number) => a + (b - a) * t

    const animate = () => {
      currentRef.current = {
        x: lerp(currentRef.current.x, targetRef.current.x, 0.06),
        y: lerp(currentRef.current.y, targetRef.current.y, 0.06),
      }
      setCursor({ ...currentRef.current })
      rafRef.current = requestAnimationFrame(animate)
    }

    window.addEventListener('mousemove', handleMouseMove)
    rafRef.current = requestAnimationFrame(animate)

    return () => {
      window.removeEventListener('mousemove', handleMouseMove)
      cancelAnimationFrame(rafRef.current)
    }
  }, [])

  return cursor
}