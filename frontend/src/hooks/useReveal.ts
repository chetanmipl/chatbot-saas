// ============================================================
// useReveal — IntersectionObserver for scroll-triggered reveals
// Attach the returned ref to any element you want to animate in
// ============================================================

import { useEffect, useRef } from 'react'

interface UseRevealOptions {
  threshold?: number   // 0–1, how much of element must be visible
  once?: boolean       // only trigger once (default true)
  delay?: number       // ms delay before adding .visible class
}

export function useReveal<T extends HTMLElement = HTMLDivElement>(
  options: UseRevealOptions = {}
) {
  const { threshold = 0.15, once = true, delay = 0 } = options
  const ref = useRef<T>(null)

  useEffect(() => {
    const el = ref.current
    if (!el) return

    const observer = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          if (entry.isIntersecting) {
            setTimeout(() => {
              entry.target.classList.add('visible')
            }, delay)
            if (once) observer.unobserve(entry.target)
          } else if (!once) {
            entry.target.classList.remove('visible')
          }
        })
      },
      { threshold }
    )

    observer.observe(el)
    return () => observer.disconnect()
  }, [threshold, once, delay])

  return ref
}