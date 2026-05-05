// ============================================================
// Shared Types
// ============================================================

/** Which 3D state the Spline scene should show */
export type SplineState = 'orb' | 'bot'

/** Direction the 3D object sits in a scroll section */
export type ObjectPosition = 'left' | 'right' | 'center'

/** A single scroll-driven section definition */
export interface ScrollSectionConfig {
  id: string
  splineState: SplineState
  objectPosition: ObjectPosition
  badge?: string
  headline: string
  headlineHighlight?: string   // word(s) to color in flame/glow
  subline: string
  cta?: {
    label: string
    href: string
    variant: 'primary' | 'secondary'
  }
}

/** Mouse / cursor position, normalised -1 to 1 */
export interface CursorPosition {
  x: number
  y: number
}

/** Feature card data */
export interface FeatureCard {
  icon: string
  title: string
  description: string
  badge?: string
}

/** Pricing plan data */
export interface PricingPlan {
  name: string
  price: string
  period: string
  description: string
  features: string[]
  cta: string
  highlighted: boolean
  badge?: string
}