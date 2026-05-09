// ============================================================
// Content data — edit this file to change all page text
// No need to touch components when updating copy
// ============================================================

import type { FeatureCard, PricingPlan, ScrollSectionConfig } from '../types'

// ── Hero content ──────────────────────────────────────────
export const HERO = {
  badge: '✦  Now with semantic search',
  headline: 'Your AI,',
  headlineHighlight: 'Your Brand.',
  subline:
    'Train a chatbot on your documents in minutes. Embed it anywhere. Give every customer instant, accurate answers — 24/7.',
  cta: {
    primary:   { label: 'Start building free →', href: '/register' },
    secondary: { label: '▶  See demo', href: '#how-it-works' },
  },
}

// ── Scroll sections (orb ↔ bot transitions) ───────────────
export const SCROLL_SECTIONS: ScrollSectionConfig[] = [
  {
    id: 'upload',
    splineState: 'orb',
    objectPosition: 'right',
    badge: '01 — Upload',
    headline: 'Feed it your',
    headlineHighlight: 'knowledge.',
    subline:
      'Drop in PDFs, Word docs, or plain text. Our AI reads, understands, and indexes everything — automatically. No prompts, no setup, no engineers needed.',
    cta: { label: 'Try uploading now', href: '/register', variant: 'primary' },
  },
  {
    id: 'train',
    splineState: 'bot',
    objectPosition: 'left',
    badge: '02 — Train',
    headline: 'It learns.',
    headlineHighlight: 'Instantly.',
    subline:
      'Semantic chunking, vector embeddings, hybrid search — all happening behind the scenes. Your chatbot goes from zero to expert in under a minute.',
  },
  {
    id: 'embed',
    splineState: 'orb',
    objectPosition: 'right',
    badge: '03 — Embed',
    headline: 'One line.',
    headlineHighlight: 'Live everywhere.',
    subline:
      'Copy a single script tag. Paste it into your website. Your branded AI chatbot appears — with your colors, your name, your personality.',
    cta: { label: 'Get your embed code', href: '/register', variant: 'primary' },
  },
]

// ── Features grid ─────────────────────────────────────────
export const FEATURES: FeatureCard[] = [
  {
    icon: '📄',
    title: 'Any document type',
    description:
      'PDF, DOCX, TXT and more. Upload once, works forever. Our parser handles tables, headers, everything.',
    badge: 'PDF · DOCX · TXT',
  },
  {
    icon: '🧠',
    title: 'Semantic search',
    description:
      'Ask "games" and it finds "Valorant tournament". Understands meaning, not just keywords.',
  },
  {
    icon: '⚡',
    title: 'Streaming responses',
    description:
      'Answers appear word by word — just like ChatGPT. No waiting for the full response.',
  },
  {
    icon: '🎨',
    title: 'Full white-label',
    description:
      'Your logo, your colors, your domain. Your customers never know we exist.',
  },
  {
    icon: '🔒',
    title: 'Your data stays yours',
    description:
      'Documents never leave your account. Strict tenant isolation — no cross-contamination.',
  },
  {
    icon: '📊',
    title: 'Usage analytics',
    description:
      'See what your customers ask most. Find gaps. Improve your chatbot over time.',
  },
]

// ── Pricing ───────────────────────────────────────────────
export const PRICING: PricingPlan[] = [
  {
    name: 'Free',
    price: '₹0',
    period: 'forever',
    description: 'Perfect for trying it out',
    features: [
      '1 chatbot',
      '10 documents',
      '500 conversations/month',
      'Basic analytics',
      'Embed on 1 website',
    ],
    cta: 'Start free',
    highlighted: false,
  },
  {
    name: 'Starter',
    price: '₹2,999',
    period: 'per month',
    description: 'For growing businesses',
    badge: 'Most popular',
    features: [
      '5 chatbots',
      '100 documents',
      '5,000 conversations/month',
      'Full analytics dashboard',
      'Embed on unlimited websites',
      'Priority support',
    ],
    cta: 'Start 14-day trial',
    highlighted: true,
  },
  {
    name: 'Growth',
    price: '₹7,999',
    period: 'per month',
    description: 'For scaling teams',
    features: [
      'Unlimited chatbots',
      'Unlimited documents',
      '25,000 conversations/month',
      'Team member access',
      'API access',
      'Custom domain',
      'Dedicated support',
    ],
    cta: 'Talk to sales',
    highlighted: false,
  },
]

// ── Testimonials ──────────────────────────────────────────
export const TESTIMONIALS = [
  {
    quote:
      'We went from 200 support tickets a day to under 40. Our team finally has time to breathe.',
    name: 'Priya Sharma',
    role: 'Head of Support, Zevo Commerce',
    avatar: 'PS',
  },
  {
    quote:
      "Setup took 8 minutes. I uploaded our FAQ PDF and it was live on our site before my tea went cold.",
    name: 'Arjun Mehta',
    role: 'Founder, LegalEdge',
    avatar: 'AM',
  },
  {
    quote:
      "Our students get answers at 2am now. Admissions queries dropped 60%. It's like hiring 5 more staff.",
    name: 'Dr. Kavya Nair',
    role: 'Director, Pinnacle Institute',
    avatar: 'KN',
  },
]

// ── Stats bar ─────────────────────────────────────────────
export const STATS = [
  { value: '500+',  label: 'Businesses trained' },
  { value: '2M+',   label: 'Questions answered' },
  { value: '<1s',   label: 'Avg response time' },
  { value: '99.9%', label: 'Uptime SLA' },
]

// ── Nav links ─────────────────────────────────────────────
export const NAV_LINKS = [
  { label: 'Features',  href: '#features' },
  { label: 'How it works', href: '#how-it-works' },
  { label: 'Pricing',   href: '#pricing' },
  { label: 'Docs',      href: '/docs' },
]