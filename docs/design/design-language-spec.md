# EMI Shield Designer - Design Language Specification

## Inspiration: Energy Guru by Lazarev Agency (Dribbble)

The Energy Guru design establishes a premium editorial aesthetic for industrial/scientific products. This document extracts its design principles and maps them to the EMI Shield Designer platform.

---

## 1. Design Philosophy

**Editorial Minimalism + 3D Immersive + Glassmorphism**

The Energy Guru design works because it makes an industrial product (chemical commodities) feel premium through three techniques:

1. **A single dramatic 3D visual** as the centerpiece (the blue crystal)
2. **Glassmorphism cards** floating around it as information overlays
3. **Typography as architecture** (giant "CHEMICALS" text functioning as a structural element, not content)

### How This Maps to EMI Shield Designer

| Energy Guru Element | EMI Shield Designer Equivalent |
|---------------------|-------------------------------|
| 3D crystal (Copper Sulphate) | 3D periodic table element or shield cross-section visualization |
| Floating glass cards with formulas | Floating SE result cards with physics values (dB, skin depth, impedance) |
| Category sidebar (Metals, Chemicals) | Material category sidebar (Metals, Alloys, Composites, MXenes) |
| Giant "CHEMICALS" display text | Giant "SHIELDING" or element symbol display text |
| "Submit request" CTA | "Start Simulation" CTA |

---

## 2. Color System

### Primary Palette

```css
:root {
  /* Backgrounds */
  --bg-primary: #0C0C14;        /* Deep space black (dark mode) */
  --bg-secondary: #13131F;      /* Elevated surface */
  --bg-tertiary: #1A1A2E;       /* Cards, panels */
  --bg-surface: #1E1E30;        /* Interactive surfaces */

  /* Glass effect (for dark mode glassmorphism) */
  --glass-bg: rgba(255, 255, 255, 0.06);
  --glass-border: rgba(255, 255, 255, 0.10);
  --glass-highlight: rgba(255, 255, 255, 0.15);
  --glass-blur: 20px;

  /* Text */
  --text-primary: #F0F0F8;      /* Primary content */
  --text-secondary: #8888A0;    /* Labels, descriptions */
  --text-muted: #5A5A72;        /* Disabled, hints */
  --text-display: rgba(255, 255, 255, 0.04); /* Giant watermark text */

  /* Accent - Cyan/Electric Blue (from EMI/EM wave theme) */
  --accent-primary: #00D4FF;    /* Primary actions, highlights */
  --accent-secondary: #6366F1;  /* Secondary accent (indigo) */
  --accent-tertiary: #10B981;   /* Success, positive values */
  --accent-warning: #F59E0B;    /* Warnings */
  --accent-danger: #EF4444;     /* Errors, critical */

  /* Gradients */
  --gradient-primary: linear-gradient(135deg, #00D4FF 0%, #6366F1 100%);
  --gradient-surface: linear-gradient(180deg, rgba(255,255,255,0.08) 0%, rgba(255,255,255,0.02) 100%);
  --gradient-glow: radial-gradient(circle at center, rgba(0, 212, 255, 0.15) 0%, transparent 70%);
}
```

### Element Category Colors (Periodic Table)

```css
:root {
  --element-alkali: #E74C3C;
  --element-alkaline: #E67E22;
  --element-transition: #3498DB;
  --element-post-transition: #2ECC71;
  --element-metalloid: #9B59B6;
  --element-nonmetal: #1ABC9C;
  --element-halogen: #F39C12;
  --element-noble: #95A5A6;
  --element-lanthanide: #E91E63;
  --element-actinide: #FF5722;
}
```

---

## 3. Typography System

### Font Stack

```css
:root {
  --font-display: 'General Sans', 'Inter', system-ui, sans-serif;
  --font-body: 'Inter', system-ui, sans-serif;
  --font-mono: 'JetBrains Mono', 'Fira Code', monospace;
}
```

### Type Scale

| Token | Size | Weight | Letter-spacing | Use |
|-------|------|--------|---------------|-----|
| `--text-display-xl` | 120-180px | 800 | 0.05em | Giant watermark text ("SHIELDING") |
| `--text-display` | 48-64px | 700 | -0.02em | Page titles |
| `--text-heading-1` | 32-40px | 600 | -0.01em | Section headings |
| `--text-heading-2` | 24-28px | 600 | 0 | Subsection headings |
| `--text-heading-3` | 18-20px | 500 | 0 | Card titles |
| `--text-body` | 15-16px | 400 | 0 | Body content |
| `--text-body-sm` | 13-14px | 400 | 0.01em | Secondary content |
| `--text-caption` | 11-12px | 500 | 0.06em | Labels, units, uppercase |
| `--text-mono` | 14px | 400 | 0 | Values, formulas, code |
| `--text-mono-lg` | 28-36px | 600 | -0.02em | Result numbers (SE in dB) |

---

## 4. Spacing System (8px Grid)

```css
:root {
  --space-1: 4px;
  --space-2: 8px;
  --space-3: 12px;
  --space-4: 16px;
  --space-5: 20px;
  --space-6: 24px;
  --space-8: 32px;
  --space-10: 40px;
  --space-12: 48px;
  --space-16: 64px;
  --space-20: 80px;
  --space-24: 96px;
}
```

---

## 5. Component Design Language

### Glass Cards (Glassmorphism)

```css
.glass-card {
  background: rgba(255, 255, 255, 0.04);
  backdrop-filter: blur(20px) saturate(1.3);
  border: 1px solid rgba(255, 255, 255, 0.08);
  border-radius: 16px;
  box-shadow:
    0 4px 24px rgba(0, 0, 0, 0.12),
    inset 0 1px 0 rgba(255, 255, 255, 0.06);
  transition: all 0.3s cubic-bezier(0.16, 1, 0.3, 1);
}

.glass-card:hover {
  border-color: rgba(255, 255, 255, 0.15);
  box-shadow:
    0 8px 32px rgba(0, 0, 0, 0.16),
    0 0 0 1px rgba(0, 212, 255, 0.1),
    inset 0 1px 0 rgba(255, 255, 255, 0.08);
  transform: translateY(-2px);
}
```

### Result Cards (SE Values)

```css
.result-card {
  /* Glass base */
  background: rgba(255, 255, 255, 0.04);
  backdrop-filter: blur(20px);
  border: 1px solid rgba(255, 255, 255, 0.08);
  border-radius: 16px;
  padding: 24px;

  /* Accent glow on left border */
  border-left: 3px solid var(--accent-primary);
}

.result-value {
  font-family: var(--font-mono);
  font-size: 36px;
  font-weight: 600;
  color: var(--text-primary);
  letter-spacing: -0.02em;
}

.result-unit {
  font-size: 16px;
  font-weight: 400;
  color: var(--text-secondary);
  margin-left: 4px;
}

.result-label {
  font-size: 12px;
  font-weight: 500;
  letter-spacing: 0.06em;
  text-transform: uppercase;
  color: var(--text-muted);
  margin-bottom: 8px;
}
```

### Buttons

```css
/* Primary - gradient fill */
.btn-primary {
  background: var(--gradient-primary);
  color: white;
  border: none;
  border-radius: 12px;
  padding: 12px 24px;
  font-weight: 500;
  font-size: 14px;
  transition: all 0.2s ease;
}

.btn-primary:hover {
  opacity: 0.9;
  transform: translateY(-1px);
  box-shadow: 0 4px 20px rgba(0, 212, 255, 0.3);
}

/* Secondary - outline glass */
.btn-secondary {
  background: transparent;
  color: var(--text-primary);
  border: 1px solid rgba(255, 255, 255, 0.15);
  border-radius: 12px;
  padding: 12px 24px;
}

/* Pill button (like Energy Guru "Submit request") */
.btn-pill {
  border-radius: 100px;
  padding: 10px 24px;
}
```

### Input Fields

```css
.input {
  background: rgba(255, 255, 255, 0.04);
  border: 1px solid rgba(255, 255, 255, 0.10);
  border-radius: 10px;
  padding: 12px 16px;
  color: var(--text-primary);
  font-size: 14px;
  transition: all 0.2s ease;
}

.input:focus {
  border-color: var(--accent-primary);
  box-shadow: 0 0 0 3px rgba(0, 212, 255, 0.1);
  outline: none;
}

.input-label {
  font-size: 12px;
  font-weight: 500;
  letter-spacing: 0.04em;
  text-transform: uppercase;
  color: var(--text-secondary);
  margin-bottom: 6px;
}
```

---

## 6. Animation Principles

### Core Easing

```css
:root {
  --ease-out-expo: cubic-bezier(0.16, 1, 0.3, 1);
  --ease-out-quart: cubic-bezier(0.25, 1, 0.5, 1);
  --ease-spring: cubic-bezier(0.34, 1.56, 0.64, 1);
}
```

### Standard Transitions

| Element | Duration | Easing | Properties |
|---------|----------|--------|------------|
| Hover states | 200ms | ease | opacity, transform, border-color |
| Card reveal | 400ms | ease-out-expo | opacity, transform |
| Panel open/close | 300ms | ease-out-quart | width, opacity |
| Page transition | 500ms | ease-out-expo | opacity, transform |
| Loading spinner | 1000ms | linear | rotation |

### Micro-interactions

```css
/* Floating cards animation (like Energy Guru chemical cards) */
@keyframes float {
  0%, 100% { transform: translateY(0px); }
  50% { transform: translateY(-6px); }
}

/* Subtle glow pulse on active calculations */
@keyframes glow-pulse {
  0%, 100% { box-shadow: 0 0 20px rgba(0, 212, 255, 0.1); }
  50% { box-shadow: 0 0 40px rgba(0, 212, 255, 0.2); }
}

/* Result number count-up animation */
@keyframes countUp {
  from { opacity: 0; transform: translateY(10px); }
  to { opacity: 1; transform: translateY(0); }
}
```

---

## 7. Layout Architecture

### Page Structure

```
+--------------------------------------------------+
| HEADER (sticky, glass, 64px)                     |
|  Logo  |  Nav Links  |  Status + CTA             |
+--------------------------------------------------+
|                                                   |
| HERO SECTION (100vh)                             |
|  [Giant watermark text "SHIELDING"]              |
|  [3D EM wave / shield visualization - center]    |
|  [Floating glass result cards - orbiting]        |
|  [Left: Category/mode sidebar]                   |
|  [Bottom: Scroll indicator]                      |
|                                                   |
+--------------------------------------------------+
|                                                   |
| SIMULATION WORKSPACE (below fold)                |
|  +----------+-------------------+--------+       |
|  | SIDEBAR  |    MAIN CANVAS    | CHAT   |       |
|  | Elements |  Parameters +     | AI     |       |
|  | Compos.  |  Charts +         | Assist |       |
|  | Presets  |  Results          |        |       |
|  +----------+-------------------+--------+       |
|                                                   |
+--------------------------------------------------+
```

### Three-Column Workspace

```css
.workspace {
  display: grid;
  grid-template-columns: 280px 1fr 320px;
  height: calc(100vh - 64px);
  gap: 1px; /* thin divider lines */
  background: rgba(255, 255, 255, 0.04); /* divider color */
}

.workspace-sidebar {
  background: var(--bg-secondary);
  overflow-y: auto;
  padding: var(--space-6);
}

.workspace-main {
  background: var(--bg-primary);
  overflow-y: auto;
  padding: var(--space-8);
}

.workspace-chat {
  background: var(--bg-secondary);
  display: flex;
  flex-direction: column;
}
```

---

## 8. Application to EMI Shield Designer - Page by Page

### Landing Page (/)

Apply the Energy Guru hero pattern:
- **Center:** 3D visualization of electromagnetic waves hitting a shield (could be Spline or Three.js)
- **Floating glass cards:** Key stats ("92 dB MXene record", "106 benchmark measurements", "5 physics models")
- **Giant watermark:** "SHIELDING" in display font, very low opacity
- **Left sidebar:** Quick links to different tools
- **CTA:** "Start Designing" gradient button

### Simulation Page (/simulation)

This is the main workbench:
- **Left panel:** Periodic table (compact grid) + composition builder
- **Center:** Parameters form + interactive Plotly charts with dark theme
- **Right panel:** AI chat with glassmorphism message bubbles
- Result cards use the `result-card` component with floating animation on calculation complete

### Materials Browser (/materials)

Apply the Energy Guru category navigation:
- **Left sidebar:** Material categories (Pure Metals, Alloys, Composites, MXenes, Magnetic)
- **Center:** Interactive periodic table or alloy grid
- **Floating cards:** Property tooltips with glassmorphism
- **Giant display text:** Active category name as watermark

---

## 9. Atomic Design Component Hierarchy

### Atoms
- Button (primary, secondary, pill, icon)
- Input (text, number, select)
- Badge (status, value, category)
- Icon (lucide-react)
- Tooltip
- Spinner/Loader

### Molecules
- Input Group (label + input + unit)
- Result Card (label + value + unit + confidence)
- Element Cell (periodic table cell: symbol + number + name)
- Nav Link (icon + text + active state)
- Chat Message (avatar + bubble + timestamp)
- Glass Card (backdrop-blur container)

### Organisms
- Header Navigation
- Periodic Table Grid
- Composition Panel (element list + percentages + total)
- Shield Parameters Form
- Results Dashboard (multiple result cards + chart)
- Chat Sidebar (message list + input + suggestions)
- Analysis Mode Selector

### Templates
- Landing Page Layout (hero + features + status)
- Simulation Workspace Layout (3-column grid)
- Materials Browser Layout (sidebar + content)

### Pages
- / (Landing)
- /simulation (Workbench)
- /materials (Browser)

---

## 10. Implementation Priority

1. **Update globals.css** with new design tokens (colors, typography, spacing)
2. **Create atom components** (Button, Input, Badge, GlassCard)
3. **Rebuild landing page** with Energy Guru hero pattern
4. **Update simulation workspace** with new card styles and animations
5. **Add micro-interactions** (floating cards, count-up numbers, glow effects)
6. **Add Framer Motion** for page transitions and element reveals
