# Atomic Design Reference for EMI Shield Designer

A condensed reference based on Brad Frost's *Atomic Design* (all 5 chapters),
tailored for guiding the component architecture of the EMI shielding simulation
platform.

---

## Chapter 1 -- Designing Systems (Not Pages)

### Core argument

The web is not print. Designing for fixed-dimension "pages" collapses under the
reality of infinite screen sizes, input modes, and contexts. The shift required
is from **page-based thinking** to **system-based thinking**: build a vocabulary
of reusable interface elements and combine them into higher-order structures.

### Key principles

1. **The medium shapes the method.** Print has fixed canvases; the web does not.
   A design system must embrace fluidity and composability instead of static
   mockups.

2. **Systematic UI design.** Rather than producing one-off screens, invest in a
   shared vocabulary of components that can be assembled into any screen. This
   reduces redundancy, accelerates development, and enforces consistency.

3. **Style guides as a deliverable.** The ultimate artifact of design work is not
   a set of PSD/Figma files but a living style guide -- an interactive reference
   of every component, its states, and its usage guidelines.

4. **Shared vocabulary between disciplines.** Designers, engineers, QA, and
   product should all speak the same component language. Naming matters.

### Applied to EMI Shield Designer

The platform already shows signs of page-level thinking: the homepage, the
simulation page, and the materials page each build bespoke structures. Inline
styles (long `className` strings with hard-coded color values like `#0a0a0f`,
`#2a2a3e`, `#9898b0`) are repeated across pages. Atomic Design prescribes
extracting these into a shared design-token layer and reusable components.

---

## Chapter 2 -- Atomic Design Methodology

### The five-level hierarchy

| Level        | Analogy           | Definition                                                                                     |
| ------------ | ----------------- | ---------------------------------------------------------------------------------------------- |
| **Atoms**    | HTML elements      | The smallest, indivisible UI elements: buttons, labels, inputs, icons, color tokens, type scale |
| **Molecules**| Simple groups      | Small clusters of atoms that function as a unit: a labeled input field, a search bar             |
| **Organisms**| Complex sections   | Relatively complex UI sections composed of molecules and/or atoms: a header, a sidebar, a card   |
| **Templates**| Page-level layouts | Page-level structures showing content layout without real data: wireframe-like skeletons          |
| **Pages**    | Specific instances | Templates populated with real/representative content; used for review and testing                 |

### Key principles

1. **Atoms are context-free.** A `<Button>` or `<TextInput>` should work
   anywhere. It receives props for variant, size, and state -- nothing more.

2. **Molecules combine atoms into a single responsibility.** A "frequency input"
   molecule combines a label atom, a numeric input atom, and a unit-suffix atom.
   It handles one user action.

3. **Organisms are where domain logic appears.** A "Shield Parameters" organism
   wires together multiple molecules (frequency input, thickness input, mode
   selector) and holds the logic for how they relate.

4. **Templates are content-agnostic layouts.** They define spatial relationships:
   "left sidebar (320 px) | center content (fluid) | right sidebar (320 px)".
   They say nothing about which organism goes where.

5. **Pages are the proof.** They inject real EMI calculation results, real alloy
   compositions, and real sweep charts into templates. They expose edge cases
   (long element names, extreme values, error states).

6. **The hierarchy is not strictly linear.** Organisms can contain other
   organisms. Templates can nest. The levels are a mental model, not a rigid
   tree.

7. **Abstract to concrete.** Atoms through Templates are increasingly concrete
   *structures*; Pages add *content*. This separation keeps the system testable
   at every level.

### Decomposition of the EMI Shield Designer UI

```
ATOMS
  ColorToken          -- #0a0a0f, #00d4ff, #9898b0, etc. (design tokens)
  TypographyToken     -- text-xs, text-sm, font-semibold, tracking-widest
  Icon                -- Shield, Zap, Play, Bot, AlertCircle (lucide wrappers)
  Button              -- primary, secondary, ghost, icon-only variants
  Badge               -- status dot + label (online/offline/checking)
  Input               -- numeric, text, with optional unit suffix
  Slider              -- range input with value display
  Tooltip             -- contextual help text

MOLECULES
  LabeledInput        -- Label + Input + optional unit
  StatusIndicator     -- Badge(dot) + label text
  NavLink             -- Icon + text + active-state styling
  SectionHeader       -- colored bar + uppercase label
  CompositionBadge    -- mono-font element summary (e.g. "Fe60-Ni30-Co10")
  ModeSelector        -- group of radio/toggle buttons for analysis mode
  ValidationBanner    -- AlertCircle icon + message text

MOLECULES (domain-specific)
  FrequencyInput      -- LabeledInput configured for MHz
  ThicknessInput      -- LabeledInput configured for mm
  GrainSizeInput      -- LabeledInput configured for um
  ElementTile         -- periodic table cell (symbol + name + selection state)

ORGANISMS
  TopNavBar           -- Logo(NavLink) + NavLinks + CompositionBadge + ModeSelector + ChatToggle(Button)
  PeriodicTable       -- grid of ElementTile molecules
  CompositionPanel    -- PeriodicTable + selected-elements list + percentage controls
  ShieldParameters    -- analysis-mode tabs + FrequencyInput + ThicknessInput + GrainSizeInput + sweep range controls
  ResultsPanel        -- calculation result cards + sweep charts + optimization output
  ChatSidebar         -- message list + input + AI response rendering
  FeatureCardGrid     -- grid of FeatureCard organisms
  SystemStatusBar     -- row of StatusIndicator molecules

TEMPLATES
  SimulationLayout    -- three-column: [CompositionPanel | Parameters+Results | ChatSidebar]
  LandingLayout       -- hero + feature cards + status bar
  MaterialsLayout     -- search/filter bar + material table/grid

PAGES
  SimulationPage      -- SimulationLayout populated with real stores and API calls
  HomePage            -- LandingLayout with live health checks
  MaterialsPage       -- MaterialsLayout with material database
```

---

## Chapter 3 -- Tools of the Trade (Pattern Libraries)

### What is a pattern library?

A pattern library (or component library) is a living, browsable collection of
every atom, molecule, and organism in the system. Each entry shows:

- The rendered component in all its variants and states
- The code required to use it
- Usage guidelines and do/don't examples
- Responsive behavior
- Accessibility notes

### Key principles

1. **The pattern library IS the source of truth.** It is not documentation about
   the code; it is generated from the same code that ships in production.
   Storybook, Pattern Lab, or a custom Next.js route like `/design-system` can
   serve this role.

2. **Show, don't tell.** Every component should be viewable in isolation with
   controls to manipulate its props (variant, size, disabled state, loading
   state, error state).

3. **Viewport-agnostic testing.** Each component should be resizable in the
   pattern library so designers and engineers can verify responsive behavior.

4. **Contextual documentation.** Each pattern entry should describe when to use
   it, when not to use it, and what accessibility attributes are required.

5. **Language-agnostic naming.** Component names should describe what the
   component *is*, not what it *looks like*. `ValidationBanner` is better than
   `YellowWarningBox`. `StatusIndicator` is better than `GreenDot`.

6. **Nested patterns.** The library should show both isolated atoms and composed
   molecules/organisms so developers can see how building blocks combine.

### Applied to EMI Shield Designer

Recommended tooling for this project:

- **Storybook** for the Next.js frontend: each component in
  `frontend/components/` gets a `.stories.tsx` file with knobs for all props.
- **Design token file** (`tokens.ts` or `tokens.css`): centralize every color,
  spacing value, border radius, and shadow currently scattered as Tailwind
  classes across pages.
- **Component index route** (`/design-system`): a lightweight alternative to
  Storybook if the team prefers an in-app approach.

---

## Chapter 4 -- The Atomic Workflow

### The process

Frost advocates for a workflow that simultaneously designs components *and*
assembles pages, rather than doing them sequentially. The key steps:

1. **Interface Inventory (Audit)**
   - Screenshot every unique button, input, card, header, modal across the
     existing application.
   - Group duplicates and near-duplicates.
   - Identify inconsistencies (e.g., three different shades of cyan used for
     primary actions).
   - This is the foundation for rationalizing the system.

2. **Establish Design Principles**
   - Define the non-negotiable qualities of the interface: "Scientific
     precision," "Dark-mode first," "Accessible at every level," "Data density
     without clutter."

3. **Build Atoms First, in the Browser**
   - Do not design atoms in Figma and hand them off. Build them in code (React
     components) from day one. This eliminates the "Figma drift" problem where
     the implemented component diverges from the design file.

4. **Compose Upward**
   - Combine atoms into molecules, molecules into organisms, organisms into
     templates. At each stage, review in the browser, not in a static mockup.

5. **Content Reference Sessions**
   - Use Pages (with real or realistic data) to stress-test templates. For the
     EMI platform: what happens when a composition has 8 elements? When a sweep
     produces 500 data points? When an error message is three lines long?

6. **Component-Driven Development**
   - Each pull request should ideally modify one component at one level. This
     makes code review focused and reduces merge conflicts.

### Maintaining consistency across a large application

1. **Design tokens as the single source of truth.** Every color, spacing value,
   font size, shadow, and transition duration lives in one file. Components
   reference tokens, never raw values.

   ```ts
   // tokens.ts
   export const colors = {
     bg: {
       primary: '#0a0a0f',
       surface: '#12121a',
       elevated: '#1a1a2e',
     },
     border: {
       default: '#2a2a3e',
       hover: '#3a3a52',
       active: 'rgba(0, 212, 255, 0.4)',
     },
     text: {
       primary: '#e8e8f0',
       secondary: '#9898b0',
       accent: '#00d4ff',
       warning: '#f59e0b',
     },
     // ...
   }
   ```

2. **Prop-driven variants, not className overrides.** A `<Button variant="primary">` is more maintainable than a `<button className="bg-gradient-to-r from-cyan-500 to-blue-600 ...">`. The variant logic lives inside the Button component.

3. **Composition over configuration.** Prefer composing small components over
   building mega-components with many conditional branches.

4. **Naming conventions enforced by linting.** Component file names, prop names,
   and CSS custom properties should follow a strict convention.

### Interface inventory for EMI Shield Designer

A quick audit of the current codebase reveals these inconsistencies to resolve:

| Area              | Issue                                                                     |
| ----------------- | ------------------------------------------------------------------------- |
| Colors            | Hard-coded hex values in 6+ files; no shared token file                   |
| Buttons           | At least 3 different button styles defined inline                         |
| Section headers   | Two different patterns (gradient bar + uppercase vs. plain text)          |
| Cards             | `card` and `card-hover` CSS classes exist but inline overrides are common |
| Status indicators | Duplicated dot+label pattern in HomePage and TopNavBar                    |
| Spacing           | Mix of `p-4`, `p-5`, `p-6`, `px-6 py-5` with no clear rationale         |

---

## Chapter 5 -- Maintaining Design Systems

### The system is never finished

A design system is a product, not a project. It requires ongoing investment,
governance, and iteration -- just like the application it supports.

### Key principles

1. **Make it official.** The design system needs an owner (a person or a team).
   Without ownership, entropy wins and components drift.

2. **Holy, defined, flexible.** Frost classifies patterns into three tiers:
   - **Holy**: core tokens and foundational atoms. Changes here affect
     everything and require careful review (colors, typography, spacing scale).
   - **Defined**: stable molecules and organisms with documented APIs. Changes
     are allowed but must go through review.
   - **Flexible**: page-level compositions and experimental patterns. Teams can
     iterate freely here.

3. **Make changes defined and visible.** Every component change should be
   versioned and announced. A changelog or visual diff tool (like Chromatic for
   Storybook) catches unintended regressions.

4. **Browser testing, not pixel-matching.** Verify components in real browsers
   across viewports. Automated visual regression tests (Percy, Chromatic) catch
   drift better than manual review.

5. **Evolve, do not revolution.** Deprecate patterns gradually. When replacing a
   component, keep the old one available with a deprecation warning until all
   consumers migrate.

6. **Measure adoption.** Track what percentage of the application uses the
   design system versus custom one-off styles. High adoption = high consistency.

7. **Reduce friction for contributors.** Make it easy to add new patterns:
   provide a component template, automated tests, and documentation scaffolding.

8. **Regular audits.** Periodically re-run the interface inventory to detect
   new inconsistencies that have crept in.

### Governance model for EMI Shield Designer

Given the project's size, a lightweight governance model is appropriate:

- **Token changes** (colors, typography, spacing): require explicit review and a
  note in the changelog.
- **Atom/molecule changes**: require one reviewer who checks the Storybook
  output.
- **Organism/template changes**: standard PR review.
- **Page changes**: no special process beyond normal code review.

---

## Consolidated Action Plan for EMI Shield Designer

### Phase 1: Foundation (design tokens + atoms)

1. Create `frontend/lib/tokens.ts` extracting all hard-coded colors, spacing,
   and typography values from existing components.
2. Build atomic components in `frontend/components/ui/`:
   - `Button.tsx` (primary, secondary, ghost, icon-only, loading, disabled)
   - `Input.tsx` (numeric, text, with unit suffix support)
   - `Badge.tsx` (status dot + label)
   - `SectionHeader.tsx` (gradient bar + uppercase label)
   - `Icon.tsx` (thin wrapper over lucide-react for consistent sizing)
   - `Card.tsx` (surface, elevated, interactive variants)
3. Refactor existing pages to consume these atoms instead of inline styles.

### Phase 2: Molecules

4. Build domain-aware molecules in `frontend/components/simulation/`:
   - `LabeledInput.tsx` (combines Label + Input + unit)
   - `FrequencyInput.tsx`, `ThicknessInput.tsx`, `GrainSizeInput.tsx`
   - `ElementTile.tsx` (periodic table cell)
   - `ModeSelector.tsx` (analysis mode toggle group)
   - `ValidationBanner.tsx` (extract from SimulationPage into reusable molecule)
   - `CompositionBadge.tsx`
5. Build navigation molecules in `frontend/components/nav/`:
   - `NavLink.tsx`
   - `StatusIndicator.tsx` (consolidate the two existing implementations)

### Phase 3: Organisms + templates

6. Refactor organisms to compose molecules:
   - `TopNavBar.tsx` uses NavLink, CompositionBadge, ModeSelector, Button
   - `CompositionPanel.tsx` uses PeriodicTable (organism) + LabeledInput molecules
   - `ShieldParameters.tsx` uses ModeSelector + domain-specific input molecules
   - `ResultsPanel.tsx` uses Card atoms + chart components
7. Extract layout templates:
   - `SimulationLayout.tsx` (three-column with collapsible sidebar)
   - `LandingLayout.tsx` (hero + cards + status)
8. Pages become thin: they select a template and wire up stores/API calls.

### Phase 4: Pattern library + governance

9. Add Storybook (or `/design-system` route) with stories for every atom and
   molecule.
10. Add visual regression testing to CI.
11. Document component APIs and usage guidelines inline via JSDoc or MDX.
12. Establish the holy/defined/flexible classification for each component tier.

---

## Quick Reference: Atomic Design Principles

| #  | Principle                                          | Practical rule                                                                 |
| -- | -------------------------------------------------- | ------------------------------------------------------------------------------ |
| 1  | Design systems, not pages                          | Every UI element should exist as a named, reusable component                   |
| 2  | Atoms are context-free                             | No atom should know which page it lives on                                     |
| 3  | Molecules have a single responsibility             | One user action per molecule                                                   |
| 4  | Organisms are where domain logic lives             | Wire together molecules with business/domain rules                             |
| 5  | Templates define spatial structure                 | Layout only; no hard-coded content                                             |
| 6  | Pages prove the system works with real data         | Use realistic edge-case content to stress-test                                 |
| 7  | Build in the browser, not in mockup tools          | Components are code artifacts first                                            |
| 8  | Design tokens are the single source of truth       | Never hard-code a color, spacing, or font size in a component                  |
| 9  | Prop-driven variants over className overrides       | `<Button variant="primary">` not `<button className="bg-cyan-500...">`         |
| 10 | The pattern library IS the documentation           | Auto-generated from production code                                            |
| 11 | Conduct regular interface inventories              | Audit for drift and inconsistency quarterly                                    |
| 12 | The design system is a product, not a project      | It needs an owner, a roadmap, and continuous maintenance                       |

---

*Reference: Brad Frost, Atomic Design (2016). https://atomicdesign.bradfrost.com*
