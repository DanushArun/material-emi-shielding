'use client'

import Link from 'next/link'
import { ArrowRight, Activity, Database, GitMerge, FileBarChart2 } from 'lucide-react'
import { Button } from '@/components/ui/Button'

export default function Home() {
  return (
    <div className="flex flex-col min-h-[calc(100vh-64px)] bg-bg-canvas overflow-y-auto">
      <main className="flex-1 max-w-7xl w-full mx-auto px-6 py-12 flex flex-col gap-16">
        
        {/* Hero Section */}
        <section className="flex flex-col md:flex-row gap-12 items-center">
          <div className="flex-1 flex flex-col gap-6">
            <div className="inline-flex items-center gap-2 px-3 py-1 rounded bg-bg-panel border border-border-panel w-fit">
              <Activity className="w-4 h-4 text-accent-selection" />
              <span className="text-xs font-medium text-text-primary tracking-wide">Version 4.0 | Transfer Matrix Engine</span>
            </div>
            
            <h1 className="font-display text-5xl font-bold text-white tracking-tight leading-tight">
              Advanced EMI Shielding <br/>
              <span className="text-text-secondary">Analysis Environment</span>
            </h1>
            
            <p className="text-sm text-text-secondary leading-relaxed max-w-lg">
              A high-fidelity computer-aided engineering (CAE) platform for evaluating electromagnetic interference. Configure complex composites, run sweeping parametric studies, and visualize spatial wave attenuation.
            </p>
            
            <div className="flex items-center gap-4 pt-2">
              <Link href="/simulation">
                <Button variant="primary" className="gap-2 px-6 py-2">
                  Launch Workbench <ArrowRight className="w-4 h-4" />
                </Button>
              </Link>
              <Link href="/materials">
                <Button variant="secondary" className="px-6 py-2 gap-2">
                  <Database className="w-4 h-4" /> Material Database
                </Button>
              </Link>
            </div>
          </div>
          
          {/* Hero Visual - Simulated Software Screenshot */}
          <div className="flex-1 w-full bg-bg-panel border border-border-panel rounded-lg shadow-2xl overflow-hidden flex flex-col">
            <div className="h-8 bg-bg-subpanel border-b border-border-panel flex items-center px-4 gap-2">
              <div className="w-2.5 h-2.5 rounded-full bg-border-panel"></div>
              <div className="w-2.5 h-2.5 rounded-full bg-border-panel"></div>
              <div className="w-2.5 h-2.5 rounded-full bg-border-panel"></div>
              <span className="ml-2 text-[10px] text-text-muted font-mono">emi-shield-designer-v4</span>
            </div>
            <div className="p-6 flex flex-col gap-4">
              <div className="flex justify-between items-end border-b border-border-panel pb-2">
                <span className="text-xs font-semibold text-text-primary">Frequency Sweep: Cu-Ni Composite</span>
                <span className="text-[10px] text-text-secondary font-mono">1 GHz - 10 GHz | 100 Points</span>
              </div>
              <div className="h-48 w-full border border-border-panel bg-bg-canvas relative overflow-hidden flex items-end">
                {/* Mock Chart lines */}
                <svg className="w-full h-full" preserveAspectRatio="none" viewBox="0 0 100 100">
                  <path d="M0,80 Q25,20 50,40 T100,10" fill="none" stroke="#005fb8" strokeWidth="2" />
                  <path d="M0,90 Q30,60 60,70 T100,50" fill="none" stroke="#238636" strokeWidth="1" strokeDasharray="4,2" />
                  <rect x="0" y="0" width="100" height="100" fill="url(#grid)" />
                  <defs>
                    <pattern id="grid" width="10" height="10" patternUnits="userSpaceOnUse">
                      <path d="M 10 0 L 0 0 0 10" fill="none" stroke="#2d2d33" strokeWidth="0.5"/>
                    </pattern>
                  </defs>
                </svg>
              </div>
            </div>
          </div>
        </section>

        {/* Feature Grid */}
        <section className="grid grid-cols-1 md:grid-cols-3 gap-6 pt-12 border-t border-border-panel">
          <FeatureCard 
            icon={<GitMerge className="w-5 h-5 text-accent-selection" />}
            title="Transfer Matrix Method"
            desc="Solve Maxwell's equations across multi-layered planar media. Accurately model wave impedance, reflection, and absorption coefficients."
          />
          <FeatureCard 
            icon={<FileBarChart2 className="w-5 h-5 text-accent-success" />}
            title="Parametric Sweeps"
            desc="Configure multi-variable parameter spaces. Sweep across frequency spectrums, material thicknesses, or varying composite ratios."
          />
          <FeatureCard 
            icon={<Activity className="w-5 h-5 text-accent-warning" />}
            title="Spatial Visualization"
            desc="Go beyond scalar results. Generate high-resolution heatmaps and spatial E-field attenuation curves through the shield thickness."
          />
        </section>

      </main>
    </div>
  )
}

function FeatureCard({ icon, title, desc }: { icon: React.ReactNode, title: string, desc: string }) {
  return (
    <div className="bg-bg-panel border border-border-panel p-6 rounded flex flex-col gap-3">
      <div className="w-8 h-8 flex items-center justify-center bg-bg-subpanel border border-border-panel rounded">
        {icon}
      </div>
      <h3 className="text-sm font-semibold text-text-primary">{title}</h3>
      <p className="text-xs text-text-secondary leading-relaxed">{desc}</p>
    </div>
  )
}
