'use client'

import { useWorkbenchStore } from '@/lib/store'

export function ViewportCanvas() {
  const { thickness, composition } = useWorkbenchStore()

  return (
    <div className="w-full h-full flex flex-col items-center justify-center border border-border-panel bg-[#0d0d10] rounded relative overflow-hidden">
      
      <div className="absolute top-4 left-4 bg-bg-panel border border-border-panel px-3 py-1.5 rounded text-[10px] font-mono text-text-secondary z-10 flex flex-col gap-1">
        <span>Geometry Viewer (2D Cross-section)</span>
        <span>Layer 1: {Object.entries(composition).map(([k,v]) => `${k}(${v}%)`).join(', ')}</span>
        <span>Thickness: {thickness} mm</span>
      </div>

      {/* Axis Guide */}
      <div className="absolute bottom-4 left-4 flex flex-col items-center">
        <div className="w-px h-8 bg-[#238636] relative">
          <span className="absolute -top-3 left-1 text-[10px] font-mono text-[#238636]">y</span>
        </div>
        <div className="flex items-center">
          <div className="w-2 h-2 rounded-full bg-border-panel border border-text-muted" title="z-axis (out of page)" />
          <div className="h-px w-8 bg-[#da3633] relative">
            <span className="absolute -right-3 -top-1.5 text-[10px] font-mono text-[#da3633]">x</span>
          </div>
        </div>
      </div>

      {/* Wave Visualization SVG */}
      <svg width="600" height="300" viewBox="0 0 600 300" className="opacity-80">
        <defs>
          <pattern id="grid" width="20" height="20" patternUnits="userSpaceOnUse">
            <path d="M 20 0 L 0 0 0 20" fill="none" stroke="#1c1c22" strokeWidth="0.5"/>
          </pattern>
        </defs>
        <rect width="600" height="300" fill="url(#grid)" />

        {/* Incident Wave (Red) */}
        <path d="M0,150 Q25,100 50,150 T100,150 T150,150 T200,150" fill="none" stroke="#da3633" strokeWidth="2" />
        <text x="50" y="130" fill="#da3633" fontSize="10" fontFamily="monospace">Incident Wave (E)</text>

        {/* Reflected Wave (Yellow) */}
        <path d="M200,150 Q175,180 150,150 T100,150 T50,150" fill="none" stroke="#d29922" strokeWidth="1.5" strokeDasharray="4,4" />
        
        {/* Shield Material Block */}
        <rect x="200" y="50" width={Math.max(20, Math.min(200, thickness * 50))} height="200" fill="#005fb8" fillOpacity="0.2" stroke="#005fb8" strokeWidth="2" />
        
        {/* Transmitted Wave (Green - Attenuated) */}
        <path d={`M${200 + Math.max(20, Math.min(200, thickness * 50))},150 Q${225 + Math.max(20, Math.min(200, thickness * 50))},135 ${250 + Math.max(20, Math.min(200, thickness * 50))},150 T${300 + Math.max(20, Math.min(200, thickness * 50))},150`} fill="none" stroke="#238636" strokeWidth="1" />
        <text x={220 + Math.max(20, Math.min(200, thickness * 50))} y="130" fill="#238636" fontSize="10" fontFamily="monospace">Transmitted Wave</text>

        {/* Thickness Dimension line */}
        <line x1="200" y1="270" x2={200 + Math.max(20, Math.min(200, thickness * 50))} y2="270" stroke="#8b8b99" strokeWidth="1" markerEnd="url(#arrow)" markerStart="url(#arrow)" />
        <text x={200 + (Math.max(20, Math.min(200, thickness * 50)) / 2) - 15} y="285" fill="#8b8b99" fontSize="10" fontFamily="monospace">{thickness}mm</text>
      </svg>
      
    </div>
  )
}
