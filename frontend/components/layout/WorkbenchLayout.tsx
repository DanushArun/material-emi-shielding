'use client'

import { ReactNode } from 'react'

interface WorkbenchLayoutProps {
  ribbon: ReactNode
  tree: ReactNode
  viewport: ReactNode
  properties: ReactNode
  consolePanel: ReactNode
}

export function WorkbenchLayout({ ribbon, tree, viewport, properties, consolePanel }: WorkbenchLayoutProps) {
  return (
    <div className="flex flex-col h-[calc(100vh-48px)] overflow-hidden bg-bg-canvas">
      {/* Top Ribbon */}
      <div className="h-10 shrink-0 border-b border-border-panel bg-bg-panel flex items-center px-4">
        {ribbon}
      </div>

      <div className="flex-1 flex overflow-hidden">
        {/* Left: Project Tree */}
        <div className="w-64 shrink-0 border-r border-border-panel bg-bg-panel flex flex-col">
          <div className="px-3 py-1.5 border-b border-border-panel bg-bg-subpanel flex items-center">
            <span className="text-[10px] font-semibold uppercase tracking-wider text-text-secondary">Project Manager</span>
          </div>
          <div className="flex-1 overflow-auto p-1">
            {tree}
          </div>
        </div>

        {/* Center: Main Viewport + Console */}
        <div className="flex-1 flex flex-col min-w-0 relative">
          <div className="flex-1 overflow-auto p-2 bg-bg-canvas">
            {viewport}
          </div>
          {/* Bottom Console */}
          <div className="h-48 shrink-0 border-t border-border-panel">
            {consolePanel}
          </div>
        </div>

        {/* Right: Property Editor */}
        <div className="w-72 shrink-0 border-l border-border-panel bg-bg-panel flex flex-col">
          <div className="px-3 py-1.5 border-b border-border-panel bg-bg-subpanel flex items-center">
            <span className="text-[10px] font-semibold uppercase tracking-wider text-text-secondary">Properties</span>
          </div>
          <div className="flex-1 overflow-auto p-3">
            {properties}
          </div>
        </div>
      </div>
    </div>
  )
}
