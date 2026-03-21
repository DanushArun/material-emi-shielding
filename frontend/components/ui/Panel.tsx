'use client'

import { HTMLAttributes } from 'react'
import { cn } from '@/lib/utils'

interface PanelProps extends HTMLAttributes<HTMLDivElement> {
  title?: string
  noPadding?: boolean
}

export function Panel({ className, title, noPadding = false, children, ...props }: PanelProps) {
  return (
    <div
      className={cn(
        "bg-bg-panel border border-border-panel flex flex-col",
        className
      )}
      {...props}
    >
      {title && (
        <div className="px-3 py-2 border-b border-border-panel bg-bg-subpanel flex items-center">
          <span className="text-xs font-semibold text-text-primary tracking-wide">{title}</span>
        </div>
      )}
      <div className={cn("flex-1 overflow-auto", !noPadding && "p-4")}>
        {children}
      </div>
    </div>
  )
}
