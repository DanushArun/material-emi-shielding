'use client'

import Link from 'next/link'
import { Hexagon, Activity, LayoutGrid } from 'lucide-react'
import { cn } from '@/lib/utils'
import { usePathname } from 'next/navigation'

export function Header() {
  const pathname = usePathname()

  const links = [
    { href: '/', label: 'Start Page', icon: Hexagon },
    { href: '/simulation', label: 'Workbench', icon: Activity },
    { href: '/materials', label: 'Material Database', icon: LayoutGrid },
  ]

  return (
    <header className="sticky top-0 z-50 h-12 border-b border-border-panel bg-[#0d0d10] flex items-center justify-between px-4">
      
      <div className="flex items-center gap-6 h-full">
        {/* Branding */}
        <div className="flex items-center gap-2 pr-4 border-r border-border-panel h-full">
          <div className="w-5 h-5 bg-accent-selection rounded-sm flex items-center justify-center">
            <span className="text-white font-mono font-bold text-[10px]">TMM</span>
          </div>
          <span className="font-display font-semibold text-xs tracking-wide text-text-primary">
            EMI Shield Designer
          </span>
        </div>

        {/* Navigation */}
        <nav className="flex items-center gap-1 h-full">
          {links.map((link) => {
            const isActive = pathname === link.href
            return (
              <Link
                key={link.href}
                href={link.href}
                className={cn(
                  "h-full px-3 flex items-center gap-2 text-xs font-medium transition-colors border-b-2",
                  isActive
                    ? "border-accent-selection text-text-primary bg-bg-panel"
                    : "border-transparent text-text-secondary hover:text-text-primary hover:bg-bg-panel/50"
                )}
              >
                <link.icon className={cn("w-3.5 h-3.5", isActive ? "text-accent-selection" : "text-text-muted")} />
                {link.label}
              </Link>
            )
          })}
        </nav>
      </div>

      {/* System Status */}
      <div className="flex items-center gap-3">
        <div className="flex items-center gap-2">
          <div className="w-2 h-2 rounded-full bg-accent-success" />
          <span className="text-[10px] font-mono text-text-secondary">Ready</span>
        </div>
      </div>
      
    </header>
  )
}
