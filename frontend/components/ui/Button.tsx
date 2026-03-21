import { ButtonHTMLAttributes, forwardRef } from 'react'
import { cn } from '@/lib/utils'

interface ButtonProps extends ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: 'primary' | 'secondary' | 'ghost' | 'danger'
  size?: 'sm' | 'md'
}

export const Button = forwardRef<HTMLButtonElement, ButtonProps>(
  ({ className, variant = 'primary', size = 'md', ...props }, ref) => {
    return (
      <button
        ref={ref}
        className={cn(
          "inline-flex items-center justify-center font-medium transition-colors disabled:opacity-50 disabled:cursor-not-allowed",
          size === 'sm' ? "px-2 py-1 text-xs rounded-sm" : "px-4 py-1.5 text-xs rounded",
          variant === 'primary' && "bg-accent-selection text-white hover:bg-[#0070d6]",
          variant === 'secondary' && "bg-bg-subpanel text-text-primary border border-border-panel hover:border-text-secondary",
          variant === 'ghost' && "bg-transparent text-text-secondary hover:text-text-primary hover:bg-bg-subpanel",
          variant === 'danger' && "bg-transparent text-accent-danger border border-accent-danger hover:bg-accent-danger hover:text-white",
          className
        )}
        {...props}
      />
    )
  }
)
Button.displayName = 'Button'
