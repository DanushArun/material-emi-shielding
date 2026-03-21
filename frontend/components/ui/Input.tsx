import { InputHTMLAttributes, forwardRef } from 'react'
import { cn } from '@/lib/utils'

interface InputProps extends InputHTMLAttributes<HTMLInputElement> {
  label?: string
  unit?: string
}

export const Input = forwardRef<HTMLInputElement, InputProps>(
  ({ className, label, unit, id, ...props }, ref) => {
    return (
      <div className="flex flex-col gap-1.5 w-full">
        {label && (
          <label htmlFor={id} className="text-xs font-medium tracking-[0.04em] uppercase text-text-secondary">
            {label}
          </label>
        )}
        <div className="relative flex items-center">
          <input
            id={id}
            ref={ref}
            className={cn(
              "w-full bg-glass-bg border border-glass-border rounded-lg px-4 py-3 text-sm text-text-primary",
              "transition-all duration-200 focus:outline-none focus:border-accent-primary focus:shadow-[0_0_0_3px_rgba(0,212,255,0.1)]",
              className
            )}
            {...props}
          />
          {unit && (
            <span className="absolute right-4 text-xs font-medium text-text-muted">
              {unit}
            </span>
          )}
        </div>
      </div>
    )
  }
)
Input.displayName = 'Input'
