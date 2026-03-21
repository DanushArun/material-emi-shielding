import type { Metadata } from 'next'
import { Inter } from 'next/font/google'
import './globals.css'
import { Providers } from './providers'
import { Header } from '@/components/layout/Header'

const inter = Inter({ subsets: ['latin'], variable: '--font-inter' })

export const metadata: Metadata = {
  title: 'EMI Shield Designer',
  description: 'AI-powered electromagnetic interference shielding simulation platform',
}

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className="dark">
      <body className={`${inter.variable} antialiased selection:bg-accent-primary/30`}>
        <Providers>
          <div className="min-h-screen flex flex-col bg-bg-primary text-text-primary">
            <Header />
            <main className="flex-1 flex flex-col relative z-0">
              {children}
            </main>
          </div>
        </Providers>
      </body>
    </html>
  )
}
