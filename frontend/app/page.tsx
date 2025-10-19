'use client'

import { useState } from 'react'
import { Shield, Zap, BarChart3, Settings } from 'lucide-react'

export default function Home() {
  const [apiStatus, setApiStatus] = useState<string>('checking...')

  // Check API health on mount
  useState(() => {
    fetch(`${process.env.NEXT_PUBLIC_API_URL}/health`)
      .then(res => res.json())
      .then(data => setApiStatus(data.status || 'unknown'))
      .catch(() => setApiStatus('offline'))
  })

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 via-white to-purple-50">
      {/* Header */}
      <header className="bg-white/80 backdrop-blur-sm border-b border-gray-200 sticky top-0 z-10">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-4">
          <div className="flex justify-between items-center">
            <div className="flex items-center space-x-3">
              <Shield className="w-8 h-8 text-blue-600" />
              <div>
                <h1 className="text-2xl font-bold text-gray-900">EMI Shield Designer</h1>
                <p className="text-sm text-gray-600">Version 4.0 - Microservices Architecture</p>
              </div>
            </div>
            <div className="flex items-center space-x-4">
              <div className="flex items-center space-x-2">
                <div className={`w-2 h-2 rounded-full ${apiStatus === 'healthy' ? 'bg-green-500' : 'bg-red-500'}`} />
                <span className="text-sm text-gray-600">API: {apiStatus}</span>
              </div>
              <button className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors">
                Sign In
              </button>
            </div>
          </div>
        </div>
      </header>

      {/* Hero Section */}
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
        <div className="text-center mb-16">
          <h2 className="text-5xl font-bold text-gray-900 mb-4">
            Next-Generation EMI Shielding Design
          </h2>
          <p className="text-xl text-gray-600 max-w-3xl mx-auto">
            AI-powered electromagnetic interference shielding calculator with real-time physics simulations,
            material optimization, and collaborative design tools.
          </p>
        </div>

        {/* Feature Cards */}
        <div className="grid md:grid-cols-3 gap-8 mb-16">
          <FeatureCard
            icon={<Zap className="w-8 h-8" />}
            title="Physics-Based Calculations"
            description="Accurate EMI shielding predictions using Maxwell's equations and advanced microstructure modeling"
            color="blue"
          />
          <FeatureCard
            icon={<BarChart3 className="w-8 h-8" />}
            title="Real-Time Analysis"
            description="Frequency sweeps, thickness optimization, and multi-parameter visualizations in milliseconds"
            color="purple"
          />
          <FeatureCard
            icon={<Settings className="w-8 h-8" />}
            title="Material Database"
            description="Extensive library of EMI shielding materials with composition-based property calculations"
            color="green"
          />
        </div>

        {/* Quick Start Section */}
        <div className="bg-white rounded-2xl shadow-xl p-8 mb-16">
          <h3 className="text-2xl font-bold text-gray-900 mb-6">Quick Start</h3>
          <div className="grid md:grid-cols-2 gap-6">
            <div className="border-2 border-dashed border-gray-300 rounded-xl p-6 hover:border-blue-500 transition-colors cursor-pointer">
              <h4 className="text-lg font-semibold text-gray-900 mb-2">Calculate Shielding</h4>
              <p className="text-gray-600 mb-4">
                Get instant shielding effectiveness calculations for your material composition
              </p>
              <button className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors">
                Start Calculation
              </button>
            </div>
            <div className="border-2 border-dashed border-gray-300 rounded-xl p-6 hover:border-purple-500 transition-colors cursor-pointer">
              <h4 className="text-lg font-semibold text-gray-900 mb-2">Explore Materials</h4>
              <p className="text-gray-600 mb-4">
                Browse our curated database of EMI shielding materials and their properties
              </p>
              <button className="px-4 py-2 bg-purple-600 text-white rounded-lg hover:bg-purple-700 transition-colors">
                View Materials
              </button>
            </div>
          </div>
        </div>

        {/* System Status */}
        <div className="bg-white rounded-2xl shadow-xl p-8">
          <h3 className="text-2xl font-bold text-gray-900 mb-6">System Status</h3>
          <div className="grid md:grid-cols-3 gap-6">
            <StatusCard label="Backend API" status={apiStatus} />
            <StatusCard label="Database" status="pending" />
            <StatusCard label="ML Models" status="pending" />
          </div>
          <div className="mt-6 p-4 bg-blue-50 rounded-lg">
            <p className="text-sm text-blue-800">
              <strong>Development Mode:</strong> The microservices architecture is now active.
              Use <code className="bg-blue-100 px-2 py-1 rounded">docker-compose up</code> to start all services.
            </p>
          </div>
        </div>
      </main>

      {/* Footer */}
      <footer className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 text-center text-gray-600">
        <p>EMI Shield Designer v4.0.0 - Microservices Architecture</p>
        <p className="text-sm mt-2">API Documentation: <a href="http://localhost:8000/docs" className="text-blue-600 hover:underline">http://localhost:8000/docs</a></p>
      </footer>
    </div>
  )
}

function FeatureCard({ icon, title, description, color }: {
  icon: React.ReactNode
  title: string
  description: string
  color: string
}) {
  const colorClasses = {
    blue: 'bg-blue-100 text-blue-600',
    purple: 'bg-purple-100 text-purple-600',
    green: 'bg-green-100 text-green-600',
  }

  return (
    <div className="bg-white rounded-xl shadow-lg p-6 hover:shadow-xl transition-shadow">
      <div className={`w-16 h-16 rounded-lg ${colorClasses[color as keyof typeof colorClasses]} flex items-center justify-center mb-4`}>
        {icon}
      </div>
      <h3 className="text-xl font-semibold text-gray-900 mb-2">{title}</h3>
      <p className="text-gray-600">{description}</p>
    </div>
  )
}

function StatusCard({ label, status }: { label: string; status: string }) {
  const statusColors = {
    healthy: 'text-green-600 bg-green-100',
    offline: 'text-red-600 bg-red-100',
    pending: 'text-yellow-600 bg-yellow-100',
    checking: 'text-gray-600 bg-gray-100',
  }

  const statusColor = statusColors[status as keyof typeof statusColors] || statusColors.checking

  return (
    <div className="border border-gray-200 rounded-lg p-4">
      <div className="flex items-center justify-between">
        <span className="text-gray-700 font-medium">{label}</span>
        <span className={`px-3 py-1 rounded-full text-sm font-medium ${statusColor}`}>
          {status}
        </span>
      </div>
    </div>
  )
}
