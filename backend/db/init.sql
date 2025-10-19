-- EMI Shield Designer Database Schema
-- Version 4.0 - Microservices Architecture

-- Enable UUID extension
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Users table
CREATE TABLE IF NOT EXISTS users (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    email VARCHAR(255) UNIQUE NOT NULL,
    username VARCHAR(100) UNIQUE NOT NULL,
    hashed_password VARCHAR(255) NOT NULL,
    full_name VARCHAR(255),
    is_active BOOLEAN DEFAULT TRUE,
    is_verified BOOLEAN DEFAULT FALSE,
    subscription_tier VARCHAR(50) DEFAULT 'free', -- free, pro, enterprise
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    last_login TIMESTAMP WITH TIME ZONE
);

-- Create index on email for faster lookups
CREATE INDEX idx_users_email ON users(email);
CREATE INDEX idx_users_subscription ON users(subscription_tier);

-- Materials table (user-saved materials)
CREATE TABLE IF NOT EXISTS materials (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    name VARCHAR(255) NOT NULL,
    description TEXT,
    composition JSONB NOT NULL, -- {element: percentage}
    conductivity DOUBLE PRECISION, -- S/m
    permeability DOUBLE PRECISION, -- relative
    permittivity DOUBLE PRECISION, -- relative
    density DOUBLE PRECISION, -- kg/m3
    grain_size DOUBLE PRECISION, -- micrometers
    is_public BOOLEAN DEFAULT FALSE,
    source VARCHAR(50) DEFAULT 'user', -- user, database, ai_generated
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_materials_user_id ON materials(user_id);
CREATE INDEX idx_materials_public ON materials(is_public) WHERE is_public = TRUE;
CREATE INDEX idx_materials_composition ON materials USING GIN(composition);

-- Calculations table (calculation history)
CREATE TABLE IF NOT EXISTS calculations (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    material_id UUID REFERENCES materials(id) ON DELETE SET NULL,
    calculation_type VARCHAR(50) NOT NULL, -- single, frequency_sweep, thickness_optimization
    input_params JSONB NOT NULL, -- All input parameters
    results JSONB NOT NULL, -- Calculation results
    execution_time_ms INTEGER, -- How long calculation took
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_calculations_user_id ON calculations(user_id);
CREATE INDEX idx_calculations_created_at ON calculations(created_at DESC);
CREATE INDEX idx_calculations_type ON calculations(calculation_type);

-- Projects table (for collaborative design)
CREATE TABLE IF NOT EXISTS projects (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    owner_id UUID REFERENCES users(id) ON DELETE CASCADE,
    name VARCHAR(255) NOT NULL,
    description TEXT,
    configuration JSONB, -- Project settings and requirements
    is_public BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_projects_owner_id ON projects(owner_id);

-- Project collaborators (many-to-many)
CREATE TABLE IF NOT EXISTS project_collaborators (
    project_id UUID REFERENCES projects(id) ON DELETE CASCADE,
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    role VARCHAR(50) DEFAULT 'viewer', -- owner, editor, viewer
    invited_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (project_id, user_id)
);

-- ML Model predictions cache
CREATE TABLE IF NOT EXISTS ml_predictions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    input_hash VARCHAR(64) UNIQUE NOT NULL, -- Hash of input parameters
    model_version VARCHAR(50) NOT NULL,
    input_features JSONB NOT NULL,
    prediction JSONB NOT NULL,
    confidence DOUBLE PRECISION,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_ml_predictions_hash ON ml_predictions(input_hash);
CREATE INDEX idx_ml_predictions_created_at ON ml_predictions(created_at DESC);

-- API usage tracking (for rate limiting and billing)
CREATE TABLE IF NOT EXISTS api_usage (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    endpoint VARCHAR(255) NOT NULL,
    method VARCHAR(10) NOT NULL,
    status_code INTEGER,
    response_time_ms INTEGER,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_api_usage_user_id ON api_usage(user_id);
CREATE INDEX idx_api_usage_timestamp ON api_usage(timestamp DESC);

-- Material library (curated database materials)
CREATE TABLE IF NOT EXISTS material_library (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(255) UNIQUE NOT NULL,
    category VARCHAR(100), -- copper, aluminum, steel, alloy, composite
    composition JSONB NOT NULL,
    conductivity DOUBLE PRECISION NOT NULL,
    permeability DOUBLE PRECISION DEFAULT 1.0,
    permittivity DOUBLE PRECISION DEFAULT 1.0,
    density DOUBLE PRECISION,
    cost_per_kg DOUBLE PRECISION,
    typical_applications TEXT[],
    datasheet_url TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_material_library_category ON material_library(category);
CREATE INDEX idx_material_library_name ON material_library(name);

-- Insert common EMI shielding materials
INSERT INTO material_library (name, category, composition, conductivity, permeability, density, typical_applications) VALUES
('Copper (Pure)', 'copper', '{"Cu": 100}', 5.96e7, 1.0, 8960, ARRAY['High-frequency shielding', 'RF enclosures']),
('Aluminum (Pure)', 'aluminum', '{"Al": 100}', 3.77e7, 1.0, 2700, ARRAY['Lightweight shielding', 'Aerospace']),
('Steel (Mild)', 'steel', '{"Fe": 99, "C": 1}', 1.0e7, 100, 7850, ARRAY['Structural shielding', 'Enclosures']),
('Nickel (Pure)', 'nickel', '{"Ni": 100}', 1.43e7, 100, 8900, ARRAY['Magnetic shielding', 'Batteries']),
('Silver (Pure)', 'silver', '{"Ag": 100}', 6.30e7, 1.0, 10490, ARRAY['Premium RF shielding', 'High-performance']),
('Mu-Metal', 'alloy', '{"Ni": 77, "Fe": 16, "Cu": 5, "Mo": 2}', 1.8e6, 20000, 8700, ARRAY['Low-frequency magnetic shielding', 'Transformers']),
('Brass (70/30)', 'alloy', '{"Cu": 70, "Zn": 30}', 1.57e7, 1.0, 8520, ARRAY['Moderate shielding', 'Connectors']),
('Stainless Steel 304', 'steel', '{"Fe": 70, "Cr": 19, "Ni": 10, "Mn": 1}', 1.45e6, 1.0, 8000, ARRAY['Corrosion-resistant shielding', 'Medical devices'])
ON CONFLICT (name) DO NOTHING;

-- Function to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Triggers for updated_at
CREATE TRIGGER update_users_updated_at BEFORE UPDATE ON users
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_materials_updated_at BEFORE UPDATE ON materials
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_projects_updated_at BEFORE UPDATE ON projects
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Create default admin user (password: admin123 - CHANGE IN PRODUCTION)
-- Hashed with bcrypt
INSERT INTO users (email, username, hashed_password, full_name, subscription_tier, is_verified)
VALUES (
    'admin@emishield.com',
    'admin',
    '$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/LewY5/4nR7VoGnvqy',
    'System Administrator',
    'enterprise',
    TRUE
) ON CONFLICT (email) DO NOTHING;
