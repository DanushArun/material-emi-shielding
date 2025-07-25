-- EMI Shielding Measurements Database Schema
-- PostgreSQL database for storing scraped EMI shielding data

-- Enable UUID extension
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Main materials table
CREATE TABLE IF NOT EXISTS materials (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(500) NOT NULL,
    material_class VARCHAR(100), -- metal, polymer, composite, carbon-based, etc.
    composition JSONB, -- {"Fe": 70, "Co": 30} or complex compositions
    synthesis_method TEXT,
    processing_temperature FLOAT, -- Celsius
    processing_time FLOAT, -- hours
    processing_atmosphere VARCHAR(100),
    particle_size FLOAT, -- nm
    particle_size_unit VARCHAR(20) DEFAULT 'nm',
    morphology VARCHAR(200), -- nanoparticles, fibers, sheets, etc.
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(name, composition)
);

-- EMI measurements table
CREATE TABLE IF NOT EXISTS emi_measurements (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    material_id UUID REFERENCES materials(id) ON DELETE CASCADE,
    
    -- Material properties
    conductivity FLOAT, -- S/m
    conductivity_unit VARCHAR(20) DEFAULT 'S/m',
    relative_permeability FLOAT,
    relative_permittivity FLOAT,
    density FLOAT, -- g/cm³
    
    -- Measurement conditions
    thickness FLOAT NOT NULL, -- mm
    thickness_unit VARCHAR(20) DEFAULT 'mm',
    frequency FLOAT NOT NULL, -- Hz
    frequency_unit VARCHAR(20) DEFAULT 'Hz',
    temperature FLOAT DEFAULT 25, -- Celsius
    
    -- EMI shielding results
    total_se FLOAT NOT NULL, -- dB
    reflection_loss FLOAT, -- dB
    absorption_loss FLOAT, -- dB
    multiple_reflection_loss FLOAT, -- dB
    
    -- For composites
    filler_loading FLOAT, -- percentage
    filler_type VARCHAR(20), -- wt% or vol%
    matrix_material VARCHAR(200),
    
    -- Measurement metadata
    measurement_standard VARCHAR(100), -- ASTM D4935, etc.
    measurement_technique VARCHAR(200), -- coaxial, waveguide, etc.
    sample_dimensions JSONB, -- {"diameter": 7, "thickness": 2}
    
    -- Source information
    source_type VARCHAR(50), -- paper, patent, database
    source_doi VARCHAR(200),
    source_title TEXT,
    source_authors TEXT,
    source_year INTEGER,
    source_url TEXT,
    
    -- Data quality
    confidence_score FLOAT DEFAULT 1.0, -- 0-1 confidence in data
    is_experimental BOOLEAN DEFAULT TRUE,
    is_validated BOOLEAN DEFAULT FALSE,
    validation_notes TEXT,
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Frequency sweep data (for full spectrum measurements)
CREATE TABLE IF NOT EXISTS frequency_sweeps (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    measurement_id UUID REFERENCES emi_measurements(id) ON DELETE CASCADE,
    frequencies FLOAT[], -- Array of frequencies
    shielding_effectiveness FLOAT[], -- Array of SE values
    reflection_loss FLOAT[], -- Array of RL values
    absorption_loss FLOAT[], -- Array of AL values
    frequency_unit VARCHAR(20) DEFAULT 'Hz',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Processing conditions table
CREATE TABLE IF NOT EXISTS processing_conditions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    material_id UUID REFERENCES materials(id) ON DELETE CASCADE,
    step_number INTEGER,
    process_type VARCHAR(200), -- ball milling, sintering, CVD, etc.
    temperature FLOAT,
    pressure FLOAT,
    pressure_unit VARCHAR(20),
    duration FLOAT, -- hours
    atmosphere VARCHAR(100),
    additional_parameters JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Literature sources table
CREATE TABLE IF NOT EXISTS sources (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    doi VARCHAR(200) UNIQUE,
    title TEXT NOT NULL,
    authors TEXT,
    journal VARCHAR(500),
    year INTEGER,
    volume VARCHAR(50),
    pages VARCHAR(50),
    abstract TEXT,
    keywords TEXT[],
    source_type VARCHAR(50), -- journal, conference, patent, thesis
    pdf_path TEXT,
    url TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Link measurements to sources
CREATE TABLE IF NOT EXISTS measurement_sources (
    measurement_id UUID REFERENCES emi_measurements(id) ON DELETE CASCADE,
    source_id UUID REFERENCES sources(id) ON DELETE CASCADE,
    page_number INTEGER,
    table_number VARCHAR(50),
    figure_number VARCHAR(50),
    notes TEXT,
    PRIMARY KEY (measurement_id, source_id)
);

-- Scraped data staging table (before validation)
CREATE TABLE IF NOT EXISTS scraped_data_staging (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    raw_data JSONB NOT NULL,
    source_url TEXT,
    scraper_name VARCHAR(100),
    extraction_method VARCHAR(100), -- table, graph, text
    is_processed BOOLEAN DEFAULT FALSE,
    processing_errors TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- User feedback table (for ML training)
CREATE TABLE IF NOT EXISTS user_feedback (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    measurement_id UUID REFERENCES emi_measurements(id),
    predicted_se FLOAT,
    actual_se FLOAT,
    feedback_type VARCHAR(50), -- correction, validation, dispute
    notes TEXT,
    user_email VARCHAR(255),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for better query performance
CREATE INDEX idx_materials_name ON materials(name);
CREATE INDEX idx_materials_composition ON materials USING GIN(composition);
CREATE INDEX idx_measurements_material ON emi_measurements(material_id);
CREATE INDEX idx_measurements_frequency ON emi_measurements(frequency);
CREATE INDEX idx_measurements_thickness ON emi_measurements(thickness);
CREATE INDEX idx_measurements_se ON emi_measurements(total_se);
CREATE INDEX idx_measurements_source ON emi_measurements(source_doi);
CREATE INDEX idx_sources_doi ON sources(doi);
CREATE INDEX idx_sources_year ON sources(year);

-- Create views for common queries
CREATE OR REPLACE VIEW material_performance AS
SELECT 
    m.name,
    m.composition,
    em.thickness,
    em.frequency,
    em.total_se,
    em.conductivity,
    em.relative_permeability,
    em.source_doi
FROM materials m
JOIN emi_measurements em ON m.id = em.material_id
WHERE em.is_validated = TRUE;

CREATE OR REPLACE VIEW high_performance_materials AS
SELECT DISTINCT
    m.name,
    m.composition,
    AVG(em.total_se) as avg_se,
    COUNT(*) as measurement_count
FROM materials m
JOIN emi_measurements em ON m.id = em.material_id
WHERE em.frequency BETWEEN 1e9 AND 10e9  -- 1-10 GHz
GROUP BY m.id, m.name, m.composition
HAVING AVG(em.total_se) > 60  -- > 60 dB
ORDER BY avg_se DESC;

-- Trigger to update timestamps
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

CREATE TRIGGER update_materials_updated_at BEFORE UPDATE
    ON materials FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_measurements_updated_at BEFORE UPDATE
    ON emi_measurements FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();