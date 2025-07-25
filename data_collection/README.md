# EMI Shielding Data Collection System

A comprehensive web scraping and data collection framework for gathering EMI shielding effectiveness measurements from scientific literature, patents, and materials databases.

## Features

- **Multi-source data collection**: Scientific papers, patents, materials databases
- **Advanced extraction methods**: 
  - PDF table extraction
  - Graph/plot digitization
  - Natural language text mining
- **Materials database integration**: Materials Project API support
- **PostgreSQL database**: Structured storage with comprehensive schema
- **Rate limiting & error handling**: Respectful scraping with retry logic
- **Data validation**: Automatic unit conversion and validation

## Project Structure

```
data_collection/
├── scrapers/           # Web scraping modules
│   ├── base_scraper.py        # Base class with common functionality
│   ├── materials_project_api.py # Materials Project integration
│   ├── ieee_scraper.py        # IEEE Xplore (to implement)
│   └── sciencedirect_scraper.py # ScienceDirect (to implement)
├── parsers/            # Data extraction modules
│   ├── table_extractor.py     # Extract data from PDF tables
│   ├── graph_digitizer.py     # Extract data from plots
│   └── text_miner.py          # Extract data from text
├── database/           # Database management
│   ├── schema.sql             # PostgreSQL schema
│   └── connection.py          # Database connection manager
├── utils/              # Utility functions
├── tests/              # Unit tests
└── output/             # Scraped data output
```

## Installation

1. **Clone the repository**
```bash
cd /path/to/EMI-shielding
```

2. **Install dependencies**
```bash
pip install -r data_collection/requirements.txt
```

3. **Set up PostgreSQL database**
```bash
# Create database
createdb emi_shielding

# Run schema
psql -d emi_shielding -f data_collection/database/schema.sql
```

4. **Configure environment variables**
```bash
cp .env.example .env
# Edit .env with your database credentials and API keys
```

## Usage

### 1. PDF Table Extraction

Extract EMI shielding data from PDF tables:

```python
from data_collection.parsers.table_extractor import EMITableExtractor

extractor = EMITableExtractor()
emi_data = extractor.extract_from_pdf("path/to/paper.pdf")

for data in emi_data:
    print(f"{data.material}: {data.shielding_effectiveness} dB at {data.frequency} Hz")
```

### 2. Graph Digitization

Extract data from plots and figures:

```python
from data_collection.parsers.graph_digitizer import EMIGraphDigitizer

digitizer = EMIGraphDigitizer()
plot_data = digitizer.extract_from_image("path/to/plot.png")

for data in plot_data:
    print(f"Extracted {len(data.x_values)} data points")
```

### 3. Materials Project API

Fetch material properties:

```python
from data_collection.scrapers.materials_project_api import MaterialsProjectScraper

mp = MaterialsProjectScraper()  # Requires MP_API_KEY env variable

# Search for conductive materials
conductive = mp.fetch_conductive_materials(min_conductivity=1e6)

# Search for magnetic materials
magnetic = mp.fetch_magnetic_materials()

# Estimate EMI performance
performance = mp.estimate_emi_performance(material_data, thickness=1.0, frequency=1e9)
```

### 4. Database Operations

Store and query EMI data:

```python
from data_collection.database.connection import DatabaseManager

db = DatabaseManager()

# Insert material
material_id = db.insert_material({
    'name': 'Fe70Co30/PVDF',
    'composition': {'Fe': 70, 'Co': 30},
    'synthesis_method': 'Ball milling'
})

# Insert measurement
measurement_id = db.insert_measurement({
    'material_id': material_id,
    'thickness': 2.0,
    'frequency': 1e9,
    'total_se': 65.5
})

# Search materials
results = db.search_materials(min_se=60, frequency_range=(1e9, 10e9))
```

## Data Schema

### Materials Table
- `name`: Material name
- `composition`: Elemental composition (JSONB)
- `synthesis_method`: How the material was made
- `particle_size`: Size of particles/fillers
- `morphology`: Physical structure

### EMI Measurements Table
- `material_id`: Reference to material
- `conductivity`: Electrical conductivity (S/m)
- `permeability`: Relative magnetic permeability
- `thickness`: Sample thickness (mm)
- `frequency`: Test frequency (Hz)
- `total_se`: Total shielding effectiveness (dB)
- `reflection_loss`: Reflection component (dB)
- `absorption_loss`: Absorption component (dB)

## Expected Data Sources

1. **Scientific Literature**
   - IEEE Xplore (10,000+ papers)
   - ScienceDirect (5,000+ papers)
   - ACS Publications
   - Nature Materials
   - Journal of Applied Physics

2. **Patent Databases**
   - USPTO
   - European Patent Office
   - WIPO

3. **Materials Databases**
   - Materials Project (130,000+ materials)
   - AFLOW
   - OQMD
   - MatWeb

## Data Collection Pipeline

1. **Search Phase**: Query databases for EMI-related papers
2. **Download Phase**: Fetch PDFs and web pages
3. **Extraction Phase**: Extract tables, graphs, and text data
4. **Validation Phase**: Validate units, ranges, and consistency
5. **Storage Phase**: Store in PostgreSQL database
6. **Analysis Phase**: Generate statistics and insights

## Example: Run Complete Demo

```bash
python data_collection/example_usage.py
```

This will demonstrate:
- PDF table extraction
- Graph digitization
- Materials Project API
- Web scraping
- Database operations

## Configuration

### Environment Variables
```bash
# Database
DB_HOST=localhost
DB_PORT=5432
DB_NAME=emi_shielding
DB_USER=postgres
DB_PASSWORD=your_password

# API Keys
MP_API_KEY=your_materials_project_api_key

# Scraping
RATE_LIMIT_CALLS=10
RATE_LIMIT_PERIOD=60
```

### Rate Limiting
Default: 10 requests per 60 seconds. Adjust in base_scraper.py:
```python
@limits(calls=10, period=60)
```

## Next Steps

1. **Implement specific scrapers**:
   - IEEE Xplore scraper
   - ScienceDirect scraper
   - Patent database scrapers

2. **Enhance extraction**:
   - Improve OCR accuracy
   - Handle complex table formats
   - Multi-column PDF parsing

3. **Add ML features**:
   - Train models on collected data
   - Predict missing properties
   - Identify promising materials

4. **Scale up**:
   - Distributed scraping
   - Cloud deployment
   - Real-time updates

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add your scraper/parser
4. Write tests
5. Submit pull request

## License

This is part of the EMI Shielding research project. See main project license.