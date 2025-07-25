"""
Database connection and management utilities.
"""

import os
import logging
from typing import Dict, List, Optional, Any
from contextlib import contextmanager
import psycopg2
from psycopg2.extras import RealDictCursor, Json
from psycopg2.pool import SimpleConnectionPool
import pandas as pd
from dotenv import load_dotenv


load_dotenv()


class DatabaseManager:
    """Manage PostgreSQL database connections and operations."""
    
    def __init__(self, 
                 host: str = None,
                 port: int = None,
                 database: str = None,
                 user: str = None,
                 password: str = None,
                 min_connections: int = 1,
                 max_connections: int = 10):
        """
        Initialize database connection pool.
        
        Args:
            host: Database host
            port: Database port
            database: Database name
            user: Database user
            password: Database password
            min_connections: Minimum pool connections
            max_connections: Maximum pool connections
        """
        self.logger = logging.getLogger(__name__)
        
        # Get connection parameters from environment or arguments
        self.connection_params = {
            'host': host or os.getenv('DB_HOST', 'localhost'),
            'port': port or int(os.getenv('DB_PORT', '5432')),
            'database': database or os.getenv('DB_NAME', 'emi_shielding'),
            'user': user or os.getenv('DB_USER', 'postgres'),
            'password': password or os.getenv('DB_PASSWORD', '')
        }
        
        # Create connection pool
        try:
            self.pool = SimpleConnectionPool(
                min_connections,
                max_connections,
                **self.connection_params
            )
            self.logger.info("Database connection pool created successfully")
        except Exception as e:
            self.logger.error(f"Failed to create connection pool: {e}")
            raise
    
    @contextmanager
    def get_connection(self):
        """Get a connection from the pool."""
        connection = self.pool.getconn()
        try:
            yield connection
            connection.commit()
        except Exception as e:
            connection.rollback()
            self.logger.error(f"Database error: {e}")
            raise
        finally:
            self.pool.putconn(connection)
    
    @contextmanager
    def get_cursor(self, cursor_factory=RealDictCursor):
        """Get a cursor with automatic connection management."""
        with self.get_connection() as conn:
            cursor = conn.cursor(cursor_factory=cursor_factory)
            try:
                yield cursor
            finally:
                cursor.close()
    
    def execute_query(self, query: str, params: tuple = None) -> List[Dict]:
        """Execute a SELECT query and return results."""
        with self.get_cursor() as cursor:
            cursor.execute(query, params)
            return cursor.fetchall()
    
    def execute_insert(self, query: str, params: tuple = None) -> Optional[str]:
        """Execute an INSERT query and return the ID."""
        with self.get_cursor() as cursor:
            if 'RETURNING' not in query.upper():
                query += ' RETURNING id'
            cursor.execute(query, params)
            result = cursor.fetchone()
            return result['id'] if result else None
    
    def execute_many(self, query: str, params_list: List[tuple]) -> int:
        """Execute multiple queries efficiently."""
        with self.get_cursor() as cursor:
            cursor.executemany(query, params_list)
            return cursor.rowcount
    
    def insert_material(self, material_data: Dict) -> Optional[str]:
        """Insert a new material into the database."""
        query = """
            INSERT INTO materials (
                name, material_class, composition, synthesis_method,
                processing_temperature, processing_time, processing_atmosphere,
                particle_size, morphology
            ) VALUES (
                %(name)s, %(material_class)s, %(composition)s, %(synthesis_method)s,
                %(processing_temperature)s, %(processing_time)s, %(processing_atmosphere)s,
                %(particle_size)s, %(morphology)s
            )
            ON CONFLICT (name, composition) DO UPDATE SET
                updated_at = CURRENT_TIMESTAMP
            RETURNING id
        """
        
        # Convert composition dict to JSON
        if 'composition' in material_data and isinstance(material_data['composition'], dict):
            material_data['composition'] = Json(material_data['composition'])
        
        return self.execute_insert(query, material_data)
    
    def insert_measurement(self, measurement_data: Dict) -> Optional[str]:
        """Insert EMI measurement data."""
        query = """
            INSERT INTO emi_measurements (
                material_id, conductivity, relative_permeability, relative_permittivity,
                thickness, frequency, total_se, reflection_loss, absorption_loss,
                filler_loading, filler_type, measurement_standard, source_doi,
                source_title, source_year, confidence_score
            ) VALUES (
                %(material_id)s, %(conductivity)s, %(relative_permeability)s, 
                %(relative_permittivity)s, %(thickness)s, %(frequency)s, %(total_se)s,
                %(reflection_loss)s, %(absorption_loss)s, %(filler_loading)s,
                %(filler_type)s, %(measurement_standard)s, %(source_doi)s,
                %(source_title)s, %(source_year)s, %(confidence_score)s
            )
        """
        
        return self.execute_insert(query, measurement_data)
    
    def insert_frequency_sweep(self, measurement_id: str, sweep_data: Dict) -> Optional[str]:
        """Insert frequency sweep data."""
        query = """
            INSERT INTO frequency_sweeps (
                measurement_id, frequencies, shielding_effectiveness,
                reflection_loss, absorption_loss
            ) VALUES (
                %(measurement_id)s, %(frequencies)s, %(shielding_effectiveness)s,
                %(reflection_loss)s, %(absorption_loss)s
            )
        """
        
        sweep_data['measurement_id'] = measurement_id
        return self.execute_insert(query, sweep_data)
    
    def get_material_by_name(self, name: str) -> Optional[Dict]:
        """Get material by name."""
        query = "SELECT * FROM materials WHERE name = %s"
        results = self.execute_query(query, (name,))
        return results[0] if results else None
    
    def get_measurements_for_material(self, material_id: str) -> List[Dict]:
        """Get all measurements for a material."""
        query = """
            SELECT * FROM emi_measurements 
            WHERE material_id = %s
            ORDER BY frequency
        """
        return self.execute_query(query, (material_id,))
    
    def search_materials(self, 
                        min_se: float = None,
                        max_se: float = None,
                        frequency_range: tuple = None,
                        material_class: str = None) -> pd.DataFrame:
        """Search materials based on criteria."""
        query = """
            SELECT DISTINCT
                m.name,
                m.composition,
                m.material_class,
                AVG(em.total_se) as avg_se,
                MIN(em.total_se) as min_se,
                MAX(em.total_se) as max_se,
                COUNT(*) as measurement_count
            FROM materials m
            JOIN emi_measurements em ON m.id = em.material_id
            WHERE 1=1
        """
        
        params = []
        
        if min_se is not None:
            query += " AND em.total_se >= %s"
            params.append(min_se)
            
        if max_se is not None:
            query += " AND em.total_se <= %s"
            params.append(max_se)
            
        if frequency_range:
            query += " AND em.frequency BETWEEN %s AND %s"
            params.extend(frequency_range)
            
        if material_class:
            query += " AND m.material_class = %s"
            params.append(material_class)
        
        query += """
            GROUP BY m.id, m.name, m.composition, m.material_class
            ORDER BY avg_se DESC
        """
        
        with self.get_cursor() as cursor:
            cursor.execute(query, params)
            return pd.DataFrame(cursor.fetchall())
    
    def insert_scraped_data(self, scraped_data: Dict) -> Optional[str]:
        """Insert raw scraped data for later processing."""
        query = """
            INSERT INTO scraped_data_staging (
                raw_data, source_url, scraper_name, extraction_method
            ) VALUES (
                %(raw_data)s, %(source_url)s, %(scraper_name)s, %(extraction_method)s
            )
        """
        
        if 'raw_data' in scraped_data and isinstance(scraped_data['raw_data'], dict):
            scraped_data['raw_data'] = Json(scraped_data['raw_data'])
        
        return self.execute_insert(query, scraped_data)
    
    def get_unprocessed_scraped_data(self, limit: int = 100) -> List[Dict]:
        """Get unprocessed scraped data for validation."""
        query = """
            SELECT * FROM scraped_data_staging
            WHERE is_processed = FALSE
            ORDER BY created_at
            LIMIT %s
        """
        return self.execute_query(query, (limit,))
    
    def mark_scraped_data_processed(self, data_id: str, success: bool = True, 
                                   error_message: str = None):
        """Mark scraped data as processed."""
        query = """
            UPDATE scraped_data_staging
            SET is_processed = TRUE,
                processing_errors = %s
            WHERE id = %s
        """
        
        with self.get_cursor() as cursor:
            cursor.execute(query, (error_message, data_id))
    
    def get_statistics(self) -> Dict:
        """Get database statistics."""
        stats = {}
        
        queries = {
            'total_materials': "SELECT COUNT(*) as count FROM materials",
            'total_measurements': "SELECT COUNT(*) as count FROM emi_measurements",
            'total_sources': "SELECT COUNT(*) as count FROM sources",
            'validated_measurements': "SELECT COUNT(*) as count FROM emi_measurements WHERE is_validated = TRUE",
            'frequency_range': """
                SELECT MIN(frequency) as min_freq, MAX(frequency) as max_freq 
                FROM emi_measurements
            """,
            'se_range': """
                SELECT MIN(total_se) as min_se, MAX(total_se) as max_se,
                       AVG(total_se) as avg_se
                FROM emi_measurements
            """
        }
        
        for key, query in queries.items():
            result = self.execute_query(query)
            if result:
                stats[key] = result[0]
        
        return stats
    
    def close(self):
        """Close all connections in the pool."""
        if hasattr(self, 'pool'):
            self.pool.closeall()
            self.logger.info("Database connection pool closed")