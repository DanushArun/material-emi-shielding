"""
Accuracy validation and measurement system for EMI predictions.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import json
import os


class AccuracyValidator:
    """Validates and tracks accuracy of EMI shielding predictions."""
    
    def __init__(self, data_path: str = "data/measurements.json"):
        """Initialize the accuracy validator."""
        self.data_path = data_path
        self.ground_truth_sources = {
            'experimental': [],      # Lab measurements
            'published': [],         # Peer-reviewed data
            'certified': [],         # NIST/ISO standards
            'industrial': [],        # Industry test data
            'user_feedback': []      # User-provided measurements
        }
        self.confidence_thresholds = {
            'high': 0.95,
            'medium': 0.85,
            'low': 0.70
        }
        self._load_existing_data()
    
    def _load_existing_data(self):
        """Load existing measurement data if available."""
        if os.path.exists(self.data_path):
            try:
                with open(self.data_path, 'r') as f:
                    data = json.load(f)
                    for source, measurements in data.items():
                        if source in self.ground_truth_sources:
                            self.ground_truth_sources[source] = measurements
            except Exception as e:
                print(f"Warning: Could not load existing data: {e}")
    
    def _save_data(self):
        """Save measurement data to disk."""
        os.makedirs(os.path.dirname(self.data_path), exist_ok=True)
        with open(self.data_path, 'w') as f:
            json.dump(self.ground_truth_sources, f, indent=2, default=str)
    
    def add_measurement(self, 
                       composition: Dict[str, float],
                       predicted_se: float,
                       measured_se: float,
                       frequency: float,
                       thickness: float,
                       source: str = 'user_feedback',
                       conditions: Optional[Dict] = None) -> str:
        """Add a new measurement to the validation database."""
        measurement_id = f"{source}_{datetime.now().timestamp()}"
        
        measurement = {
            'id': measurement_id,
            'timestamp': datetime.now().isoformat(),
            'composition': composition,
            'predicted_se': predicted_se,
            'measured_se': measured_se,
            'frequency_mhz': frequency,
            'thickness_mm': thickness,
            'error': measured_se - predicted_se,
            'relative_error': abs(measured_se - predicted_se) / measured_se * 100,
            'conditions': conditions or {}
        }
        
        self.ground_truth_sources[source].append(measurement)
        self._save_data()
        
        return measurement_id
    
    def calculate_accuracy_metrics(self, 
                                 predicted: np.ndarray, 
                                 measured: np.ndarray) -> Dict[str, float]:
        """Calculate comprehensive accuracy metrics."""
        # Ensure arrays
        predicted = np.array(predicted)
        measured = np.array(measured)
        
        # Basic metrics
        mae = mean_absolute_error(measured, predicted)
        rmse = np.sqrt(mean_squared_error(measured, predicted))
        mape = np.mean(np.abs((predicted - measured) / measured)) * 100
        r2 = r2_score(measured, predicted)
        
        # Additional metrics
        max_error = np.max(np.abs(predicted - measured))
        bias = np.mean(predicted - measured)
        
        # Confidence intervals (assuming normal distribution of errors)
        errors = predicted - measured
        std_error = np.std(errors)
        ci_95 = 1.96 * std_error
        
        return {
            'mae': mae,
            'rmse': rmse,
            'mape': mape,
            'r_squared': r2,
            'max_error': max_error,
            'bias': bias,
            'std_error': std_error,
            'ci_95': ci_95,
            'n_samples': len(predicted)
        }
    
    def check_physical_constraints(self, prediction: Dict) -> float:
        """Validate prediction against physical laws."""
        confidence = 1.0
        
        # Check if SE is positive
        if prediction['total_se'] < 0:
            confidence *= 0.1
        
        # Check if reflection + absorption >= total (approximately)
        components_sum = prediction['reflection_loss'] + prediction['absorption_loss']
        if components_sum < prediction['total_se'] * 0.8:
            confidence *= 0.8
        
        # Check skin depth is positive and reasonable
        if prediction['skin_depth'] <= 0 or prediction['skin_depth'] > 1:  # > 1m is unrealistic
            confidence *= 0.7
        
        # Check impedance is reasonable
        if prediction['intrinsic_impedance_real'] < 0:
            confidence *= 0.5
        
        return confidence
    
    def estimate_prediction_confidence(self, 
                                     composition: Dict[str, float],
                                     prediction: Dict,
                                     similar_materials_threshold: float = 0.1) -> Dict:
        """Estimate confidence in a prediction."""
        # 1. Physics-based confidence
        physics_confidence = self.check_physical_constraints(prediction)
        
        # 2. Data availability confidence
        similar_count = self._count_similar_materials(composition, similar_materials_threshold)
        data_confidence = min(1.0, similar_count / 10)  # Confidence increases with more data
        
        # 3. Interpolation vs extrapolation
        interpolation_confidence = self._check_interpolation(composition)
        
        # 4. Model uncertainty (placeholder - will be from ML model)
        model_confidence = 0.85  # Default until ML models provide uncertainty
        
        # Weighted combination
        total_confidence = (
            0.25 * physics_confidence +
            0.25 * data_confidence +
            0.25 * interpolation_confidence +
            0.25 * model_confidence
        )
        
        # Convert to uncertainty in dB
        if total_confidence > self.confidence_thresholds['high']:
            uncertainty_db = 2.0
        elif total_confidence > self.confidence_thresholds['medium']:
            uncertainty_db = 5.0
        elif total_confidence > self.confidence_thresholds['low']:
            uncertainty_db = 10.0
        else:
            uncertainty_db = 15.0
        
        return {
            'total_confidence': total_confidence,
            'physics_confidence': physics_confidence,
            'data_confidence': data_confidence,
            'interpolation_confidence': interpolation_confidence,
            'model_confidence': model_confidence,
            'uncertainty_db': uncertainty_db,
            'confidence_level': self._get_confidence_level(total_confidence),
            'needs_validation': total_confidence < self.confidence_thresholds['medium']
        }
    
    def _count_similar_materials(self, 
                                composition: Dict[str, float], 
                                threshold: float) -> int:
        """Count materials with similar composition in database."""
        count = 0
        
        for source, measurements in self.ground_truth_sources.items():
            for measurement in measurements:
                if self._calculate_composition_similarity(
                    composition, 
                    measurement.get('composition', {})
                ) > (1 - threshold):
                    count += 1
        
        return count
    
    def _calculate_composition_similarity(self, 
                                        comp1: Dict[str, float], 
                                        comp2: Dict[str, float]) -> float:
        """Calculate similarity between two compositions."""
        if not comp1 or not comp2:
            return 0.0
        
        all_elements = set(comp1.keys()) | set(comp2.keys())
        
        similarity = 0.0
        for element in all_elements:
            val1 = comp1.get(element, 0)
            val2 = comp2.get(element, 0)
            similarity += 1 - abs(val1 - val2) / max(val1 + val2, 1)
        
        return similarity / len(all_elements)
    
    def _check_interpolation(self, composition: Dict[str, float]) -> float:
        """Check if prediction is interpolation or extrapolation."""
        # Simplified check - in reality would check if composition
        # falls within convex hull of training data
        total_percentage = sum(composition.values())
        
        # Check if any element has extreme percentage
        for element, percentage in composition.items():
            if percentage > 90 or percentage < 0.1:
                return 0.7  # Likely extrapolation
        
        return 0.9  # Likely interpolation
    
    def _get_confidence_level(self, confidence: float) -> str:
        """Get confidence level description."""
        if confidence > self.confidence_thresholds['high']:
            return 'high'
        elif confidence > self.confidence_thresholds['medium']:
            return 'medium'
        elif confidence > self.confidence_thresholds['low']:
            return 'low'
        else:
            return 'very_low'
    
    def get_accuracy_report(self, material_class: Optional[str] = None) -> Dict:
        """Generate comprehensive accuracy report."""
        all_predictions = []
        all_measurements = []
        
        # Collect all prediction-measurement pairs
        for source, measurements in self.ground_truth_sources.items():
            for measurement in measurements:
                if 'predicted_se' in measurement and 'measured_se' in measurement:
                    all_predictions.append(measurement['predicted_se'])
                    all_measurements.append(measurement['measured_se'])
        
        if not all_predictions:
            return {
                'status': 'insufficient_data',
                'message': 'No validated measurements available yet'
            }
        
        # Calculate metrics
        metrics = self.calculate_accuracy_metrics(
            np.array(all_predictions),
            np.array(all_measurements)
        )
        
        # Add summary statistics
        metrics['total_measurements'] = len(all_predictions)
        metrics['sources'] = {
            source: len(measurements) 
            for source, measurements in self.ground_truth_sources.items()
            if measurements
        }
        
        return {
            'status': 'success',
            'metrics': metrics,
            'last_updated': datetime.now().isoformat()
        }
    
    def identify_systematic_errors(self) -> Dict:
        """Identify patterns in prediction errors."""
        errors_by_frequency = {}
        errors_by_thickness = {}
        errors_by_element = {}
        
        for source, measurements in self.ground_truth_sources.items():
            for measurement in measurements:
                if 'error' not in measurement:
                    continue
                
                error = measurement['error']
                
                # Group by frequency range
                freq = measurement.get('frequency_mhz', 0)
                freq_range = f"{int(freq/1000)}GHz" if freq >= 1000 else f"{int(freq/100)*100}MHz"
                if freq_range not in errors_by_frequency:
                    errors_by_frequency[freq_range] = []
                errors_by_frequency[freq_range].append(error)
                
                # Group by thickness
                thickness = measurement.get('thickness_mm', 0)
                thickness_range = f"{int(thickness)}mm"
                if thickness_range not in errors_by_thickness:
                    errors_by_thickness[thickness_range] = []
                errors_by_thickness[thickness_range].append(error)
                
                # Group by dominant element
                composition = measurement.get('composition', {})
                if composition:
                    dominant = max(composition.items(), key=lambda x: x[1])[0]
                    if dominant not in errors_by_element:
                        errors_by_element[dominant] = []
                    errors_by_element[dominant].append(error)
        
        # Calculate statistics for each group
        patterns = {
            'frequency_dependent': self._analyze_error_groups(errors_by_frequency),
            'thickness_dependent': self._analyze_error_groups(errors_by_thickness),
            'element_dependent': self._analyze_error_groups(errors_by_element)
        }
        
        return patterns
    
    def _analyze_error_groups(self, error_groups: Dict[str, List[float]]) -> Dict:
        """Analyze error patterns in grouped data."""
        analysis = {}
        
        for group, errors in error_groups.items():
            if len(errors) >= 3:  # Need at least 3 samples
                analysis[group] = {
                    'mean_error': np.mean(errors),
                    'std_error': np.std(errors),
                    'systematic_bias': abs(np.mean(errors)) > np.std(errors),
                    'n_samples': len(errors)
                }
        
        return analysis