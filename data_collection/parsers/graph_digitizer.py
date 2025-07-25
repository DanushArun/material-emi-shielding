"""
Graph digitizer for extracting EMI shielding data from plots and figures.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
from typing import List, Tuple, Dict, Optional, Any
import logging
from pathlib import Path
import json
from dataclasses import dataclass
import pytesseract
from PIL import Image


@dataclass
class PlotData:
    """Structure for extracted plot data."""
    x_values: np.ndarray  # Frequency values
    y_values: np.ndarray  # SE values
    x_label: str
    y_label: str
    x_unit: str
    y_unit: str
    material_name: Optional[str] = None
    plot_type: str = "line"  # line, scatter, bar
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'x_values': self.x_values.tolist(),
            'y_values': self.y_values.tolist(),
            'x_label': self.x_label,
            'y_label': self.y_label,
            'x_unit': self.x_unit,
            'y_unit': self.y_unit,
            'material_name': self.material_name,
            'plot_type': self.plot_type
        }


class EMIGraphDigitizer:
    """Extract numerical data from EMI shielding plots."""
    
    # Common axis labels for EMI plots
    X_AXIS_PATTERNS = {
        'frequency': ['frequency', 'freq', 'f ', 'ghz', 'mhz', 'hz'],
        'thickness': ['thickness', 't ', 'mm', 'cm', 'μm']
    }
    
    Y_AXIS_PATTERNS = {
        'shielding_effectiveness': ['shielding', 'se ', 'emi se', 'db'],
        'reflection_loss': ['reflection', 'rl ', 'r '],
        'absorption_loss': ['absorption', 'al ', 'a ']
    }
    
    def __init__(self):
        """Initialize the graph digitizer."""
        self.logger = logging.getLogger(__name__)
        
    def extract_from_image(self, image_path: Path) -> List[PlotData]:
        """
        Extract data from a plot image.
        
        Args:
            image_path: Path to image file
            
        Returns:
            List of PlotData objects
        """
        # Load image
        image = cv2.imread(str(image_path))
        if image is None:
            self.logger.error(f"Failed to load image: {image_path}")
            return []
        
        # Try to extract text (labels, legend)
        text_data = self._extract_text_ocr(image)
        
        # Detect plot area
        plot_region = self._detect_plot_region(image)
        if plot_region is None:
            self.logger.warning("Could not detect plot region")
            return []
        
        # Extract axis information
        axis_info = self._extract_axis_info(image, plot_region, text_data)
        
        # Detect data lines/points
        data_lines = self._detect_data_lines(image, plot_region)
        
        # Digitize each line
        plot_data_list = []
        for i, line_pixels in enumerate(data_lines):
            if len(line_pixels) > 0:
                plot_data = self._digitize_line(
                    line_pixels, 
                    plot_region, 
                    axis_info,
                    material_name=f"Material_{i+1}"
                )
                if plot_data:
                    plot_data_list.append(plot_data)
        
        return plot_data_list
    
    def _extract_text_ocr(self, image: np.ndarray) -> Dict[str, List[Tuple[str, Tuple[int, int]]]]:
        """Extract text from image using OCR."""
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Use pytesseract to extract text with positions
        try:
            data = pytesseract.image_to_data(gray, output_type=pytesseract.Output.DICT)
            
            text_data = {
                'all_text': [],
                'x_labels': [],
                'y_labels': [],
                'legend': []
            }
            
            for i in range(len(data['text'])):
                if data['text'][i].strip():
                    text = data['text'][i].lower()
                    position = (data['left'][i], data['top'][i])
                    text_data['all_text'].append((text, position))
                    
                    # Classify text by position
                    if position[1] > image.shape[0] * 0.8:  # Bottom area
                        text_data['x_labels'].append((text, position))
                    elif position[0] < image.shape[1] * 0.2:  # Left area
                        text_data['y_labels'].append((text, position))
                        
            return text_data
            
        except Exception as e:
            self.logger.warning(f"OCR failed: {e}")
            return {'all_text': [], 'x_labels': [], 'y_labels': [], 'legend': []}
    
    def _detect_plot_region(self, image: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """Detect the main plot area (axes box)."""
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Edge detection
        edges = cv2.Canny(gray, 50, 150)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Find the largest rectangle (likely the plot area)
        largest_rect = None
        max_area = 0
        
        for contour in contours:
            # Approximate contour to polygon
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            # Check if it's a rectangle
            if len(approx) == 4:
                x, y, w, h = cv2.boundingRect(approx)
                area = w * h
                
                # Filter by size and aspect ratio
                if area > max_area and area > 0.1 * image.shape[0] * image.shape[1]:
                    if 0.3 < w/h < 3:  # Reasonable aspect ratio
                        max_area = area
                        largest_rect = (x, y, x+w, y+h)
        
        return largest_rect
    
    def _extract_axis_info(self, image: np.ndarray, plot_region: Tuple[int, int, int, int],
                          text_data: Dict) -> Dict[str, Any]:
        """Extract axis labels and ranges."""
        x1, y1, x2, y2 = plot_region
        
        axis_info = {
            'x_label': 'Frequency',
            'y_label': 'Shielding Effectiveness',
            'x_unit': 'GHz',
            'y_unit': 'dB',
            'x_range': [0, 10],  # Default ranges
            'y_range': [0, 100],
            'x_scale': 'linear',
            'y_scale': 'linear'
        }
        
        # Try to identify axis labels from OCR text
        for text, pos in text_data['x_labels']:
            for pattern_type, patterns in self.X_AXIS_PATTERNS.items():
                if any(p in text for p in patterns):
                    axis_info['x_label'] = pattern_type.title()
                    if 'ghz' in text:
                        axis_info['x_unit'] = 'GHz'
                    elif 'mhz' in text:
                        axis_info['x_unit'] = 'MHz'
                    break
        
        for text, pos in text_data['y_labels']:
            for pattern_type, patterns in self.Y_AXIS_PATTERNS.items():
                if any(p in text for p in patterns):
                    axis_info['y_label'] = pattern_type.replace('_', ' ').title()
                    if 'db' in text:
                        axis_info['y_unit'] = 'dB'
                    break
        
        # Try to extract axis ranges from tick labels
        axis_info['x_range'] = self._extract_axis_range(image, plot_region, 'x', text_data)
        axis_info['y_range'] = self._extract_axis_range(image, plot_region, 'y', text_data)
        
        # Detect if logarithmic scale
        if self._is_log_scale(axis_info['x_range']):
            axis_info['x_scale'] = 'log'
        if self._is_log_scale(axis_info['y_range']):
            axis_info['y_scale'] = 'log'
        
        return axis_info
    
    def _extract_axis_range(self, image: np.ndarray, plot_region: Tuple[int, int, int, int],
                           axis: str, text_data: Dict) -> List[float]:
        """Extract numerical range for an axis."""
        x1, y1, x2, y2 = plot_region
        
        numbers = []
        
        # Look for numbers near the axis
        for text, pos in text_data['all_text']:
            try:
                # Try to parse as number
                num = float(text.replace(',', ''))
                
                # Check if near the correct axis
                if axis == 'x' and y2 - 50 < pos[1] < y2 + 100:
                    if x1 - 50 < pos[0] < x2 + 50:
                        numbers.append((num, pos[0]))
                elif axis == 'y' and x1 - 100 < pos[0] < x1 + 50:
                    if y1 - 50 < pos[1] < y2 + 50:
                        numbers.append((num, pos[1]))
                        
            except ValueError:
                continue
        
        if len(numbers) >= 2:
            # Sort by position and extract min/max
            numbers.sort(key=lambda x: x[1])
            return [numbers[0][0], numbers[-1][0]]
        else:
            # Default ranges
            return [0, 10] if axis == 'x' else [0, 100]
    
    def _is_log_scale(self, axis_range: List[float]) -> bool:
        """Check if axis uses logarithmic scale."""
        if axis_range[1] / axis_range[0] > 100:
            return True
        return False
    
    def _detect_data_lines(self, image: np.ndarray, 
                          plot_region: Tuple[int, int, int, int]) -> List[List[Tuple[int, int]]]:
        """Detect data lines or points in the plot."""
        x1, y1, x2, y2 = plot_region
        
        # Crop to plot region
        plot_image = image[y1:y2, x1:x2]
        
        # Convert to HSV for better color detection
        hsv = cv2.cvtColor(plot_image, cv2.COLOR_BGR2HSV)
        
        # Detect different colored lines
        lines = []
        
        # Common plot colors (in HSV ranges)
        color_ranges = [
            # Blue
            ((100, 50, 50), (130, 255, 255)),
            # Red
            ((0, 50, 50), (10, 255, 255)),
            ((170, 50, 50), (180, 255, 255)),
            # Green
            ((40, 50, 50), (80, 255, 255)),
            # Black/Gray (for black lines)
            ((0, 0, 0), (180, 30, 100)),
        ]
        
        for lower, upper in color_ranges:
            # Create mask for color
            mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
            
            # Remove noise
            kernel = np.ones((3, 3), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            
            # Find connected components
            num_labels, labels = cv2.connectedComponents(mask)
            
            # Extract pixels for each component
            for label in range(1, num_labels):
                component_mask = (labels == label).astype(np.uint8) * 255
                
                # Get pixel coordinates
                y_coords, x_coords = np.where(component_mask > 0)
                
                if len(x_coords) > 10:  # Minimum points for a line
                    line_pixels = list(zip(x_coords, y_coords))
                    lines.append(line_pixels)
        
        return lines
    
    def _digitize_line(self, line_pixels: List[Tuple[int, int]], 
                      plot_region: Tuple[int, int, int, int],
                      axis_info: Dict[str, Any],
                      material_name: str = None) -> Optional[PlotData]:
        """Convert pixel coordinates to data values."""
        if not line_pixels:
            return None
        
        x1, y1, x2, y2 = plot_region
        plot_width = x2 - x1
        plot_height = y2 - y1
        
        # Sort pixels by x coordinate
        line_pixels.sort(key=lambda p: p[0])
        
        # Convert pixel coordinates to data values
        x_values = []
        y_values = []
        
        for px, py in line_pixels:
            # Normalize to [0, 1]
            x_norm = px / plot_width
            y_norm = 1 - (py / plot_height)  # Invert y-axis
            
            # Convert to data values
            if axis_info['x_scale'] == 'log':
                x_val = 10 ** (np.log10(axis_info['x_range'][0]) + 
                              x_norm * (np.log10(axis_info['x_range'][1]) - 
                                      np.log10(axis_info['x_range'][0])))
            else:
                x_val = axis_info['x_range'][0] + x_norm * (axis_info['x_range'][1] - 
                                                           axis_info['x_range'][0])
            
            if axis_info['y_scale'] == 'log':
                y_val = 10 ** (np.log10(axis_info['y_range'][0]) + 
                              y_norm * (np.log10(axis_info['y_range'][1]) - 
                                      np.log10(axis_info['y_range'][0])))
            else:
                y_val = axis_info['y_range'][0] + y_norm * (axis_info['y_range'][1] - 
                                                           axis_info['y_range'][0])
            
            x_values.append(x_val)
            y_values.append(y_val)
        
        # Remove duplicates and sort
        unique_points = {}
        for x, y in zip(x_values, y_values):
            if x not in unique_points or y > unique_points[x]:
                unique_points[x] = y
        
        x_values = sorted(unique_points.keys())
        y_values = [unique_points[x] for x in x_values]
        
        # Smooth the data using spline interpolation
        if len(x_values) > 3:
            try:
                # Create interpolation function
                f = interpolate.UnivariateSpline(x_values, y_values, s=0.5)
                
                # Generate smooth curve
                x_smooth = np.linspace(min(x_values), max(x_values), 100)
                y_smooth = f(x_smooth)
                
                x_values = x_smooth
                y_values = y_smooth
            except:
                # If interpolation fails, use original values
                pass
        
        return PlotData(
            x_values=np.array(x_values),
            y_values=np.array(y_values),
            x_label=axis_info['x_label'],
            y_label=axis_info['y_label'],
            x_unit=axis_info['x_unit'],
            y_unit=axis_info['y_unit'],
            material_name=material_name
        )
    
    def extract_from_pdf_figures(self, pdf_path: Path, output_dir: Path) -> List[PlotData]:
        """Extract all figures from PDF and digitize them."""
        import fitz  # PyMuPDF
        
        output_dir.mkdir(exist_ok=True)
        all_plot_data = []
        
        # Open PDF
        pdf_document = fitz.open(pdf_path)
        
        for page_num in range(len(pdf_document)):
            page = pdf_document[page_num]
            
            # Get images from page
            image_list = page.get_images()
            
            for img_index, img in enumerate(image_list):
                # Extract image
                xref = img[0]
                pix = fitz.Pixmap(pdf_document, xref)
                
                if pix.n - pix.alpha < 4:  # GRAY or RGB
                    # Save image
                    img_path = output_dir / f"page{page_num}_img{img_index}.png"
                    pix.save(str(img_path))
                    
                    # Try to digitize
                    plot_data = self.extract_from_image(img_path)
                    all_plot_data.extend(plot_data)
                
                pix = None
        
        pdf_document.close()
        
        return all_plot_data