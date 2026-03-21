"""Shared test configuration and fixtures."""
import sys
import os

# Ensure project root is on the path for all tests
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'backend'))
