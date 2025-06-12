"""
Unit test package for iso2mesh.
"""

import os
import sys

# Add the parent directory to sys.path so iso2mesh module can be imported
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
