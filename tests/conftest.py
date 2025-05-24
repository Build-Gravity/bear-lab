import sys
import os

# Add the project root directory to the Python path
# This allows pytest to find the 'src' module
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root) 