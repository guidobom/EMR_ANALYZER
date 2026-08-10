#!/Users/massimoguidoboni/miniconda3/envs/emr-analyzer/bin/python
"""EMR Analyzer — Entry point for the clinical document analysis system.

Usage:
    python run.py
"""

import sys
import os

# Ensure the project root is on the Python path
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from emr_analyzer.app import EMRAnalyzerApp


def main():
    app = EMRAnalyzerApp()
    sys.exit(app.run())


if __name__ == "__main__":
    main()
