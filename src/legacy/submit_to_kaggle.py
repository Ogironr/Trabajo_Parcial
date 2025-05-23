#!/usr/bin/env python
import os
import sys
import glob
import argparse
import subprocess
from datetime import datetime

def find_latest_submission():
    """Find the latest submission file in the data/final directory."""
    submission_files = glob.glob(os.path.join('data', 'final', 'submission_*.csv'))
    if not submission_files:
        return None
    return max(submission_files, key=os.path.getctime)

def submit_to_kaggle(submission_file, message=None):
    """Submit the solution to Kaggle competition."""
    if not os.path.exists(submission_file):
        print(f"Error: Submission file {submission_file} not found!")
        return False

    # Si no se proporciona un mensaje, usar uno por defecto con timestamp
    if message is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        message = f"Submission from {timestamp}"

    # Comando de Kaggle
    command = [
        'kaggle', 'competitions', 'submit',
        '-c', 'insuficiencia-cardiaca-cronica-ann',
        '-f', submission_file,
        '-m', message
    ]

    try:
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode == 0:
            print("Submission successful!")
            print(result.stdout)
            return True
        else:
            print("Error during submission:")
            print(result.stderr)
            return False
    except Exception as e:
        print(f"Error executing Kaggle command: {str(e)}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Submit solution to Kaggle competition')
    parser.add_argument('--file', '-f', help='Path to submission file (if not provided, will use latest)')
    parser.add_argument('--message', '-m', help='Submission message')
    args = parser.parse_args()

    # Si no se proporciona archivo, buscar el más reciente
    submission_file = args.file
    if submission_file is None:
        submission_file = find_latest_submission()
        if submission_file is None:
            print("Error: No submission files found in data/final directory!")
            return 1
        print(f"Using latest submission file: {submission_file}")

    # Realizar la submission
    success = submit_to_kaggle(submission_file, args.message)
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())
