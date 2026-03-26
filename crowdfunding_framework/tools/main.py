import argparse
import sys
import os

# Ensure project root is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from crowdfunding_framework.tools.extraction_flow import ExtractionFlow

def main():
    parser = argparse.ArgumentParser(description="Crowdfunding Tools CLI")
    
    # Subcommands if we add more tools later, but for now 'extract' is the main tool.
    # Let's use subparser 'extract' to be explicit, like 'tools/main.py extract ...'
    # This allows adding 'tools/main.py visualize ...' later.
    subparsers = parser.add_subparsers(dest='command', help='Tool command', required=True)
    
    ext_parser = subparsers.add_parser('extract', help='Extract Active/Upcoming projects')
    ext_parser.add_argument('--date', type=str, required=True, help='Reference Date (YYYY-MM-DD)')
    ext_parser.add_argument('--weeks', type=int, default=8, help='Horizon for upcoming projects')
    ext_parser.add_argument('--output', type=str, default='.', help='Output directory')

    args = parser.parse_args()

    if args.command == 'extract':
        flow = ExtractionFlow()
        flow.run(args)

if __name__ == "__main__":
    main()
