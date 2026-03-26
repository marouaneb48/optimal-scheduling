import argparse
import sys
import os

# Ensure project root is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from crowdfunding_framework.optimization.optimization_flow import OptimizationFlow

def main():
    parser = argparse.ArgumentParser(description="Crowdfunding Optimization CLI")
    
    # We can either use subparsers or just arguments if it's the only command.
    # User might like 'python .../main.py optimize ...' or just 'python .../main.py --args'
    # Keeping subparser 'optimize' for consistency or strictly defaulting?
    # Let's keep strict arguments to avoid 'optimize' repetition: `python optimization/main.py --weeks ...`
    # BUT the previous workflow was `main.py optimize ...`. 
    # Let's make it simpler: `python optimization/main.py ...` (implicitly runs optimize)
    # But for extensibility, let's keep 'optimize' command? No, separation means specialization.
    # I'll enable running WITHOUT a subcommand for simplicity, akin to "Run Optimization".
    
    parser.add_argument('--population', type=int, default=10, help='GA Population Size')
    parser.add_argument('--generations', type=int, default=30, help='GA Generations')
    parser.add_argument('--projects', type=str, help='Path to upcoming projects CSV (Real IO)')
    parser.add_argument('--context', type=str, help='Path to currently active projects CSV (Context)')
    parser.add_argument('--weeks', type=int, default=8, help='Optimization Horizon (Weeks)')

    args = parser.parse_args()

    flow = OptimizationFlow()
    flow.run(args)

if __name__ == "__main__":
    main()
