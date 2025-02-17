# Quick script to test whether cnf is properly loaded with all correct python version and all that 
# Run this on your login node with CoNFiLD environment active
import sys
print("\n".join(sys.path))
try:
    import cnf
    print("CNF found at:", cnf.__file__)
except ImportError as e:
    print("Import failed:", e)
