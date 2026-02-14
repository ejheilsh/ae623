# learning notes for Max 

from pathlib import Path

# relative to script location
HERE = Path(__file__).resolve().parent
path = HERE / "data" / "input.txt"

# check file exists
if path.exists():
    print("File found")
    
# get filename without extension
path.stem        # "input"
path.suffix      # ".txt"


# RULE OF THUMB
# Use pathlib for laths, use open() or path.open() for opening 

with open(path) as f:
    #####
    pass


