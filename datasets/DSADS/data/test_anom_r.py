pth = r'/workspaces/FTSAD/datasets/DSADS/data'
import os

for root , subdir, file in os.walk(pth):
    print(root,subdir,file)