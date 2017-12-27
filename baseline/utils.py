import os
from os import path

def create_if_nonexist(outdir):
    if not path.exists(outdir):
        os.makedirs(outdir)