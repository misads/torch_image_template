import os
tags = ['fold_1', 'fold_2']


for tag in tags:
    cmd = "python3 test.py --load %s" % tag
    print(cmd)
    os.system(cmd)
