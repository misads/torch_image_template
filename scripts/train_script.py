import os
import utils.misc_utils as utils

times = 5

for i in range(times):
    command = "python3 train.py --tag fold_%d --batch_size 16 --gpu_ids 0"

    utils.color_print('============================================', 3)
    utils.color_print(command, 2)
    utils.color_print('============================================', 3)
    os.system(command)


