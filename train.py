import os
import time

import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision.transforms import transforms

from dataloader import dual_residual_dataset
from network.Model import Model
from options import opt
from utils.torch_utils import create_summary_writer, write_image, write_meters_loss
import utils.misc_utils as utils

data_name = 'RESIDE'
data_root = './datasets/' + data_name + '/ITS/'
imlist_pth = './datasets/' + data_name + '/indoor_train_list.txt'


# dstroot for saving models.
# logroot for writting some log(s), if is needed.
save_root = os.path.join(opt.checkpoints_dir, opt.tag)
log_root = os.path.join(opt.log_dir, opt.tag)


utils.try_make_dir(save_root)
utils.try_make_dir(log_root)

if opt.debug:
    opt.save_freq = 1
    opt.eval_freq = 1
    opt.log_freq = 1

# Transform
transform = transforms.ToTensor()
# Dataloader
max_size = 9999999
if opt.debug:
    max_size = opt.batch_size * 10

convertor = dual_residual_dataset.ImageSet(data_root, imlist_pth,
                                            transform=transform, is_train=True,
                                            with_aug=False, crop_size=opt.crop, max_size=max_size)
dataloader = DataLoader(convertor, batch_size=opt.batch_size, shuffle=False, num_workers=5)



model = Model(opt)

# if len(opt.gpu_ids):
#     model = torch.nn.DataParallel(model, device_ids=opt.gpu_ids)
model = model.cuda()

#optimizer_G, optimizer_D = model.module.optimizer_G, model.module.optimizer_D
# optimizer_G = model.g_optimizer

start_epoch = opt.which_epoch if opt.which_epoch else 0
model.train()

# Start training
print('Start training...')
start_step = start_epoch * len(dataloader)
global_step = start_step
total_steps = opt.epochs * len(dataloader)
start = time.time()

writer = create_summary_writer(log_root)

for epoch in range(start_epoch, opt.epochs):
    for iteration, data in enumerate(dataloader):
        global_step += 1
        rate = (global_step - start_step) / (time.time() - start)
        remaining = (total_steps - global_step) / rate

        img, label, _ = data
        img_var = Variable(img, requires_grad=False).cuda()
        label_var = Variable(label, requires_grad=False).cuda()

        # Cleaning noisy images
        cleaned = model.cleaner(img_var)

        if epoch % opt.log_freq == opt.log_freq - 1 and iteration < 5:
            write_image(writer, 'train/%d' % iteration, 'input', img.data[0], epoch)
            write_image(writer, 'train/%d' % iteration, 'output', cleaned.data[0], epoch)
            write_image(writer, 'train/%d' % iteration, 'target', label_var.data[0], epoch)

        # update
        model.update_G(cleaned, label_var)

        pre_msg = 'Epoch:%d' % epoch

        msg = '(loss) %s ETA: %s' % (str(model.avg_meters), utils.format_time(remaining))
        utils.progress_bar(iteration, len(dataloader), pre_msg, msg)
        # print(pre_msg, msg)

        # print('Epoch(' + str(epoch + 1) + '), iteration(' + str(iteration + 1) + '): ' +'%.4f, %.4f' % (-ssim_loss.item(),
        #                                                                                                l1_loss.item()))

    #write_loss(writer, 'train', 'F1', 0.78, epoch)
    write_meters_loss(writer, 'train', model.avg_meters, epoch)

    if epoch % opt.save_freq == opt.save_freq - 1:  # 每隔10次save checkpoint
        model.save(epoch)

    if epoch % opt.eval_freq == (opt.eval_freq - 1):
        model.eval()
        # eva(cleaner, test_dataloader, epoch+1, writer)
        # eva(cleaner, real_dataloader, epoch + 1, writer, 'SINGLE')
        model.train()

    if epoch in [700, 1400]:
        for param_group in model.g_optimizer.param_groups:
            param_group['lr'] *= 0.1

