import re
import matplotlib.pyplot as plt
import os.path as osp

# flag: 0 -> only one slurm-xx.out file
# flag: 1 -> two slurm-xx.out files (train for twice / extra epoch needed)
flag = 1

fullpath = osp.abspath('./slurm-2133611.out')
filedir, filename = osp.split(fullpath)

print(fullpath)

# plot loss curves for training set
loss, ce_loss, tcr_01_loss, tcr_00_loss, tcr_11_loss, iters = [], [], [], [], [], []

with open(fullpath, 'rb') as f:
    while True:
        line = f.readline()
        line = line.decode('utf-8')
        if line == 'Done\n':
            break
        if not line.startswith('Epoch: ['):
            continue

        # print(line)

        # Example: Epoch: [1/70] Iter:[0/801], Time: 3.34, lr: [0.0009871336250004174], 
        # Loss: 0.120430, CE_Loss: 0.203374, TCR_01_Loss: -0.000289, TCR_00_Loss: 0.036242, TCR_11_Loss: 0.001533
        _, start_epoch = re.search('Epoch: \[', line, flags=0).span()
        end_epoch, _ = re.search('/70]', line, flags=0).span()
        current_epoch = float(line[start_epoch:end_epoch])
        # print(current_epoch)

        _, start_iter = re.search('Iter:\[', line, flags=0).span()
        end_iter, _ = re.search('/801]', line, flags=0).span()
        current_iter = float(line[start_iter:end_iter])
        # print(current_iter)

        iters.append(current_epoch * 801 + current_iter)

        _, start_loss = re.search('Loss: ', line, flags=0).span()
        end_loss, _ = re.search(', CE_Loss: ', line, flags=0).span()
        current_loss = float(line[start_loss:end_loss])
        # print(current_loss)

        loss.append(current_loss)

        _, start_ce_loss = re.search(', CE_Loss: ', line, flags=0).span()
        end_ce_loss, _ = re.search(', TCR_01_Loss: ', line, flags=0).span()
        current_ce_loss = float(line[start_ce_loss:end_ce_loss])
        # print(current_ce_loss)

        ce_loss.append(current_ce_loss)

        _, start_tcr_01_loss = re.search(', TCR_01_Loss: ', line, flags=0).span()
        end_tcr_01_loss, _ = re.search(', TCR_00_Loss: ', line, flags=0).span()
        current_tcr_01_loss = float(line[start_tcr_01_loss:end_tcr_01_loss])
        # print(current_tcr_01_loss)

        tcr_01_loss.append(current_tcr_01_loss)

        _, start_tcr_00_loss = re.search(', TCR_00_Loss: ', line, flags=0).span()
        end_tcr_00_loss, _ = re.search(', TCR_11_Loss: ', line, flags=0).span()
        current_tcr_00_loss = float(line[start_tcr_00_loss:end_tcr_00_loss])
        # print(current_tcr_00_loss)

        tcr_00_loss.append(current_tcr_00_loss)

        _, start_tcr_11_loss = re.search(', TCR_11_Loss: ', line, flags=0).span()
        end_tcr_11_loss, _ = re.search('\n', line, flags=0).span()
        current_tcr_11_loss = float(line[start_tcr_11_loss:end_tcr_11_loss])
        # print(current_tcr_11_loss)

        tcr_11_loss.append(current_tcr_11_loss)

if flag == 1:
    extrapath = osp.abspath('./slurm-2779461.out')

    print(extrapath)

    with open(extrapath, 'rb') as f:
        while True:
            line = f.readline()
            line = line.decode('utf-8')
            if line == 'Done\n':
                break
            if not line.startswith('Epoch: ['):
                continue

            # print(line)

            # Example: Epoch: [70/140] Iter:[0/801], Time: 8.34, lr: [0.0005358867312681466], 
            # Loss: 0.027249, CE_Loss: 0.262346, TCR_01_Loss: -0.000181, TCR_00_Loss: 0.011601, TCR_11_Loss: 0.001113
            _, start_epoch = re.search('Epoch: \[', line, flags=0).span()
            end_epoch, _ = re.search('/140]', line, flags=0).span()
            current_epoch = float(line[start_epoch:end_epoch])
            # print(current_epoch)

            _, start_iter = re.search('Iter:\[', line, flags=0).span()
            end_iter, _ = re.search('/801]', line, flags=0).span()
            current_iter = float(line[start_iter:end_iter])
            # print(current_iter)

            iters.append(current_epoch * 801 + current_iter)

            _, start_loss = re.search('Loss: ', line, flags=0).span()
            end_loss, _ = re.search(', CE_Loss: ', line, flags=0).span()
            current_loss = float(line[start_loss:end_loss])
            # print(current_loss)

            loss.append(current_loss)

            _, start_ce_loss = re.search(', CE_Loss: ', line, flags=0).span()
            end_ce_loss, _ = re.search(', TCR_01_Loss: ', line, flags=0).span()
            current_ce_loss = float(line[start_ce_loss:end_ce_loss])
            # print(current_ce_loss)

            ce_loss.append(current_ce_loss)

            _, start_tcr_01_loss = re.search(', TCR_01_Loss: ', line, flags=0).span()
            end_tcr_01_loss, _ = re.search(', TCR_00_Loss: ', line, flags=0).span()
            current_tcr_01_loss = float(line[start_tcr_01_loss:end_tcr_01_loss])
            # print(current_tcr_01_loss)

            tcr_01_loss.append(current_tcr_01_loss)

            _, start_tcr_00_loss = re.search(', TCR_00_Loss: ', line, flags=0).span()
            end_tcr_00_loss, _ = re.search(', TCR_11_Loss: ', line, flags=0).span()
            current_tcr_00_loss = float(line[start_tcr_00_loss:end_tcr_00_loss])
            # print(current_tcr_00_loss)

            tcr_00_loss.append(current_tcr_00_loss)

            _, start_tcr_11_loss = re.search(', TCR_11_Loss: ', line, flags=0).span()
            end_tcr_11_loss, _ = re.search('\n', line, flags=0).span()
            current_tcr_11_loss = float(line[start_tcr_11_loss:end_tcr_11_loss])
            # print(current_tcr_11_loss)

            tcr_11_loss.append(current_tcr_11_loss)

plt.figure(1)
plt.plot(iters, loss, label="loss")
plt.plot(iters, ce_loss, label="ce_loss")
plt.plot(iters, tcr_01_loss, label="tcr_01_loss")
plt.plot(iters, tcr_00_loss, label="tcr_00_loss")
plt.plot(iters, tcr_11_loss, label="tcr_11_loss")
plt.xlabel('iters')
plt.ylabel('training loss')
pngName = filename.split('.')[0] + '_loss'
plt.legend()
plt.savefig(osp.join(filedir, pngName))

plt.figure(2)
plt.plot(iters, tcr_01_loss)
plt.xlabel('iters')
plt.ylabel('training tcr_01_loss')
pngName = filename.split('.')[0] + '_tcr_01_loss'
plt.savefig(osp.join(filedir, pngName))

plt.figure(3)
plt.plot(iters, tcr_00_loss)
plt.xlabel('iters')
plt.ylabel('training tcr_00_loss')
pngName = filename.split('.')[0] + '_tcr_00_loss'
plt.savefig(osp.join(filedir, pngName))

plt.figure(4)
plt.plot(iters, tcr_11_loss)
plt.xlabel('iters')
plt.ylabel('training tcr_11_loss')
pngName = filename.split('.')[0] + '_tcr_11_loss'
plt.savefig(osp.join(filedir, pngName))

# plot loss and miou curves for validation set
val_loss, val_miou, epochs = [], [], []
epoch = 0
with open(fullpath, 'rb') as f:
    while True:
        line = f.readline()
        line = line.decode('utf-8')
        if line == 'Done\n':
            break
        if not line.startswith('Loss: '):
            continue
        # print(line)

        # Example: Loss: 0.250, MeanIU:  0.4352, Best_mIoU:  0.4352
        epochs.append(epoch)

        _, start_val_loss = re.search('Loss: ', line, flags=0).span()
        end_val_loss, _ = re.search(', MeanIU:  ', line, flags=0).span()
        current_val_loss = float(line[start_val_loss:end_val_loss])
        # print(current_val_loss)

        val_loss.append(current_val_loss)

        _, start_miou = re.search(', MeanIU:  ', line, flags=0).span()
        end_miou, _ = re.search(', Best_mIoU:  ', line, flags=0).span()
        current_miou = float(line[start_miou:end_miou])
        # print(current_miou)

        val_miou.append(current_miou)

        epoch += 1

if flag == 1:
    with open(extrapath, 'rb') as f:
        while True:
            line = f.readline()
            line = line.decode('utf-8')
            if line == 'Done\n':
                break
            if not line.startswith('Loss: '):
                continue
            # print(line)

            # Example: Loss: 0.250, MeanIU:  0.4352, Best_mIoU:  0.4352
            epochs.append(epoch)

            _, start_val_loss = re.search('Loss: ', line, flags=0).span()
            end_val_loss, _ = re.search(', MeanIU:  ', line, flags=0).span()
            current_val_loss = float(line[start_val_loss:end_val_loss])
            # print(current_val_loss)

            val_loss.append(current_val_loss)

            _, start_miou = re.search(', MeanIU:  ', line, flags=0).span()
            end_miou, _ = re.search(', Best_mIoU:  ', line, flags=0).span()
            current_miou = float(line[start_miou:end_miou])
            # print(current_miou)

            val_miou.append(current_miou)

            epoch += 1

plt.figure(5)
plt.plot(epochs, val_loss)
plt.xlabel('epochs')
plt.ylabel('val_loss')
pngName = filename.split('.')[0] + '_val_loss'
plt.savefig(osp.join(filedir, pngName))

plt.figure(6)
plt.plot(epochs, val_miou)
plt.xlabel('epochs')
plt.ylabel('val_miou')
pngName = filename.split('.')[0] + '_val_miou'
plt.savefig(osp.join(filedir, pngName))