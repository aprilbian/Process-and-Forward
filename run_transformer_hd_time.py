import numpy as np 
import copy
import torch
import torch.utils.data as data
from collections import OrderedDict
from tqdm import tqdm
import torch.optim as optim
import torch.optim.lr_scheduler as LS

from get_args import get_args
from modules import *
from dataset import CIFAR10, ImageNet, Kodak
from utils import *
from relay_network_hd import *


from torch.utils.tensorboard import SummaryWriter

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#device = torch.device("cpu")

###### Parameter Setting
args = get_args()
args.device = device

job_name = 'JSCC_half_duplex_transformer_'+args.channel_mode+'_relay_' + args.relay_mode +'_dataset_'+str(args.dataset)+'_layers_'+str(args.layers)+\
            '_P_' + str(args.P) + '_gamma1_' + str(args.gamma1) + '_gamma2_' + str(args.gamma2) + '_unit_trans_feat_' + str(args.unit_trans_feat)+ '_max_trans_feat_' + str(args.max_trans_feat)\
            + '_hidden_size_' + str(args.hidden_size) + '_n_heads_' + str(args.n_heads) + '_n_layers_' + str(args.n_layers) + '_feedforward_size_' + str(args.feedforward_size)

if args.adapt:
    job_name = job_name + '_gamma_rng_' + str(args.gamma_rng)

print(args)
print(job_name)

frame_size = (32, 32)
src_ratio = args.cout / (3*4*4)

writter = SummaryWriter('runs/' + job_name)

train_set = CIFAR10('datasets/cifar-10-batches-py', 'TRAIN')
valid_set = CIFAR10('datasets/cifar-10-batches-py', 'VALIDATE')
eval_set = CIFAR10('datasets/cifar-10-batches-py', 'EVALUATE')


###### The JSCC Model

source_enc = tf_encoder(args).to(args.device)
source_dec = tf_decoder(args).to(args.device)
if args.relay_mode == 'PF':
    args_parity_enc = copy.deepcopy(args)
    args_parity_enc.n_feat = args.max_trans_feat*args.unit_trans_feat
    args_parity_enc.n_trans_feat = args.max_trans_feat*args.unit_trans_feat
    args_parity_enc.n_layers = 4
    relay_enc = tf_encoder(args_parity_enc).to(args.device)
    jscc_model = RelayHD_Transformer_Full_Adapt(args, source_enc, source_dec, relay_enc)
else:
    jscc_model = RelayHD_Transformer_Full_Adapt(args, source_enc, source_dec, None)





# load pre-trained
if args.resume == False:
    pass
else:
    _ = load_weights(job_name, jscc_model)

solver = optim.Adam(jscc_model.parameters(), lr=args.lr)
scheduler = LS.MultiplicativeLR(solver, lr_lambda=lambda x: 0.9)
es = EarlyStopping(mode='min', min_delta=0, patience=args.train_patience)

###### Dataloader
train_loader = data.DataLoader(
    dataset=train_set,
    batch_size=args.train_batch_size,
    shuffle=True,
    num_workers=2
        )

valid_loader = data.DataLoader(
    dataset=valid_set,
    batch_size=args.val_batch_size,
    shuffle=True,
    num_workers=2
        )

eval_loader = data.DataLoader(
    dataset=eval_set,
    batch_size=args.val_batch_size,
    shuffle=True,
    num_workers=2
)


def train_epoch(loader, model, solvers):

    model.train()

    with tqdm(loader, unit='batch') as tepoch:
        for _, (images, _) in enumerate(tepoch):
            
            epoch_postfix = OrderedDict()

            images = images.to(args.device).float()
            
            solvers.zero_grad()
            #output, _, _ = model(images, is_train = True)
            output = model(images, is_train = True)

            loss = nn.MSELoss()(output, images)

            loss.backward()
            solvers.step()

            epoch_postfix['l2_loss'] = '{:.4f}'.format(loss.item())

            tepoch.set_postfix(**epoch_postfix)


def validate_epoch(loader, model):

    model.eval()

    loss_hist = []
    psnr_hist = []
    ssim_hist = []
    theta_hist = []
    beta_hist = []
    #msssim_hist = []

    with torch.no_grad():
        with tqdm(loader, unit='batch') as tepoch:
            for _, (images, _) in enumerate(tepoch):

                epoch_postfix = OrderedDict()

                images = images.to(args.device).float()

                #output, theta, beta = model(images, is_train = False)
                output = model(images, is_train = False)
                loss = nn.MSELoss()(output, images)

                epoch_postfix['l2_loss'] = '{:.4f}'.format(loss.item())

                ######  Predictions  ######
                predictions = torch.chunk(output, chunks=output.size(0), dim=0)
                target = torch.chunk(images, chunks=images.size(0), dim=0)

                ######  PSNR/SSIM/etc  ######

                psnr_vals = calc_psnr(predictions, target)
                psnr_hist.extend(psnr_vals)
                epoch_postfix['psnr'] = torch.mean(torch.tensor(psnr_vals)).item()

                ssim_vals = calc_ssim(predictions, target)
                ssim_hist.extend(ssim_vals)
                epoch_postfix['ssim'] = torch.mean(torch.tensor(ssim_vals)).item()
                
                # Show the snr/loss/psnr/ssim
                tepoch.set_postfix(**epoch_postfix)

                loss_hist.append(loss.item())
                #theta_hist.append(theta.cpu().numpy())
                #beta_hist.append(beta.cpu().numpy())
            
            loss_mean = np.nanmean(loss_hist)
            #theta_mean = np.nanmean(theta_hist)
            #beta_mean = np.nanmean(beta_hist)

            psnr_hist = torch.tensor(psnr_hist)
            psnr_mean = torch.mean(psnr_hist).item()
            psnr_std = torch.sqrt(torch.var(psnr_hist)).item()

            ssim_hist = torch.tensor(ssim_hist)
            ssim_mean = torch.mean(ssim_hist).item()
            ssim_std = torch.sqrt(torch.var(ssim_hist)).item()

            predictions = torch.cat(predictions, dim=0)[:, [2, 1, 0]]
            target = torch.cat(target, dim=0)[:, [2, 1, 0]]

            return_aux = {'psnr': psnr_mean,
                            'ssim': ssim_mean,
                            'predictions': predictions,
                            'target': target,
                            'psnr_std': psnr_std,
                            'ssim_std': ssim_std,
                            #'theta': theta_mean,
                            #'beta': beta_mean
                            }

        
    return loss_mean, return_aux



if __name__ == '__main__':
    epoch = 0

    while epoch < args.epoch and not args.resume:
        
        epoch += 1
        
        train_epoch(train_loader, jscc_model, solver)

        valid_loss, valid_aux = validate_epoch(valid_loader, jscc_model)

        writter.add_scalar('loss', valid_loss, epoch)
        writter.add_scalar('psnr', valid_aux['psnr'], epoch)

        flag, best, best_epoch, bad_epochs = es.step(torch.Tensor([valid_loss]), epoch)
        if flag:
            print('ES criterion met; loading best weights from epoch {}'.format(best_epoch))
            _ = load_weights(job_name, jscc_model, args.device)
            break
        else:
            # TODO put this in trainer
            if bad_epochs == 0:
                print('average l2_loss: ', valid_loss.item())
                save_nets(job_name, jscc_model, epoch)
                best_epoch = epoch
                print('saving best net weights...')
            elif bad_epochs % (es.patience//3) == 0:
                scheduler.step()
                print('lr updated: {:.5f}'.format(scheduler.get_last_lr()[0]))



    print('evaluating...')
    print(job_name)

    psnr_list = []
    psnr_std_list = []
    ssim_list = []

    for c_sr in [0, 10/3, 20/3, 9.99]:
        for c_rd in [0, 10/3, 20/3, 9.99]:

            jscc_model.gamma1, jscc_model.gamma2 = c_sr, c_rd

            _, eval_aux = validate_epoch(eval_loader, jscc_model)

            psnr_list.append(eval_aux['psnr'])
            ssim_list.append(eval_aux['ssim'])
            psnr_std_list.append(eval_aux['psnr_std'])

    print(psnr_list)
    print(ssim_list)
    print(psnr_std_list)