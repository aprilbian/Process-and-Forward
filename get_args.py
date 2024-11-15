import argparse

def get_args():
    ################################
    # Setup Parameters and get args
    ################################
    parser = argparse.ArgumentParser()

    parser.add_argument('-dataset', default  = 'cifar')

    ##### Neural Network setting
    parser.add_argument('-cout', type=int, default  = 12)
    parser.add_argument('-cfeat', type=int, default  = 256)

    # The transformer setting
    parser.add_argument('-n_patches', type=int, default  = 64)
    parser.add_argument('-n_feat', type=int, default  = 48)
    parser.add_argument('-hidden_size', type=int, default  = 256)
    parser.add_argument('-feedforward_size', type=int, default  = 1024)
    parser.add_argument('-n_heads', type=int, default  = 8)
    parser.add_argument('-n_layers', type=int, default  = 8)
    parser.add_argument('-dropout_prob', type=float, default  = 0.1)

    parser.add_argument('-unit_trans_feat', type=int, default  = 4)
    parser.add_argument('-max_trans_feat', type=int, default  = 6)
    parser.add_argument('-n_trans_feat', type=int, default  = 24)

    ##### The relay channel
    parser.add_argument('-relay_mode', default  = 'PF')
    parser.add_argument('-channel_mode', default = 'awgn')
    parser.add_argument('-layers', default = 5)
    parser.add_argument('-unit', default = 4)


    parser.add_argument('-adapt', default  = True)
    parser.add_argument('-n_adapt_embed', default  = 4)

    # Discription 3
    parser.add_argument('-P',  default  = 3.0)
    parser.add_argument('-gamma1',  default  = 5)
    parser.add_argument('-gamma2',  default  = 5)

    parser.add_argument('-gamma_rng',  default  = 5)
    parser.add_argument('-P_rng',  default  = 3)
    parser.add_argument('-layer_rng',  default  = 3)

    parser.add_argument('-fading',  default  = False)


    ##### training setting
    parser.add_argument('-epoch', type=int, default  = 2000)
    parser.add_argument('-lr', type=float, default  = 1e-4)
    parser.add_argument('-train_patience', type=int, default  = 30)
    parser.add_argument('-train_batch_size', type=int, default  = 64)

    parser.add_argument('-val_batch_size', type=int, default  = 32)
    parser.add_argument('-resume', default  = False)
    parser.add_argument('-path', default  = 'models/')

    args = parser.parse_args()

    return args
