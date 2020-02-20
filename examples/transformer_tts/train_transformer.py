import os
from tqdm import tqdm
from tensorboardX import SummaryWriter
from pathlib import Path
from collections import OrderedDict
import argparse
from parse import add_config_options_to_parser
from pprint import pprint
from ruamel import yaml
from matplotlib import cm
import numpy as np
import paddle.fluid as fluid
import paddle.fluid.dygraph as dg
import paddle.fluid.layers as layers
from parakeet.models.transformer_tts.utils import cross_entropy
from data import LJSpeechLoader
from parakeet.models.transformer_tts.transformer_tts import TransformerTTS

def load_checkpoint(step, model_path):
    model_dict, opti_dict = fluid.dygraph.load_dygraph(os.path.join(model_path, step))
    new_state_dict = OrderedDict()
    for param in model_dict:
        if param.startswith('_layers.'):
            new_state_dict[param[8:]] = model_dict[param]
        else:
            new_state_dict[param] = model_dict[param]
    return new_state_dict, opti_dict


def main(args):
    local_rank = dg.parallel.Env().local_rank if args.use_data_parallel else 0
    nranks = dg.parallel.Env().nranks if args.use_data_parallel else 1

    with open(args.config_path) as f:
        cfg = yaml.load(f, Loader=yaml.Loader)

    global_step = 0
    place = (fluid.CUDAPlace(dg.parallel.Env().dev_id)
             if args.use_data_parallel else fluid.CUDAPlace(0)
             if args.use_gpu else fluid.CPUPlace())

    if not os.path.exists(args.log_dir):
            os.mkdir(args.log_dir)
    path = os.path.join(args.log_dir,'transformer')

    writer = SummaryWriter(path) if local_rank == 0 else None
    
    with dg.guard(place):
        model = TransformerTTS(cfg)

        model.train()
        optimizer = fluid.optimizer.AdamOptimizer(learning_rate=dg.NoamDecay(1/(cfg['warm_up_step'] *( args.lr ** 2)), cfg['warm_up_step']), 
                                                  parameter_list=model.parameters())
        
        reader = LJSpeechLoader(cfg, args, nranks, local_rank, shuffle=True).reader()

        if args.checkpoint_path is not None:
            model_dict, opti_dict = load_checkpoint(str(args.transformer_step), os.path.join(args.checkpoint_path, "transformer"))
            model.set_dict(model_dict)
            optimizer.set_dict(opti_dict)
            global_step = args.transformer_step
            print("load checkpoint!!!")

        if args.use_data_parallel:
            strategy = dg.parallel.prepare_context()
            model = fluid.dygraph.parallel.DataParallel(model, strategy)
        
        for epoch in range(args.epochs):
            pbar = tqdm(reader)
            for i, data in enumerate(pbar):
                pbar.set_description('Processing at epoch %d'%epoch)
                character, mel, mel_input, pos_text, pos_mel, text_length, _ = data

                global_step += 1
                mel_pred, postnet_pred, attn_probs, stop_preds, attn_enc, attn_dec = model(character, mel_input, pos_text, pos_mel)
                

                label = (pos_mel == 0).astype(np.float32)
                    
                mel_loss = layers.mean(layers.abs(layers.elementwise_sub(mel_pred, mel)))
                post_mel_loss = layers.mean(layers.abs(layers.elementwise_sub(postnet_pred, mel)))
                loss = mel_loss + post_mel_loss
                # Note: When used stop token loss the learning did not work.
                if args.stop_token:
                    stop_loss = cross_entropy(stop_preds, label)
                    loss = loss + stop_loss

                if local_rank==0:
                    writer.add_scalars('training_loss', {
                        'mel_loss':mel_loss.numpy(),
                        'post_mel_loss':post_mel_loss.numpy()
                    }, global_step)

                    if args.stop_token:
                        writer.add_scalar('stop_loss', stop_loss.numpy(), global_step)

                    if args.use_data_parallel:
                        writer.add_scalars('alphas', {
                            'encoder_alpha':model._layers.encoder.alpha.numpy(),
                            'decoder_alpha':model._layers.decoder.alpha.numpy(),
                        }, global_step)
                    else:
                        writer.add_scalars('alphas', {
                            'encoder_alpha':model.encoder.alpha.numpy(),
                            'decoder_alpha':model.decoder.alpha.numpy(),
                        }, global_step)

                    writer.add_scalar('learning_rate', optimizer._learning_rate.step().numpy(), global_step)

                    if global_step % args.image_step == 1:
                        for i, prob in enumerate(attn_probs):
                            for j in range(4):
                                    x = np.uint8(cm.viridis(prob.numpy()[j*16]) * 255)
                                    writer.add_image('Attention_%d_0'%global_step, x, i*4+j, dataformats="HWC")

                        for i, prob in enumerate(attn_enc):
                            for j in range(4):
                                x = np.uint8(cm.viridis(prob.numpy()[j*16]) * 255)
                                writer.add_image('Attention_enc_%d_0'%global_step, x, i*4+j, dataformats="HWC")

                        for i, prob in enumerate(attn_dec):
                            for j in range(4):
                                x = np.uint8(cm.viridis(prob.numpy()[j*16]) * 255)
                                writer.add_image('Attention_dec_%d_0'%global_step, x, i*4+j, dataformats="HWC")
                                
                if args.use_data_parallel:
                    loss = model.scale_loss(loss)
                    loss.backward()
                    model.apply_collective_grads()
                else:
                    loss.backward()
                optimizer.minimize(loss, grad_clip = fluid.dygraph_grad_clip.GradClipByGlobalNorm(cfg['grad_clip_thresh']))
                model.clear_gradients()
                
                # save checkpoint
                if local_rank==0 and global_step % args.save_step == 0:
                    if not os.path.exists(args.save_path):
                        os.mkdir(args.save_path)
                    save_path = os.path.join(args.save_path,'transformer/%d' % global_step)
                    dg.save_dygraph(model.state_dict(), save_path)
                    dg.save_dygraph(optimizer.state_dict(), save_path)
        if local_rank==0:
            writer.close()
                    

if __name__ =='__main__':
    parser = argparse.ArgumentParser(description="Train TransformerTTS model")
    add_config_options_to_parser(parser)

    args = parser.parse_args()
    # Print the whole config setting.
    pprint(args)
    main(args)
