from tensorboardX import SummaryWriter
import os
from tqdm import tqdm
from pathlib import Path
from collections import OrderedDict
import argparse
from ruamel import yaml
from parse import add_config_options_to_parser
from pprint import pprint
import paddle.fluid as fluid
import paddle.fluid.dygraph as dg
import paddle.fluid.layers as layers
from data import LJSpeechLoader
from parakeet.models.transformer_tts.vocoder import Vocoder

def load_checkpoint(step, model_path):
    model_dict, opti_dict = dg.load_dygraph(os.path.join(model_path, step))
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
    path = os.path.join(args.log_dir,'postnet')

    writer = SummaryWriter(path) if local_rank == 0 else None

    with dg.guard(place):   
        model = Vocoder(cfg, args.batch_size)

        model.train()
        optimizer = fluid.optimizer.AdamOptimizer(learning_rate=dg.NoamDecay(1/(cfg['warm_up_step'] *( args.lr ** 2)), cfg['warm_up_step']),
                                                  parameter_list=model.parameters())


        if args.checkpoint_path is not None:
            model_dict, opti_dict = load_checkpoint(str(args.vocoder_step), os.path.join(args.checkpoint_path, "postnet"))
            model.set_dict(model_dict)
            optimizer.set_dict(opti_dict)
            global_step = args.vocoder_step
            print("load checkpoint!!!")

        if args.use_data_parallel:
            strategy = dg.parallel.prepare_context()
            model = fluid.dygraph.parallel.DataParallel(model, strategy)

        reader = LJSpeechLoader(cfg, args, nranks, local_rank, is_vocoder=True).reader()

        for epoch in range(args.epochs):
            pbar = tqdm(reader)
            for i, data in enumerate(pbar):
                pbar.set_description('Processing at epoch %d'%epoch)
                mel, mag = data
                mag = dg.to_variable(mag.numpy())
                mel = dg.to_variable(mel.numpy())
                global_step += 1

                mag_pred = model(mel)
                loss = layers.mean(layers.abs(layers.elementwise_sub(mag_pred, mag)))
                
                if args.use_data_parallel:
                    loss = model.scale_loss(loss)
                    loss.backward()
                    model.apply_collective_grads()
                else:
                    loss.backward()
                optimizer.minimize(loss, grad_clip = fluid.dygraph_grad_clip.GradClipByGlobalNorm(cfg['grad_clip_thresh']))
                model.clear_gradients()
                
                if local_rank==0:
                    writer.add_scalars('training_loss',{
                        'loss':loss.numpy(),
                    }, global_step)

                    if global_step % args.save_step == 0:
                        if not os.path.exists(args.save_path):
                            os.mkdir(args.save_path)
                        save_path = os.path.join(args.save_path,'postnet/%d' % global_step)
                        dg.save_dygraph(model.state_dict(), save_path)
                        dg.save_dygraph(optimizer.state_dict(), save_path)

        if local_rank==0:
            writer.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train postnet model")
    add_config_options_to_parser(parser)
    args = parser.parse_args()
    # Print the whole config setting.
    pprint(args)
    main(args)