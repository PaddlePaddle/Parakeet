from tensorboardX import SummaryWriter
import os
from tqdm import tqdm
from pathlib import Path
from collections import OrderedDict
import jsonargparse
from parse import add_config_options_to_parser
from pprint import pprint
from parakeet.models.dataloader.ljspeech import LJSpeechLoader
from network import *

def load_checkpoint(step, model_path):
    model_dict, opti_dict = fluid.dygraph.load_dygraph(os.path.join(model_path, step))
    new_state_dict = OrderedDict()
    for param in model_dict:
        if param.startswith('_layers.'):
            new_state_dict[param[8:]] = model_dict[param]
        else:
            new_state_dict[param] = model_dict[param]
    return new_state_dict, opti_dict

def main(cfg):
    
    local_rank = dg.parallel.Env().local_rank if cfg.use_data_parallel else 0
    nranks = dg.parallel.Env().nranks if cfg.use_data_parallel else 1
    
    if local_rank == 0:
        # Print the whole config setting.
        pprint(jsonargparse.namespace_to_dict(cfg))

    global_step = 0
    place = (fluid.CUDAPlace(dg.parallel.Env().dev_id)
             if cfg.use_data_parallel else fluid.CUDAPlace(0)
             if cfg.use_gpu else fluid.CPUPlace())
    
    if not os.path.exists(cfg.log_dir):
            os.mkdir(cfg.log_dir)
    path = os.path.join(cfg.log_dir,'postnet')

    writer = SummaryWriter(path) if local_rank == 0 else None

    with dg.guard(place):   
        model = ModelPostNet(cfg)

        model.train()
        optimizer = fluid.optimizer.AdamOptimizer(learning_rate=dg.NoamDecay(1/(cfg.warm_up_step *( cfg.lr ** 2)), cfg.warm_up_step),
                                                  parameter_list=model.parameters())


        if cfg.checkpoint_path is not None:
            model_dict, opti_dict = load_checkpoint(str(cfg.postnet_step), os.path.join(cfg.checkpoint_path, "postnet"))
            model.set_dict(model_dict)
            optimizer.set_dict(opti_dict)
            global_step = cfg.postnet_step
            print("load checkpoint!!!")

        if cfg.use_data_parallel:
            strategy = dg.parallel.prepare_context()
            model = fluid.dygraph.parallel.DataParallel(model, strategy)

        reader = LJSpeechLoader(cfg, nranks, local_rank, is_vocoder=True).reader()

        for epoch in range(cfg.epochs):
            pbar = tqdm(reader)
            for i, data in enumerate(pbar):
                pbar.set_description('Processing at epoch %d'%epoch)
                mel, mag = data
                mag = dg.to_variable(mag.numpy())
                mel = dg.to_variable(mel.numpy())
                global_step += 1

                mag_pred = model(mel)
                loss = layers.mean(layers.abs(layers.elementwise_sub(mag_pred, mag)))
                
                if cfg.use_data_parallel:
                    loss = model.scale_loss(loss)
                    loss.backward()
                    model.apply_collective_grads()
                else:
                    loss.backward()
                optimizer.minimize(loss, grad_clip = fluid.dygraph_grad_clip.GradClipByGlobalNorm(cfg.grad_clip_thresh))
                model.clear_gradients()
                
                if local_rank==0:
                    writer.add_scalars('training_loss',{
                        'loss':loss.numpy(),
                    }, global_step)

                    if global_step % cfg.save_step == 0:
                        if not os.path.exists(cfg.save_path):
                            os.mkdir(cfg.save_path)
                        save_path = os.path.join(cfg.save_path,'postnet/%d' % global_step)
                        dg.save_dygraph(model.state_dict(), save_path)
                        dg.save_dygraph(optimizer.state_dict(), save_path)

        if local_rank==0:
            writer.close()

if __name__ == '__main__':
    parser = jsonargparse.ArgumentParser(description="Train postnet model", formatter_class='default_argparse')
    add_config_options_to_parser(parser)
    cfg = parser.parse_args('-c ./config/train_postnet.yaml'.split())
    main(cfg)