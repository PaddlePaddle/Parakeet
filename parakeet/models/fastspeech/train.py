import numpy as np
import argparse
import os
import time
import math
import jsonargparse
from pathlib import Path
from tqdm import tqdm
from tensorboardX import SummaryWriter
import paddle.fluid.dygraph as dg
import paddle.fluid.layers as layers
import paddle.fluid as fluid
from parse import add_config_options_to_parser
from pprint import pprint
from network import FastSpeech
from utils import get_alignment
from parakeet.models.dataloader.jlspeech import LJSpeechLoader
from parakeet.models.transformerTTS.network import TransformerTTS

class MyDataParallel(dg.parallel.DataParallel):
    """
    A data parallel proxy for model.
    """

    def __init__(self, layers, strategy):
        super(MyDataParallel, self).__init__(layers, strategy)

    def __getattr__(self, key):
        if key in self.__dict__:
            return object.__getattribute__(self, key)
        elif key is "_layers":
            return object.__getattribute__(self, "_sub_layers")["_layers"]
        else:
            return getattr(
                object.__getattribute__(self, "_sub_layers")["_layers"], key)

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
    path = os.path.join(cfg.log_dir,'fastspeech')

    writer = SummaryWriter(path) if local_rank == 0 else None

    with dg.guard(place):
        transformerTTS = TransformerTTS(cfg)
        model_path = os.path.join(cfg.transtts_path, "transformer")
        model_dict, _ = fluid.dygraph.load_dygraph(os.path.join(model_path, str(cfg.transformer_step)))
        #for param in transformerTTS.state_dict():
        #   print(param)
        
        transformerTTS.set_dict(model_dict)
        transformerTTS.eval()

        model = FastSpeech(cfg)
        model.train()
        optimizer = fluid.optimizer.AdamOptimizer(learning_rate=dg.NoamDecay(1/(cfg.warm_up_step *( cfg.lr ** 2)), cfg.warm_up_step),
                                                  parameter_list=model.parameters())
        reader = LJSpeechLoader(cfg, nranks, local_rank).reader()
        
        if cfg.checkpoint_path is not None:
            model_dict, opti_dict = fluid.dygraph.load_dygraph(cfg.checkpoint_path)
            model.set_dict(model_dict)
            optimizer.set_dict(opti_dict)
            print("load checkpoint!!!")

        if cfg.use_data_parallel:
            strategy = dg.parallel.prepare_context()
            model = MyDataParallel(model, strategy)

        for epoch in range(cfg.epochs):
            pbar = tqdm(reader)

            for i, data in enumerate(pbar):
                pbar.set_description('Processing at epoch %d'%epoch)
                character, mel, mel_input, pos_text, pos_mel, text_length = data

                _, _, attn_probs, _, _, _ = transformerTTS(character, mel_input, pos_text, pos_mel)
                alignment = dg.to_variable(get_alignment(attn_probs, cfg.transformer_head)).astype(np.float32)

                global_step += 1
                    
                #Forward
                result= model(character, 
                              pos_text, 
                              mel_pos=pos_mel,  
                              length_target=alignment)
                mel_output, mel_output_postnet, duration_predictor_output, _, _ = result
                mel_loss = layers.mse_loss(mel_output, mel)
                mel_postnet_loss = layers.mse_loss(mel_output_postnet, mel)
                duration_loss = layers.mean(layers.abs(layers.elementwise_sub(duration_predictor_output, alignment)))
                total_loss = mel_loss + mel_postnet_loss + duration_loss

                if local_rank==0:
                    print('epoch:{}, step:{}, mel_loss:{}, mel_postnet_loss:{}, duration_loss:{}'.format(epoch, global_step, mel_loss.numpy(), mel_postnet_loss.numpy(), duration_loss.numpy()))

                    writer.add_scalar('mel_loss', mel_loss.numpy(), global_step)
                    writer.add_scalar('post_mel_loss', mel_postnet_loss.numpy(), global_step)
                    writer.add_scalar('duration_loss', duration_loss.numpy(), global_step)
                    writer.add_scalar('learning_rate', optimizer._learning_rate.step().numpy(), global_step)


                if cfg.use_data_parallel:
                    total_loss = model.scale_loss(total_loss)
                    total_loss.backward()
                    model.apply_collective_grads()
                else:
                    total_loss.backward()
                optimizer.minimize(total_loss, grad_clip = fluid.dygraph_grad_clip.GradClipByGlobalNorm(cfg.grad_clip_thresh))
                model.clear_gradients()

                 # save checkpoint
                if local_rank==0 and global_step % cfg.save_step == 0:
                    if not os.path.exists(cfg.save_path):
                        os.mkdir(cfg.save_path)
                    save_path = os.path.join(cfg.save_path,'fastspeech/%d' % global_step)
                    dg.save_dygraph(model.state_dict(), save_path)
                    dg.save_dygraph(optimizer.state_dict(), save_path)
        if local_rank==0:
            writer.close()


if __name__ =='__main__':
    parser = jsonargparse.ArgumentParser(description="Train Fastspeech model", formatter_class='default_argparse')
    add_config_options_to_parser(parser)
    cfg = parser.parse_args('-c config/fastspeech.yaml'.split())
    main(cfg)