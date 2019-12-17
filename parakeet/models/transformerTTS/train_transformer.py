import os
from tqdm import tqdm
import paddle.fluid.dygraph as dg
import paddle.fluid.layers as layers
from network import *
from tensorboardX import SummaryWriter
from pathlib import Path
import jsonargparse
from parse import add_config_options_to_parser
from pprint import pprint
from matplotlib import cm
from data import LJSpeechLoader

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
    path = os.path.join(cfg.log_dir,'transformer')

    writer = SummaryWriter(path) if local_rank == 0 else None
    
    with dg.guard(place):
        model = Model('transtts', cfg)

        model.train()
        optimizer = fluid.optimizer.AdamOptimizer(learning_rate=dg.NoamDecay(1/(4000 *( cfg.lr ** 2)), 4000))
        
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

                global_step += 1
                
                mel_pred, postnet_pred, attn_probs, stop_preds, attn_enc, attn_dec = model(character, mel_input, pos_text, pos_mel)
        
                mel_loss = layers.mean(layers.abs(layers.elementwise_sub(mel_pred, mel)))
                post_mel_loss = layers.mean(layers.abs(layers.elementwise_sub(postnet_pred, mel)))
                loss = mel_loss + post_mel_loss

                if local_rank==0:
                    writer.add_scalars('training_loss', {
                        'mel_loss':mel_loss.numpy(),
                        'post_mel_loss':post_mel_loss.numpy(),
                    }, global_step)

                    writer.add_scalars('alphas', {
                        'encoder_alpha':model.encoder.alpha.numpy(),
                        'decoder_alpha':model.decoder.alpha.numpy(),
                    }, global_step)

                    writer.add_scalar('learning_rate', optimizer._learning_rate.step().numpy(), global_step)

                    if global_step % cfg.image_step == 1:
                        for i, prob in enumerate(attn_probs):
                            for j in range(4):
                                    x = np.uint8(cm.viridis(prob.numpy()[j*16]) * 255)
                                    writer.add_image('Attention_enc_%d_0'%global_step, x, i*4+j, dataformats="HWC")

                        for i, prob in enumerate(attn_enc):
                            for j in range(4):
                                x = np.uint8(cm.viridis(prob.numpy()[j*16]) * 255)
                                writer.add_image('Attention_enc_%d_0'%global_step, x, i*4+j, dataformats="HWC")

                        for i, prob in enumerate(attn_dec):
                            for j in range(4):
                                x = np.uint8(cm.viridis(prob.numpy()[j*16]) * 255)
                                writer.add_image('Attention_dec_%d_0'%global_step, x, i*4+j, dataformats="HWC")

                if cfg.use_data_parallel:
                    loss = model.scale_loss(loss)
                    loss.backward()
                    model.apply_collective_grads()
                else:
                    loss.backward()
                optimizer.minimize(loss, grad_clip = fluid.dygraph_grad_clip.GradClipByGlobalNorm(1))
                model.clear_gradients()

                # save checkpoint
                if local_rank==0 and global_step % cfg.save_step == 0:
                    if not os.path.exists(cfg.save_path):
                        os.mkdir(cfg.save_path)
                    save_path = os.path.join(cfg.save_path,'transformer/%d' % global_step)
                    dg.save_dygraph(model.state_dict(), save_path)
                    dg.save_dygraph(optimizer.state_dict(), save_path)
        if local_rank==0:
            writer.close()
                    

if __name__ =='__main__':
    parser = jsonargparse.ArgumentParser(description="Train TransformerTTS model", formatter_class='default_argparse')
    add_config_options_to_parser(parser)
    cfg = parser.parse_args('-c ./config/train_transformer.yaml'.split())
    main(cfg)