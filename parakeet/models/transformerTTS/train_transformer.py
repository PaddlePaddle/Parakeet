from preprocess import batch_examples, LJSpeech
import os
from tqdm import tqdm
import paddle.fluid.dygraph as dg
import paddle.fluid.layers as layers
from network import *
from tensorboardX import SummaryWriter
from parakeet.data.datacargo import DataCargo
from pathlib import Path
import jsonargparse
from parse import add_config_options_to_parser
from pprint import pprint
from matplotlib import cm

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


def main():
    parser = jsonargparse.ArgumentParser(description="Train TransformerTTS model", formatter_class='default_argparse')
    add_config_options_to_parser(parser)
    cfg = parser.parse_args('-c ./config/train_transformer.yaml'.split())
    
    local_rank = dg.parallel.Env().local_rank

    if local_rank == 0:
        # Print the whole config setting.
        pprint(jsonargparse.namespace_to_dict(cfg))


    LJSPEECH_ROOT = Path(cfg.data_path)
    dataset = LJSpeech(LJSPEECH_ROOT)
    dataloader = DataCargo(dataset, batch_size=cfg.batch_size, shuffle=True, collate_fn=batch_examples, drop_last=True)
    global_step = 0
    place = (fluid.CUDAPlace(dg.parallel.Env().dev_id)
             if cfg.use_data_parallel else fluid.CUDAPlace(0)
             if cfg.use_gpu else fluid.CPUPlace())

    if not os.path.exists(cfg.log_dir):
            os.mkdir(cfg.log_dir)
    path = os.path.join(cfg.log_dir,'transformer')

    writer = SummaryWriter(path) if local_rank == 0 else None
    
    with dg.guard(place):
        if cfg.use_data_parallel:
            strategy = dg.parallel.prepare_context()

        # dataloader
        input_fields = {
                'names': ['character', 'mel', 'mel_input', 'pos_text', 'pos_mel', 'text_len'],
                'shapes':
                [[cfg.batch_size, None], [cfg.batch_size, None, 80], [cfg.batch_size, None, 80], [cfg.batch_size, 1], [cfg.batch_size, 1], [cfg.batch_size, 1]],
                'dtypes': ['float32', 'float32', 'float32', 'int64', 'int64', 'int64'],
                'lod_levels': [0, 0, 0, 0, 0, 0]
            }

        inputs = [
            fluid.data(
                name=input_fields['names'][i],
                shape=input_fields['shapes'][i],
                dtype=input_fields['dtypes'][i],
                lod_level=input_fields['lod_levels'][i])
            for i in range(len(input_fields['names']))
        ]

        reader = fluid.io.DataLoader.from_generator(
            feed_list=inputs,
            capacity=32,
            iterable=True,
            use_double_buffer=True,
            return_list=True)

        model = Model('transtts', cfg)

        model.train()
        optimizer = fluid.optimizer.AdamOptimizer(learning_rate=dg.NoamDecay(1/(4000 *( cfg.lr ** 2)), 4000))

        if cfg.checkpoint_path is not None:
            model_dict, opti_dict = fluid.dygraph.load_dygraph(cfg.checkpoint_path)
            model.set_dict(model_dict)
            optimizer.set_dict(opti_dict)
            print("load checkpoint!!!")

        if cfg.use_data_parallel:
            model = MyDataParallel(model, strategy)

        for epoch in range(cfg.epochs):
            reader.set_batch_generator(dataloader, place)
            pbar = tqdm(reader())
            for i, data in enumerate(pbar):
                pbar.set_description('Processing at epoch %d'%epoch)
                character, mel, mel_input, pos_text, pos_mel, text_length = data

                global_step += 1
                
                mel_pred, postnet_pred, attn_probs, stop_preds, attn_enc, attn_dec = model(character, mel_input, pos_text, pos_mel)
        
                mel_loss = layers.mean(layers.abs(layers.elementwise_sub(mel_pred, mel)))
                post_mel_loss = layers.mean(layers.abs(layers.elementwise_sub(postnet_pred, mel)))
                loss = mel_loss + post_mel_loss

                if cfg.use_data_parallel:
                    loss = model.scale_loss(loss)

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

                loss.backward()
                if cfg.use_data_parallel:
                    model.apply_collective_grads()
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
    main()