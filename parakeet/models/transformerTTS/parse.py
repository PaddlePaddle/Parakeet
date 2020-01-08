import jsonargparse

def add_config_options_to_parser(parser):
    parser.add_argument('--audio.num_mels', type=int, default=80,
        help="the number of mel bands when calculating mel spectrograms.")
    parser.add_argument('--audio.n_fft', type=int, default=2048,
        help="the number of fft components.")
    parser.add_argument('--audio.sr', type=int, default=22050,
        help="the sampling rate of audio data file.")
    parser.add_argument('--audio.preemphasis', type=float, default=0.97,
        help="the preemphasis coefficient.")
    parser.add_argument('--audio.hop_length', type=int, default=128,
        help="the number of samples to advance between frames.")
    parser.add_argument('--audio.win_length', type=int, default=1024,
        help="the length (width) of the window function.")
    parser.add_argument('--audio.power', type=float, default=1.4,
        help="the power to raise before griffin-lim.")
    parser.add_argument('--audio.min_level_db', type=int, default=-100,
        help="the minimum level db.")
    parser.add_argument('--audio.ref_level_db', type=int, default=20,
        help="the reference level db.")
    parser.add_argument('--audio.outputs_per_step', type=int, default=1,
        help="the outputs per step.")
    
    parser.add_argument('--hidden_size', type=int, default=256,
        help="the hidden size in network.")
    parser.add_argument('--embedding_size', type=int, default=512,
        help="the embedding vector size.")

    parser.add_argument('--warm_up_step', type=int, default=4000,
        help="the warm up step of learning rate.")
    parser.add_argument('--grad_clip_thresh', type=float, default=1.0,
        help="the threshold of grad clip.")
    parser.add_argument('--batch_size', type=int, default=32,
        help="batch size for training.")
    parser.add_argument('--epochs', type=int, default=10000,
        help="the number of epoch for training.")
    parser.add_argument('--lr', type=float, default=0.001,
        help="the learning rate for training.")
    parser.add_argument('--save_step', type=int, default=500,
        help="checkpointing interval during training.")
    parser.add_argument('--image_step', type=int, default=2000,
        help="attention image interval during training.")
    parser.add_argument('--max_len', type=int, default=400,
        help="The max length of audio when synthsis.")
    parser.add_argument('--transformer_step', type=int, default=160000,
        help="Global step to restore checkpoint of transformer in synthesis.")
    parser.add_argument('--postnet_step', type=int, default=100000,
        help="Global step to restore checkpoint of postnet in synthesis.")
    parser.add_argument('--use_gpu', type=bool, default=True,
        help="use gpu or not during training.")
    parser.add_argument('--use_data_parallel', type=bool, default=False,
        help="use data parallel or not during training.")

    parser.add_argument('--data_path', type=str, default='./dataset/LJSpeech-1.1',
        help="the path of dataset.")
    parser.add_argument('--checkpoint_path', type=str, default=None,
        help="the path to load checkpoint or pretrain model.")
    parser.add_argument('--save_path', type=str, default='./checkpoint',
        help="the path to save checkpoint.")
    parser.add_argument('--log_dir', type=str, default='./log',
        help="the directory to save tensorboard log.")
    parser.add_argument('--sample_path', type=str, default='./log',
        help="the directory to save audio sample in synthesis.")
    

    parser.add_argument('-c', '--config', action=jsonargparse.ActionConfigFile)
