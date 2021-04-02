from paddle.io import Dataset
from pathlib import Path
import re
import pickle
import yaml
import tqdm
from parakeet.audio import AudioProcessor, LogMagnitude
import numpy as np
import multiprocessing as mp
from functools import partial

zh_pattern = re.compile("[\u4e00-\u9fa5]")

_tones = {'<pad>', '<s>', '</s>', '0', '1', '2', '3', '4', '5'}

_pauses = {'#1', '#2', '#3', '#4'}

_initials = {
    'b', 'p', 'm', 'f',
    'd', 't', 'n', 'l',
    'g', 'k', 'h',
    'j', 'q', 'x',
    'zh', 'ch', 'sh',
    'r',
    'z', 'c', 's',
}

_finals = {
    'ii', 'iii', 'a', 'o', 'e', 'ea', 'ai', 'ei', 'ao', 'ou', 'an', 'en', 'ang', 'eng', 'er',
    'i', 'ia', 'io', 'ie', 'iai', 'iao', 'iou', 'ian', 'ien', 'iang', 'ieng',
    'u', 'ua', 'uo', 'uai', 'uei', 'uan', 'uen', 'uang', 'ueng',
    'v', 've', 'van', 'ven', 'veng',
}   

_ernized_symbol = {'&r'}

_specials = {'<pad>', '<unk>', '<s>', '</s>'}

_phones = _initials | _finals | _ernized_symbol | _specials | _pauses

def is_zh(word):
    global zh_pattern
    match = zh_pattern.search(word)
    return match is not None


def ernized(syllable):
    return syllable[:2] != "er" and syllable[-2] == 'r'

def convert(syllable):
    # expansion of o -> uo
    syllable = re.sub(r"([bpmf])o$", r"\1uo", syllable)
    # syllable = syllable.replace("bo", "buo").replace("po", "puo").replace("mo", "muo").replace("fo", "fuo")
    # expansion for iong, ong
    syllable = syllable.replace("iong", "veng").replace("ong", "ueng")

    # expansion for ing, in
    syllable = syllable.replace("ing", "ieng").replace("in", "ien")
    
    # expansion for un, ui, iu
    syllable = syllable.replace("un", "uen").replace("ui", "uei").replace("iu", "iou")

    # rule for variants of i
    syllable = syllable.replace("zi", "zii").replace("ci", "cii").replace("si", "sii")\
        .replace("zhi", "zhiii").replace("chi", "chiii").replace("shi", "shiii")\
        .replace("ri", "riii")
    
    # rule for y preceding i, u
    syllable = syllable.replace("yi", "i").replace("yu", "v").replace("y", "i")

    # rule for w
    syllable = syllable.replace("wu", "u").replace("w", "u")

    # rule for v following j, q, x
    syllable = syllable.replace("ju", "jv").replace("qu", "qv").replace("xu", "xv")

    return syllable

def split_syllable(syllable:str):
    if syllable.startswith("#"):
        # syllable, tone
        return [syllable], ['0']

    tone = syllable[-1]
    syllable = convert(syllable[:-1])

    phones = []
    tones = []

    global _initials
    if syllable[:2] in _initials:
        phones.append(syllable[:2])
        tones.append('0')
        phones.append(syllable[2:])
        tones.append(tone)
    elif syllable[0] in _initials:
        phones.append(syllable[0])
        tones.append('0')
        phones.append(syllable[1:])
        tones.append(tone)
    else:
        phones.append(syllable)
        tones.append(tone)
    return phones, tones


def load_baker_transcription(text:str, pinyin:str):
    sentence_id, text = text.strip().split("\t")
    syllables = pinyin.strip().split()

    j = 0
    i = 0
    results = []
    while i < len(syllables) and j < len(text):
        if is_zh(text[j]):
            if not ernized(syllables[i]):
                results.append(syllables[i])
            else:
                results.append(syllables[i][:-2] + syllables[i][-1])
                results.append('&r5')
            j += 2 if ernized(syllables[i]) else 1
            i += 1
        elif text[j] == "#":
            results.append(text[j: j+2])
            j += 2
        else:
            j += 1
    
    if j < len(text):
        if text[j] == "#":
            results.append(text[j: j+2])
            j += 2
        else:
            j += 1

    phones = []
    tones = []
    for syllable in results:
        p, t = split_syllable(syllable)
        phones.extend(p)
        tones.extend(t)
    for p in phones:
        assert p in _phones, p
    return {"sentence_id": sentence_id, "text": text, "syllables": results, "phones": phones, "tones": tones}

def process_utterance(record, p, n, dataset_root, mel_dir):
    audio_path = (dataset_root / "Wave" / record["sentence_id"]).with_suffix(".wav")
    mel = p.mel_spectrogram(p.read_wav(str(audio_path)))
    mel = n.transform(mel)
    np.save(str(mel_dir / record["sentence_id"]), mel)

def process_baker(dataset_root, output_dir):
    dataset_root = Path(dataset_root).expanduser()
    output_dir = Path(output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    mel_dir = output_dir / "mel"
    mel_dir.mkdir(parents=True, exist_ok=True)

    p = AudioProcessor(22050, 1024, 1024, 256, f_max=8000)
    n = LogMagnitude(1e-5)
    prosody_label_path = dataset_root / "ProsodyLabeling" / "000001-010000.txt"
    with open(prosody_label_path, 'rt') as f: 
        lines = [line.strip() for line in f]
    
    records = []
    for i in range(0, len(lines), 2):
        records.append(
            (lines[i], lines[i+1])
        )
    
    processed_records = []
    for record in tqdm.tqdm(records):
        if 'Ｂ' in record[0] or 'P' in record[1]:
            continue
        new_record = load_baker_transcription(*record)
        processed_records.append(new_record)
        #print(new_record)

    with open(output_dir / "metadata.pickle", 'wb') as f: 
        pickle.dump(processed_records, f)
    
    with open(output_dir / "metadata.yaml", 'wt', encoding="utf-8") as f: 
        yaml.safe_dump(processed_records, f, default_flow_style=None, allow_unicode=True)
    
    print("metadata done!")
    
    func = partial(process_utterance, p=p, n=n, dataset_root=dataset_root, mel_dir=mel_dir)
    with mp.Pool(16) as pool:
        list(tqdm.tqdm(pool.imap(func, processed_records), desc="Baker", total=len(processed_records)))        



if __name__ == "__main__":
    process_baker("~/datasets/BZNSYP", "~/datasets/processed_BZNSYP")