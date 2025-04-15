# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
import sys

import soundfile as sf
import torch
import torchaudio

from feature_utils import get_path_iterator, dump_feature
from fairseq.data.audio.audio_utils import get_features_or_waveform
from tqdm import tqdm
import numpy as np

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger("dump_mfcc_feature")


class MelFeatureReader(object):
    def __init__(self, sample_rate):
        self.sample_rate = sample_rate




    def get_feats(self, path, mean_std, ref_len=None):
    
        waveform, sr = torchaudio.load(path)
        waveform = waveform*(2**15)
        y = torchaudio.compliance.kaldi.fbank(
                            waveform,
                            num_mel_bins=40,
                            sample_frequency=16000,
                            window_type='hamming',
                            frame_length=25,
                            frame_shift=10)
        # Normalize by the mean and std of Librispeech
        mean_std = torch.from_numpy(mean_std)
        mean = mean_std[0]
        std = mean_std[1]
        mean = mean.to(y.device, dtype=torch.float32)
        std = std.to(y.device, dtype=torch.float32)
        y = (y-mean)/std
            
        return y


    
    

def main(tsv_dir, split, nshard, rank, feat_dir, sample_rate):
    reader = MelFeatureReader(sample_rate)
    generator, num = get_path_iterator(f"{tsv_dir}/{split}.tsv", nshard, rank)
    dump_feature(reader, generator, num, split, nshard, rank, feat_dir, tsv_dir)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("tsv_dir")
    parser.add_argument("split")
    parser.add_argument("nshard", type=int)
    parser.add_argument("rank", type=int)
    parser.add_argument("feat_dir")
    parser.add_argument("--sample_rate", type=int, default=16000)
    args = parser.parse_args()
    logger.info(args)

    main(**vars(args))
