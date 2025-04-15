# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
import sys

import tqdm
from npy_append_array import NpyAppendArray
import numpy as np
import torchaudio
import torch


logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger("feature_utils")


def get_shard_range(tot, nshard, rank):
    assert rank < nshard and rank >= 0, f"invaid rank/nshard {rank}/{nshard}"
    start = round(tot / nshard * rank)
    end = round(tot / nshard * (rank + 1))
    assert start < end, f"start={start}, end={end}"
    logger.info(
        f"rank {rank} of {nshard}, process {end-start} "
        f"({start}-{end}) out of {tot}"
    )
    return start, end


def get_path_iterator(tsv, nshard, rank):
    with open(tsv, "r") as f:
        root = f.readline().rstrip()
        lines = [line.rstrip() for line in f]
        start, end = get_shard_range(len(lines), nshard, rank)
        lines = lines[start:end]
        def iterate():
            for line in lines:
                subpath, nsample = line.split("\t")
                yield f"{root}/{subpath}", int(nsample)
    return iterate, len(lines)


def dump_feature(reader, generator, num, split, nshard, rank, feat_dir, tsv_path):
    iterator = generator()

    feat_path = f"{feat_dir}/{split}_{rank}_{nshard}.npy"
    leng_path = f"{feat_dir}/{split}_{rank}_{nshard}.len"

    os.makedirs(feat_dir, exist_ok=True)
    if os.path.exists(feat_path):
        os.remove(feat_path)

    feat_f = NpyAppendArray(feat_path)
    
    mean_std = get_mean_std(feat_dir, split, tsv_path)
    with open(leng_path, "w") as leng_f:
        for path, nsample in tqdm.tqdm(iterator, total=num):
            feat = reader.get_feats(path, mean_std, nsample)
            feat_f.append(feat.cpu().numpy())
            leng_f.write(f"{len(feat)}\n")
    logger.info("finished successfully")
    
    
def get_mean_std(feat_dir, split, tsv_path, mel_dim=40, ):
        if os.path.exists(os.path.join(feat_dir, "mean_std.npy")):
            mean_std = np.load(os.path.join(feat_dir, "mean_std.npy"))
            return mean_std
        
        if split != "train":
            raise ValueError("mean_std only for train split")
        
        file_pth = []
        with open(os.path.join(tsv_path, split + ".tsv"), 'r') as fp:
            root_path = fp.readline().strip()
            for x in fp:
                file_pth.append(x.strip().split()[0])
        file_pth = [os.path.join(root_path, x) for x in file_pth]

        sum_ = np.zeros((1,mel_dim))
        sum_square = np.zeros((1,mel_dim))
        total_count = 0 

        for pth in tqdm.tqdm(file_pth):
            feat = get_mel_feats(pth, mel_dim=mel_dim)
            sum_ += np.sum(feat, axis=0)
            sum_square += np.sum(feat**2, axis=0)
            total_count += len(feat)
                
        mean = sum_ / total_count
        std = ((sum_square/total_count)-(mean**2))**(1/2)
        mean_std = np.concatenate((mean, std), axis=0)
        np.save(os.path.join(feat_dir, "mean_std.npy"), mean_std)
        return mean_std
    
def read_audio(path, ref_len=None):
    wav, sr = torchaudio.load(path)
    wav = wav*(2**15)
    assert sr == 16000, sr
        
    if ref_len is not None and abs(ref_len - len(wav[0])) > 160:
        logging.warning(f"ref {ref_len} != read {len(wav[0])} ({path})")

    return wav


def get_mel_feats(path, mel_dim=40, ref_len=None):
    x = read_audio(path, ref_len)
    with torch.no_grad():
        y = torchaudio.compliance.kaldi.fbank(
                x,
                num_mel_bins=mel_dim,
                sample_frequency=16000,
                window_type='hamming',
                frame_length=25,
                frame_shift=10,
        )
        y = y.contiguous()
    y = np.array(y)

    return y


