# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
import sys
import torch

import numpy as np
from sklearn.mixture import GaussianMixture
sys.path.insert(1, "/home1/jesuraj/asr/gmmhubert/gmm-torch")
# from gmm import GaussianMixture

import joblib

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger("learn_gmm")


def get_gmm_model(
    n_components,
    covariance_type,
    max_iter,
    tol,
    n_init,
    init_params,
):
    return GaussianMixture(
        n_components=n_components,
        covariance_type=covariance_type,
        max_iter=max_iter,
        tol=tol,
        n_init=n_init,
        init_params=init_params,
        verbose=1,
        reg_covar=1e-3,
        warm_start=False,
        random_state=None,
    )
    
    # return GaussianMixture(
    #     n_components=n_components,
    #     n_features=40,
    #     covariance_type=covariance_type,
    #     init_params=init_params,
    # )


def load_feature_shard(feat_dir, split, nshard, rank, percent):
    feat_path = f"{feat_dir}/{split}_{rank}_{nshard}.npy"
    leng_path = f"{feat_dir}/{split}_{rank}_{nshard}.len"
    with open(leng_path, "r") as f:
        lengs = [int(line.rstrip()) for line in f]
        offsets = [0] + np.cumsum(lengs[:-1]).tolist()

    if percent < 0:
        return np.load(feat_path, mmap_mode="r")
    else:
        nsample = int(np.ceil(len(lengs) * percent))
        indices = np.random.choice(len(lengs), nsample, replace=False)
        feat = np.load(feat_path, mmap_mode="r")
        sampled_feat = np.concatenate(
            [feat[offsets[i]: offsets[i] + lengs[i]] for i in indices], axis=0
        )
        logger.info(
            (
                f"sampled {nsample} utterances, {len(sampled_feat)} frames "
                f"from shard {rank}/{nshard}"
            )
        )
        return sampled_feat


def load_feature(feat_dir, split, nshard, seed, percent):
    assert percent <= 1.0
    feat = np.concatenate(
        [
            load_feature_shard(feat_dir, split, nshard, r, percent)
            for r in range(nshard)
        ],
        axis=0,
    )
    logging.info(f"loaded feature with dimension {feat.shape}")
    return feat


def learn_gmm(
    feat_dir,
    split,
    nshard,
    gmm_path,
    n_components,
    seed,
    percent,
    init,
    max_iter,
    tol,
    n_init,
    covariance_type,
    device,
):
    np.random.seed(seed)
    feat = load_feature(feat_dir, split, nshard, seed, percent)
    # feat = torch.from_numpy(feat).float().to(device)
    gmm_model = get_gmm_model(
        n_components,
        covariance_type,
        max_iter,
        tol,
        n_init,
        init,
    )
    gmm_model.fit(feat)
    # gmm_model.fit(feat, n_iter=max_iter, delta=tol)
    joblib.dump(gmm_model, gmm_path)

    # inertia = gmm_model.score_samples(feat) / len(feat)
    inertia = gmm_model.score(feat)
    logger.info("total log-likelihood: %.5f", inertia)
    logger.info("finished successfully")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("feat_dir", type=str)
    parser.add_argument("split", type=str)
    parser.add_argument("nshard", type=int)
    parser.add_argument("gmm_path", type=str)
    parser.add_argument("n_components", type=int)
    parser.add_argument("--covariance_type", default='diag', type=str)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument(
        "--percent", default=-1, type=float, help="sample a subset; -1 for all"
    )
    parser.add_argument("--init", default="k-means++")
    parser.add_argument("--max_iter", default=100, type=int)
    parser.add_argument("--tol", default=0.0, type=float)
    parser.add_argument("--n_init", default=20, type=int)
    parser.add_argument("--device", default="cpu", type=str)
    args = parser.parse_args()
    logging.info(str(args))

    learn_gmm(**vars(args))
