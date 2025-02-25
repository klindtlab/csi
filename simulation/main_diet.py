"""
Module Name: main_diet.py
Author: Alice Bizeul, Attila Juhos, adapted from https://github.com/brendel-group/cl-ica
Description: Main script.
"""
import argparse
from abc import ABC, abstractmethod
import numpy as np
import torch
import torch.nn.functional as F
import random
import os
import csv
import plotly.graph_objects as go

import src.spaces
import src.disentanglement_utils
import src.invertible_network_utils
import src.latent_spaces
import src.encoders


DEVICE = ("cuda" if torch.cuda.is_available() else "cpu")

def parse_args():
    parser = argparse.ArgumentParser(
        description="Disentanglement with DIET - MLP Mixing"
    )

    ############################
    #### Misc args #############
    ############################

    parser.add_argument("--seed", type=int, default=None)

    parser.add_argument("--nr-indices", type=int, default=10240,
                        help="Number of indices and, therefore, number of rows of matrix W.")

    ############################
    #### Latent args ###########
    ############################

    parser.add_argument(
        "--n", type=int, default=5, help="Dimensionality of the latents and observed variables. "
        "Potential values to start with: 5, 10."
    )

    parser.add_argument(
        "--nr-clusters", type=int, default=100, help="Number of cluster vectors.")

    parser.add_argument(
        "--m-p",
        type=int,
        default=0,
        help="Type of distribution of cluster vectors (all coming from the sphere). p=0 means uniform on the sphere, "
        "all other p values correspond to (projected) L_p Exponential around the first unit vector",
    )

    parser.add_argument(
        "--m-param",
        type=float,
        default=1.0,
        help="Additional parameter for the marginal (only relevant if it is not uniform) to quantify the concentration.",
    )

    parser.add_argument(
        "--c-p",
        type=int,
        default=0,
        help="Type of distribution of cluster vectors (all coming from the sphere). p=0 means uniform on the sphere, "
        "all other p values correspond to (projected) L_p Exponential around the first unit vector",
    )

    parser.add_argument(
        "--c-param",
        type=float,
        default=10,
        help="Concentration parameter of the vMF conditional distribution around cluster vectors."
        "Working values start around 5 and can go up to high double digits.",
    )
    parser.add_argument(
        "--beta-param",
        type=float,
        default=2,
        help="Concentration parameter of the vMF conditional distribution around cluster vectors."
        "Working values start around 5 and can go up to high double digits.",
    )

    ############################
    #### Generator args ########
    ############################

    parser.add_argument(
        "--n-mixing-layer",
        type=int,
        default=4,
        help="Number of layers in nonlinear mixing network g.",
    )

    ############################
    #### Model args ############
    ############################

    parser.add_argument(
        "--n-hidden-layers",
        type=int,
        default=6,
        help="Number of hidden layers in the encoder. Minimum 2.",
    )

    parser.add_argument(
        "--sphere-norm", action="store_true", help="Normalize output to a sphere."
    )

    parser.add_argument(
        "--no-batch-norm", action="store_true", help="Do not use batch normalization in the encoder.")


    ############################
    #### Training args #########
    ############################

    parser.add_argument("--W-bias-used", action="store_true", help="Use bias in W matrix.")
    parser.add_argument("--normalize-rows-W", action="store_true", help="Normalize rows of W matrix.")
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--n-log-steps", type=int, default=1000)
    parser.add_argument("--n-epochs", type=int, default=50000)

    ############################
    #### Logging args ##########
    ############################

    parser.add_argument(
        "--save_folder", type=str, default="./selfsup", help="Dimensionality of the latents and observed variables. "
        "Potential values to start with: 5, 10."
    )


    args = parser.parse_args()

    print("Arguments:")
    for k, v in vars(args).items():
        print(f"\t{k}: {v}")

    return args


def create_generative_process(args):
    # Geometric space of both cluster and sample latents.
    space = spaces.NSphereSpace(args.n)

    # Distribution of cluster vectors.
    eta = torch.zeros(args.n)
    eta[0] = 1.0

    if args.m_p != 0:
        if args.m_p == 1:
            sample_marginal = lambda space, size, device=DEVICE: space.laplace(
                eta, 1.0, size, device
            )
        elif args.m_p == 2:
            sample_marginal = lambda space, size, device=DEVICE: space.normal(
                eta, 1.0, size, device
            )
        else:
            sample_marginal = (
                lambda space, size, device=DEVICE: space.generalized_normal(
                    eta, args.m_param, p=args.m_p, size=size, device=device
                )
            )
    else:
        sample_marginal = lambda space, size, device=DEVICE: space.uniform(
            size, device=device
        )


    if args.c_p != 0:
        if args.c_p == 1:
            sample_conditional = (
                lambda space, z, size, device=DEVICE: space.laplace(
                z, args.c_param/10, size, device)
            )
        elif args.c_p == 2:
            sample_conditional = (
                lambda space, z, size, device=DEVICE: space.normal(
                z, args.c_param/10, size, device)
            )
        elif args.c_p == 3:
            sample_conditional = (
                lambda space, z, size, device=DEVICE: space.truncated_laplace(
                z, args.c_param/10, -args.beta_param, args.beta_param,size, device)
            )
        elif args.c_p == 4:
            sample_conditional = (
                lambda space, z, size, device=DEVICE: space.generalized_normal(
                z, args.c_param/10, args.beta_param,size, device)
            )
        else:
            raise NotImplementedError
    else:
        sample_conditional = (
            lambda space, z, size, device=DEVICE: space.von_mises_fisher(
            z, args.c_param, size, device)
        )

    sample_cluster_vectors = (
        lambda nr_clusters, device=DEVICE: sample_marginal(space, size=nr_clusters, device=device)
    )
    cluster_vectors = sample_cluster_vectors(args.nr_clusters)

    def sample_latents(size, cluster_vectors=cluster_vectors, device=DEVICE):
        indices = torch.randint(0, args.nr_indices, (size,), device=device)
        cluster_indices = indices % args.nr_clusters
        corresponding_cluster_vectors = cluster_vectors[cluster_indices]
        samples = sample_conditional(space, corresponding_cluster_vectors, size, device=device)

        return samples, indices, cluster_indices

    g = invertible_network_utils.construct_invertible_mlp(
        n=args.n,
        n_layers=args.n_mixing_layer,
        act_fct="leaky_relu",
        cond_thresh_ratio=0.0,
        n_iter_cond_thresh=25000,
    )
    g = g.to(DEVICE)

    for p in g.parameters():
        p.requires_grad = False

    return cluster_vectors, sample_latents, g

def create_encoder(args):
    if args.sphere_norm:
        output_normalization = "fixed_sphere"
    else:
        output_normalization = None

    if args.n_hidden_layers < 2:
        raise ValueError("n_hidden_layers < 2: Encoder must have at least 2 hidden layers")

    layers = [args.n * 10] + [args.n * 50] * (args.n_hidden_layers - 2) + [args.n * 10]

    return encoders.get_mlp(
        n_in=args.n,
        n_out=args.n,
        layers=layers,
        layer_normalization="bn" if not args.no_batch_norm else None,
        output_normalization=output_normalization,
    ).to(DEVICE)


def test_disentanglement(sample_function, cluster_vectors, h, h2=None, n_samples=8192, return_linear_model=False):
    _h2 = h2 if h2 is not None else lambda z: z

    z_disentanglement, _, _ = sample_function(n_samples,cluster_vectors)

    return_tuple = disentanglement_utils.linear_disentanglement(
        _h2(z_disentanglement), h(z_disentanglement), mode="r2", train_test_split=True,
    )

    (linear_disentanglement_score, _), _ = return_tuple
    return linear_disentanglement_score

def test_disentanglement_W(cluster_vectors, h, n_samples=4096, return_linear_model=False):

    cluster_index=torch.arange(h.shape[0])% cluster_vectors.shape[0]

    return_tuple = disentanglement_utils.linear_disentanglement(
        cluster_vectors[cluster_index], h, mode="r2", train_test_split=True, 
    )
    
    (linear_disentanglement_score, _), _ = return_tuple
    return linear_disentanglement_score


def test_disentanglement_ortho(sample_function, cluster_vectors, h, h2=None, n_samples=4096, concentration=50, return_linear_model=False):
    _h2 = h2 if h2 is not None else lambda z: z

    z_disentanglement, _, _ = sample_function(n_samples,cluster_vectors)

    return_tuple = disentanglement_utils.orthogonal_linear_disentanglement(
        _h2(z_disentanglement), h(z_disentanglement), mode="mae", train_test_split=True, scaler=concentration  
    )

    (linear_disentanglement_ortho,_), _ = return_tuple

    return_tuple = disentanglement_utils.orthogonal_linear_disentanglement_perf(
        _h2(z_disentanglement), h(z_disentanglement), mode="r2", train_test_split=True
    )

    (linear_disentanglement_score,_), _ = return_tuple
    return linear_disentanglement_score, linear_disentanglement_ortho

def test_disentanglement_W_ortho(cluster_vectors, h, n_samples=4096, return_linear_model=False):

    cluster_index=torch.arange(h.shape[0])% cluster_vectors.shape[0]

    return_tuple = disentanglement_utils.orthogonal_linear_disentanglement(
        cluster_vectors[cluster_index], h, mode="mae", train_test_split=True, scaler=1.0
    )
    
    ((linear_disentanglement_ortho,_)), _ = return_tuple

    return_tuple = disentanglement_utils.orthogonal_linear_disentanglement_perf(
        cluster_vectors[cluster_index], h, mode="r2", train_test_split=True
    )
    
    ((linear_disentanglement_score,_)), _ = return_tuple
    return linear_disentanglement_score, linear_disentanglement_ortho

def train_diet(args, cluster_vectors, sample_latents, g, f, logger):

    W = torch.nn.Linear(args.n, args.nr_indices, bias=args.W_bias_used, device=DEVICE)

    optimizer_f = torch.optim.Adam(list(f.parameters())+list(W.parameters()), lr=1e-3, weight_decay=0)

    losses = []
    linear_disentanglement_scores, linear_disentanglement_orthos, linear_disentanglement_scores_zv, linear_disentanglement_scores_W, linear_disentanglement_orthos_W = [], [], [], [], []

    f.train()
    step = 0
    for _ in range(args.n_epochs):
        max_steps = max(args.nr_indices//args.batch_size,1)
        for _ in range(max_steps):
            step += 1
            zgt, indices, _ = sample_latents(args.batch_size)
            if args.normalize_rows_W:
                W.weight.data = W.weight.data/torch.norm(W.weight.data,p=2,dim=-1,keepdim=True)

            zinf = W(f(g(zgt)))

            loss = F.cross_entropy(zinf, indices)

            optimizer_f.zero_grad()
            loss.backward()
            optimizer_f.step()

            losses.append((step, loss))
            if step % args.n_log_steps == 0 or step == (args.n_epochs*(args.nr_indices//args.batch_size)) - 1:
                f.eval()

                # z -> z_gt
                if args.normalize_rows_W:
                    linear_disentanglement_score, linear_disentanglement_ortho = test_disentanglement_ortho(sample_function=sample_latents, cluster_vectors=cluster_vectors, h=lambda z: f(g(z)), concentration=args.c_param)
                else:
                    linear_disentanglement_score = test_disentanglement(sample_function=sample_latents, cluster_vectors=cluster_vectors, h=lambda z: f(g(z)))
                    linear_disentanglement_ortho = 0

                linear_disentanglement_scores.append((step, linear_disentanglement_score))
                linear_disentanglement_orthos.append((step, linear_disentanglement_ortho))

                # W -> c
                if args.normalize_rows_W:
                    linear_disentanglement_score_W, linear_disentanglement_ortho_W = test_disentanglement_W_ortho(cluster_vectors=cluster_vectors,h=W.weight.data)
                else:
                    linear_disentanglement_score_W = test_disentanglement_W(cluster_vectors=cluster_vectors,h=W.weight.data)
                    linear_disentanglement_ortho_W = 0

                linear_disentanglement_scores_W.append((step, linear_disentanglement_score_W))
                linear_disentanglement_orthos_W.append((step, linear_disentanglement_ortho_W))

                f.train()
                logger.update(step, {'loss': loss, 'lin. disentanglement': linear_disentanglement_score, 'lin. disentanglement ortho': linear_disentanglement_ortho, 'lin. disentanglement_W': linear_disentanglement_score_W, 'lin. disentanglement_W ortho': linear_disentanglement_ortho_W})

    return losses, linear_disentanglement_scores, linear_disentanglement_orthos, linear_disentanglement_scores_W, linear_disentanglement_orthos_W, W


class Logger(ABC):
    @abstractmethod
    def update(self, step, metrics: dict):
        pass


class StdoutLogger(Logger):
    def __init__(self, fixed_msg=None):
        self._fixed_msg = fixed_msg if fixed_msg is not None else ""

    def update(self, step, metrics: dict):
        message = self._fixed_msg + f"Step: {step} \t"
        for key, value in metrics.items():
            if isinstance(value, list):
                value = [f"{v:.4f}" for v in value]
                value = ", ".join(value)
                message += f"{key}: [{value}] \t"
            else:
                message += f"{key}: {value:.4f} \t"
        print(message)


def main():
    args = parse_args()
    if args.seed is not None:
        np.random.seed(args.seed)
        random.seed(args.seed)
        torch.manual_seed(args.seed)

    tag=f"seed_{args.seed}_nsamp_{args.nr_indices}_n_{args.n}_nclusters_{args.nr_clusters}_epochs_{args.n_epochs}_mp_{args.m_p}_mparam_{args.m_param}_cp_{args.c_p}_cparam_{args.c_param}_batch_{args.batch_size}_{args.sphere_norm}_{args.normalize_rows_W}"
    args.save_folder = os.path.join(args.save_folder,tag)
    os.makedirs(args.save_folder, exist_ok = True)

    cluster_vectors, sample_latents, g = create_generative_process(args)

    if args.normalize_rows_W:
        linear_disentanglement_score = test_disentanglement_ortho(sample_function=sample_latents, cluster_vectors=cluster_vectors, h=g)
    else:
        linear_disentanglement_score = test_disentanglement(sample_function=sample_latents, cluster_vectors=cluster_vectors, h=g)
    print("Initial linear disentanglement score of g: ", linear_disentanglement_score,flush=True)

    f = create_encoder(args)
    losses, linear_disentanglement_scores, linear_disentanglement_orthos, linear_disentanglement_scores_W, linear_disentanglement_orthos_W, W = train_diet(args,cluster_vectors, sample_latents, g, f, StdoutLogger())

    # Saving 
    with open(os.path.join(args.save_folder,f'performance_linear_{tag}.csv'), 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['zv','step','seed','nb indices', 'batch size', 'nb clusters', 'n', 'm', 'm param', 'c', 'c param', 'z norm', 'w bias', 'w norm', 'final loss', 'epochs', 'z2z r2', 'z2z ortho','Wz2v vectors', 'W2v r2', 'W2v ortho'])
        for _, ((step,v_1),(step,v_2),(step,v_4),(step,v_5)) in enumerate(zip(linear_disentanglement_scores, linear_disentanglement_scores_W,linear_disentanglement_orthos, linear_disentanglement_orthos_W)):
            writer.writerow([True,step,args.seed,args.nr_indices,args.batch_size,args.nr_clusters,args.n,args.m_p,args.m_param,args.c_p,args.c_param,args.sphere_norm,args.W_bias_used,args.normalize_rows_W,losses[-1][1].item(),args.n_epochs,v_1,v_2,v_4,v_5])  
    torch.save({"encoder_state_dict":f.state_dict(),
                "w":W},os.path.join(args.save_folder,"model.pt"))
    
if __name__ == "__main__":
    main()
