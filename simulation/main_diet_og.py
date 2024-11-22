import argparse
from abc import ABC, abstractmethod
import wandb
import numpy as np
import torch
import torch.nn.functional as F
import random
import os
import losses
import spaces
import disentanglement_utils
import invertible_network_utils
import latent_spaces
import encoders
import csv


DEVICE = ("cuda" if torch.cuda.is_available() else "cpu")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Disentanglement with DIET - MLP Mixing"
    )

    ############################
    #### Misc args #############
    ############################

    parser.add_argument("--seed", type=int, default=None)

    parser.add_argument("--nr-indices", type=int, default=10000,
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


    # This would correspond to the conditional distribution of sample latents around the cluster vectors.
    # Currently, we only consider the von Mises-Fisher distribution.
    # If added, change description of c-param.
    # parser.add_argument(
    #     "--c-p",
    #     type=int,
    #     default=0,
    #     help="Exponent of conditional Lp Exponential distribution. c_p=0 means von Mises-Fisher.",
    # )

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


    parser.add_argument("--batch-size", type=int, default=1000)
    parser.add_argument("--n-log-steps", type=int, default=1000)
    # parser.add_argument("--n-steps", type=int, default=10000)
    parser.add_argument("--n-epochs", type=int, default=50000)


    ############################
    #### Evaluation args #######
    ############################

    ############################
    #### Logging args ##########
    ############################

    parser.add_argument(
        "--save_folder", type=str, default="/cluster/home/callen/projects/cl-ica-sup/outputs", help="Dimensionality of the latents and observed variables. "
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

    # # Conditional distribution of sample latents around cluster vectors.
    # sample_conditional = (
    #     lambda space, z, size, device=DEVICE: space.von_mises_fisher(
    #         z, args.c_param, size, device)
    # )

    if args.c_p != 0:
        if args.c_p == 1:
            sample_conditional = (
                lambda space, z, size, device=DEVICE: space.laplace(
                z, 1.0, size, device)
            )
        elif args.c_p == 2:
            sample_conditional = (
                lambda space, z, size, device=DEVICE: space.normal(
                z, 1.0, size, device)
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
        # generate a batch of random indices
        indices = torch.randint(0, args.nr_indices, (size,), device=device)
        # map these indices to nr_clusters cluster vectors based on equal bins
        cluster_indices = indices % args.nr_clusters
        corresponding_cluster_vectors = cluster_vectors[cluster_indices]
        # sample latents around the cluster vectors
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


def test_disentanglement(sample_function, cluster_vectors, h, h2=None, n_samples=4096, return_linear_model=False):
    _h2 = h2 if h2 is not None else lambda z: z
    z_disentanglement, _, _ = sample_function(n_samples,cluster_vectors)
    return_tuple = disentanglement_utils.linear_disentanglement(
        _h2(z_disentanglement), h(z_disentanglement), mode="r2", train_test_split=True,
    )
    (linear_disentanglement_score, _), _ = return_tuple
    return linear_disentanglement_score

def train_diet(args, cluster_vectors, sample_latents, g, f, logger):
    # create extra linear layer with or without bias on top of the encoder
    W = torch.nn.Linear(args.n, args.nr_clusters, bias=args.W_bias_used, device=DEVICE)

    optimizer_f = torch.optim.Adam(list(f.parameters())+list(W.parameters()), lr=1e-3)

    losses = []
    linear_disentanglement_scores = []

    f.train()
    step = 0
    for _ in range(args.n_epochs):
        for _ in range(args.nr_indices//args.batch_size):
            step += 1
            zgt, _, indices = sample_latents(args.batch_size)
            if args.normalize_rows_W:
                W.weight.data = W.weight.data/torch.norm(W.weight.data,p=2,dim=-1,keepdim=True)
            zinf = W(f(g(zgt)))

            # compute cross-entropy loss
            loss = F.cross_entropy(zinf, indices.to(DEVICE))

            optimizer_f.zero_grad()
            loss.backward()
            optimizer_f.step()

            losses.append((step, loss))
            if step % args.n_log_steps == 0 or step == (args.n_epochs*(args.nr_indices//args.batch_size)) - 1:
                f.eval()
                linear_disentanglement_score = test_disentanglement(sample_function=sample_latents, cluster_vectors=cluster_vectors, h=lambda z: f(g(z)))
                linear_disentanglement_scores.append((step, linear_disentanglement_score))

                f.train()
                logger.update(step, {'loss': loss, 'lin. disentanglement': linear_disentanglement_score,})

    return losses, linear_disentanglement_scores, W


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
    tag=f"supervised_seed_{args.seed}_nsamp_{args.nr_indices}_n_{args.n}_nclusters_{args.nr_clusters}_epochs_{args.n_epochs}_mp_{args.m_p}_mparam_{args.m_param}_cp_{args.c_p}_cparam_{args.c_param}_{args.sphere_norm}_{args.normalize_rows_W}"
    args.save_folder = os.path.join(args.save_folder,tag)
    os.makedirs(args.save_folder, exist_ok = True)

    print("Device in use: ", DEVICE)
    cluster_vectors, sample_latents, g = create_generative_process(args)
    print("g: ", g)

    # Test
    linear_disentanglement_score = test_disentanglement(sample_function=sample_latents, cluster_vectors=cluster_vectors, h=g)
    print("Initial linear disentanglement score of g: ", linear_disentanglement_score)

    f = create_encoder(args)

    losses, linear_disentanglement_scores, W = train_diet(args,cluster_vectors, sample_latents, g, f, StdoutLogger())

    # Saving 
    with open(os.path.join(args.save_folder,f'performance_linear_{tag}.csv'), 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['step','seed','nb indices', 'batch size', 'nb clusters', 'n', 'm', 'm param', 'c', 'c param', 'z norm', 'w bias', 'w norm', 'final loss', 'epochs', 'z2z r2'])
        for _, ((step,v_1)) in enumerate(linear_disentanglement_scores):
            writer.writerow([step,args.seed,args.nr_indices,args.batch_size,args.nr_clusters,args.n,args.m_p,args.m_param,args.c_p,args.c_param,args.sphere_norm,args.W_bias_used,args.normalize_rows_W,losses[-1][1].item(),args.n_epochs,v_1])  
    torch.save({"encoder_state_dict":f.state_dict(),
                "w":W},os.path.join(args.save_folder,"model.pt"))
    
if __name__ == "__main__":
    main()
