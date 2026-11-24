# Bounded-tree implementation

The three scientific Python files in `experiment/` are preserved without changes from the corrected package used in the successful small-sample SCC runs. Their hashes match the execution metadata for all 60 archived pairs at n=500,1000,1500. The wrappers organize tasks, record the execution environment, validate saved results, and construct tables. The older n=2000 result records have the same scientific settings but lack execution-time source hashes.

## Distribution and sampling

The target has support $[0,1]^{31}$. In one-based indexing, node $i\geq 2$ has parent $\pi(i)=\lfloor i/2\rfloor$. Its log potential is

$$V(y)=\sum_{j=1}^{31}c_j\sin(2\pi y_j+\varphi_j)+\sum_{i=2}^{31}\left[a_i s_i\cos\{2\pi(y_{\pi(i)}-y_i)\}+b_i\sin(2\pi y_{\pi(i)})\sin(2\pi y_i)\right].$$

The nominal continuous density is proportional to $\exp V(y)$. NumPy's generator with seed 11 draws all coefficients once: $a_i=0.55U_i$ with $U_i\sim\operatorname{Unif}(0.7,1.3)$, $b_i=0.55W_i$ with $W_i\sim\operatorname{Unif}(0.4,0.8)$, $s_i\in\{-1,1\}$ equiprobably, $c_j=0.3C_j$ with $C_j\sim\operatorname{Unif}(0.5,1.5)$, and $\varphi_j\sim\operatorname{Unif}(0,2\pi)$. The source draws arrays of length 31 for the edge coefficients; their root entries are unused. This draw order is preserved.

For simulation, each coordinate has a grid of 200 midpoints. Tree message passing computes the grid normalizer and node/edge marginals. Ancestral sampling selects grid cells, followed by independent uniform jitter within each selected cell. This gives exact samples from the corresponding piecewise constant grid approximation, not exact samples from the smooth continuous density. The same approximation is used for the training sample, reference sample, and marginal-TV truth. The smooth potential $V$ is evaluated at the actual sample coordinates for the likelihood-gap diagnostic.

Training seeds are 0 through 19 at each sample size, 500, 1000, 1500, and 2000. Each method regenerates the same training sample for a given $(n,\mathrm{seed})$. The 20,000-point evaluation reference uses seed 12345 and is shared across replications. Reusing a seed at different sample sizes does not imply that all training samples are nested: the ancestral sampling routine consumes random numbers in a sample-size-dependent order.

## CINDES

The classifier is a fully connected network with three hidden layers of 128 ReLU units and a scalar output (37,249 trainable parameters). It distinguishes the $n$ observations from $2n$ independent uniform observations on $[0,1]^{31}$. These uniform observations are drawn once before training. The minibatch binary cross-entropy uses the classifier output truncated to $[-25,25]$. With a negative-to-positive ratio of two, the population logit includes an additive $-\log 2$ term; an additive constant cancels in the subsequent normalized importance weights.

For $m_t=e^{-t}$ and $\sigma_t^2=1-e^{-2t}$, each score evaluation draws $K=5000$ independent particles from the coordinatewise normal distribution with mean $y/m_t$ and variance $(\sigma_t/m_t)^2$, restricted to $[0,1]^{31}$. With these particles $Z_k$, the estimate is

$$\widehat s(y,t)=-\frac{1}{\sigma_t^2}\sum_{k=1}^{K}w_k(y-m_tZ_k),\qquad w_k=\frac{\exp\{\widehat f_R(Z_k)\}}{\sum_{\ell=1}^{K}\exp\{\widehat f_R(Z_\ell)\}}.$$

The proposal is evaluated by inverse-CDF sampling. The preserved implementation clips normal-CDF probabilities to $[10^{-6},1-10^{-6}]$ and then clips the particle coordinates to $[0,1]$. These numerical safeguards can alter the ideal truncated-normal proposal in extreme tails. They remain in the reproduction code; this package does not silently replace the sampler with a different numerical procedure. The particles are not antithetic.

The estimator does not receive the tree edges or the potential functions. It does use the known bounded support when drawing reference observations, constructing the proposal, and projecting generated observations onto the support.

## DSM and optimization

DSM directly estimates the score with three hidden layers of 128 ReLU units. Its input contains the 31 noisy coordinates and 16 time features (eight sine/cosine frequency pairs); its output has 31 coordinates. This gives 43,167 trainable parameters. It is not a Gaussian-residual score parameterization.

Each training minibatch draws $t$ log-uniformly on $[0.001,3]$ and $\epsilon\sim N(0,I)$. With $Y_t=m_tY_0+\sigma_t\epsilon$, the loss is the batch average of $\|\sigma_t s_\theta(Y_t,t)+\epsilon\|^2$.

Both methods use Adam with learning rate $10^{-3}$, weight decay $10^{-5}$, minibatch size 256, and an epoch-wise cosine learning-rate schedule. There is no $\ell_1$ penalty, validation split, or cross-validation in this bounded experiment. The requested budgets are rounded to whole epochs, giving the following update counts.

| Training observations | CINDES updates | DSM updates |
| ---: | ---: | ---: |
| 500 | 19,998 | 60,000 |
| 1,000 | 20,004 | 60,000 |
| 1,500 | 19,998 | 60,000 |
| 2,000 | 19,992 | 60,000 |

## Shared reverse sampler

Both methods start with a standard normal sample and use the same 200-step reverse grid from forward time $T=3$ to $\delta=0.001$. The first half of the reverse-time grid is uniform and the second half is exponential. If consecutive forward times are $\tau>\tau'$, let $\alpha=\exp\{-2(\tau-\tau')\}$ and $\bar\alpha_u=\exp(-2u)$. The mean update is

$$\mu=\frac{y+(1-\alpha)\widehat s(y,\tau)}{\sqrt\alpha},\qquad v=\frac{(1-\alpha)(1-\bar\alpha_{\tau'})}{1-\bar\alpha_\tau}.$$

The next observation is $\mu+\sqrt v\,Z$ except at the ordinary last reverse step, where the source sets it to $\mu$ without adding noise. There is no additional CINDES posterior-resampling or terminal-denoising operation. After the reverse loop, both methods clip every coordinate to $[0,1]$. Thus “no terminal step” refers to omission of the additional correction, not omission of final support clipping. The clipping creates boundary atoms and is part of the protocol being reproduced. Generated observations are processed in batches of 64; this batch size also affects the order in which random numbers are consumed.

## Recorded metrics and time

Each fitted method generates 10,000 observations. TV1 averages the histogram total variation across 31 coordinates, using 20 equal-width bins and the corresponding grid marginal masses. TV2 averages the same quantity across the 30 parent-child pairs, using a $20\times20$ histogram. TV2 is not the TV distance of the full 31-dimensional joint distribution, and it does not average all coordinate pairs.

The recorded likelihood gap is the reference mean of $V(Y)$ minus the generated mean of $V(Y)$. The same grid log normalizer appears in both terms and cancels. This is a signed diagnostic, not a KL divergence or a nonnegative statistical distance. Its sign and magnitude should be interpreted with the marginal diagnostics rather than as a universal quality ordering.

The raw records also retain energy distance, sliced Wasserstein distance, the boundary-near fraction, parameter counts, and update counts. Energy and sliced Wasserstein calculations subsample at most 2,000 observations; the latter uses 500 random projections. The boundary-near fraction is the fraction of generated vectors with at least one coordinate within $10^{-6}$ of zero or one.

The original `cindes_time` and `dsm_time` measure training plus sample generation. They exclude DGP/reference generation and metric evaluation. This code cannot recover separate historical fitting and generation times from these fields. The new task wall-clock metadata covers additional setup and evaluation work and should not replace these fields in a like-for-like runtime table.

## Execution and reproducibility

The preserved BU SCC helper already reads `NSLOTS` or `OMP_NUM_THREADS` when selecting its CPU thread count. The execution bridge also enforces the launcher's requested cap before invoking the driver, including any requested reduction below the allocated slots. The normal SCC allocation is four slots, as in the original script. Actual thread counts and device/software information are recorded. Runtime comparisons to earlier runs depend on those execution conditions.

Fixed seeds identify the random inputs for the retained protocol. They do not promise bitwise equality across different PyTorch releases, hardware, or CPU/GPU kernels. Fresh runs record the software and device environment. The included numerical records contain 80 paired replications at n=500,1000,1500,2000, and their scientific configurations agree exactly. Execution-time scientific-source hashes are available for the 60 small-sample pairs. The earlier n=2000 archive lacks those hash attestations and per-run environment metadata.

The helper in the older bounded reproducibility archive differs from the helper used here only in its module-level CPU-thread initialization: the successful SCC version respects the allocated thread count. All class and function definitions agree. The scientific code used in the successful small-sample study is retained unchanged in this release.

Verification covers the source/configuration identity, aggregation of all 160 archived method records, 80-task submission construction with simulated scheduler commands, and a real reduced CPU run of both methods. These checks do not constitute a rerun of the complete GPU experiment. Their reports are in `verification/`.
