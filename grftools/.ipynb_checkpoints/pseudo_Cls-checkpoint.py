import jax.numpy as jnp

def create_Gaussian_field(PRNGKey1, PRNGKey2, Cl, shape, box_size, mean=0.0):
    ell_x_min_box = 2.0 * jnp.pi / box_size[0]
    ell_y_min_box = 2.0 * jnp.pi / box_size[1]
    ell_x = jnp.fft.fftfreq(shape[0], d=1.0 / shape[0]) * ell_x_min_box
    ell_y = jnp.fft.rfftfreq(shape[1], d=1.0 / shape[1]) * ell_y_min_box
    x_idx, y_idx = jnp.meshgrid(ell_x, ell_y, indexing="ij")
    ell_grid = jnp.sqrt((x_idx) ** 2 + (y_idx) ** 2)

    Cl_grid = jnp.piecewise(
        ell_grid,
        [
            ell_grid != 0,
        ],
        [Cl, 1],
    )
    if jnp.any(Cl_grid <= 0):
        m_ft = jnp.zeros(ell_grid.shape, dtype=jnp.complex64)
        m_ft = m_ft.at[Cl_grid > 0].set(
            random.rayleigh(
                PRNGKey1,
                scale=jnp.sqrt(
                    (shape[0] / box_size[0])
                    * (shape[1] / box_size[1])
                    * shape[0]
                    * shape[1]
                    * Cl_grid[Cl_grid > 0]
                    / 2
                ),
            )
            * jnp.exp(
                2j * jnp.pi * random.uniform(PRNGKey2, ell_grid.shape)[Cl_grid > 0]
            )
        )
    else:
        m_ft = random.rayleigh(
            PRNGKey1,
            scale=jnp.sqrt(
                (shape[0] / box_size[0])
                * (shape[1] / box_size[1])
                * shape[0]
                * shape[1]
                * Cl_grid
                / 2
            ),
        ) * jnp.exp(2j * jnp.pi * random.uniform(PRNGKey2, ell_grid.shape))
    if mean == 0.0:
        m_ft = m_ft.at[ell_grid == 0].set(0.0)
    else:
        m_ft = m_ft.at[ell_grid == 0].set(random.normal(PRNGKey1, scale=mean))

    m = jnp.fft.irfft2(m_ft)
    return m


# attempt at JAX implementation
###############################
def calculate_pseudo_Cl(
    map1, map2, box_size, n_bin=None, ell_min=None, ell_max=None, logspaced=False
):
    """Estimates the cross-power spectrum of two maps.

    Required arguments:
    map1            Array of size (N, M).
    map2            Array of same shape as map1.
    box_size        Physical size (L1, L2) of the maps.

    Optional arguments:
    n_bin           Number of ell bins. If None, no binning is performed.
    ell_min         Minimum ell.
    ell_max,        Maximum ell.
    logspaced       Log-spaced bins. Default is False.

    Returns:
    Tuple (pCl_real, pCl_real_err, ell_mean, bin_edges, n_mode) with
        pCl_real        Estimated cross-power spectrum,
        pCl_real_err    Error on the mean, estimated from the scatter of the
                        individual modes,
        ell_mean        Mean ell per bin,
        bin_edges       Edges of the ell bins,
        n_mode          Number of modes per bin.
    """

    if map1.shape != map2.shape:
        raise ValueError(
            "Map dimensions don't match: {}x{} vs {}x{}".format(
                *(map1.shape + map2.shape)
            )
        )

    # This can be streamlined alot
    map1_ft = (
        jnp.fft.fft2(map1)
        * (box_size[0] / map1.shape[0])
        * (box_size[1] / map1.shape[1])
    )
    map1_ft = jnp.fft.fftshift(map1_ft, axes=0)
    map2_ft = (
        jnp.fft.fft2(map2)
        * (box_size[0] / map1.shape[0])
        * (box_size[1] / map1.shape[1])
    )
    map2_ft = jnp.fft.fftshift(map2_ft, axes=0)
    map_1x2_ft = map1_ft.conj() * map2_ft

    ell_x_min_box = 2.0 * jnp.pi / box_size[0]
    ell_y_min_box = 2.0 * jnp.pi / box_size[1]
    ell_x = (
        jnp.fft.fftshift(jnp.fft.fftfreq(map1.shape[0], d=1.0 / map1.shape[0]))
        * ell_x_min_box
    )
    ell_y = jnp.fft.fftfreq(map1.shape[1], d=1.0 / map1.shape[1]) * ell_y_min_box
    x_idx, y_idx = jnp.meshgrid(ell_x, ell_y, indexing="ij")
    ell_grid = jnp.sqrt((x_idx) ** 2 + (y_idx) ** 2)

    if n_bin == None:
        bin_edges = jnp.arange(
            start=jnp.array([ell_x_min_box, ell_y_min_box]).min() / 1.00001,
            stop=jnp.max(ell_grid),
            step=jnp.array([ell_x_min_box, ell_y_min_box]).min(),
        )
        n_bin = len(bin_edges) - 1
    else:
        if ell_max > jnp.max(ell_grid):
            raise RuntimeWarning(
                "Maximum ell is {}, where as ell_max was set as {}.".format(
                    jnp.max(ell_grid), ell_max
                )
            )
        if ell_min < jnp.min([ell_x_min_box, ell_y_min_box]):
            raise RuntimeWarning(
                "Minimum ell is {}, where as ell_min was set as {}.".format(
                    jnp.array([ell_x_min_box, ell_y_min_box]).min(), ell_min
                )
            )
        if logspaced:
            bin_edges = jnp.logspace(
                jnp.log10(ell_min), jnp.log10(ell_max), n_bin + 1, endpoint=True
            )
        else:
            bin_edges = jnp.linspace(ell_min, ell_max, n_bin + 1, endpoint=True)

    pCl_real = jnp.zeros(n_bin)
    pCl_imag = jnp.zeros(n_bin)
    pCl_real_err = jnp.zeros(n_bin)
    pCl_imag_err = jnp.zeros(n_bin)
    ell_mean = jnp.zeros(n_bin)
    n_mode = jnp.zeros(n_bin)
    bin_centers = (bin_edges[1:] + bin_edges[:-1]) / 2

    ell_sort_idx = jnp.argsort(ell_grid.flatten())
    map_1x2_ft_sorted = map_1x2_ft.flatten()[ell_sort_idx]
    ell_grid_sorted = ell_grid.flatten()[ell_sort_idx]
    bin_idx = jnp.searchsorted(ell_grid_sorted, bin_edges)

    for i in range(n_bin):
        P = map_1x2_ft_sorted[bin_idx[i] : bin_idx[i + 1]] / (box_size[0] * box_size[1])
        ell = ell_grid_sorted[bin_idx[i] : bin_idx[i + 1]]
        pCl_real = pCl_real.at[i].set(jnp.mean(P.real))
        # pCl_imag = pCl_imag.at(i).set(jnp.mean(P.imag))
        # pCl_real_err = jnp.sqrt(jnp.var(P.real) / len(P))
        # pCl_imag_err[i] = jnp.sqrt(jnp.var(P.imag) / len(P))
        ell_mean = ell_mean.at[i].set(jnp.mean(ell))
        # n_mode = len(P)

    return pCl_real, ell_mean