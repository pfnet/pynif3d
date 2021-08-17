import torch

from pynif3d.common.verification import check_equal, check_pos_int, check_shapes_match


def ray_sphere_intersection(rays_d, spheres_o, radius=1.0):
    """
    Computes the intersection points between a given set of spheres placed at
    origins `spheres_o` and a given set of ray directions `rays_d`.

    Args:
        rays_d (torch.Tensor): Tensor containing the ray directions. Its
            shape is ``(batch_size, n_rays, 3)``.
        spheres_o (torch.Tensor): Tensor containing the sphere origins. Its shape
            is ``(batch_size, n_rays, 3)``.
        radius (float): The radius of the spheres.

    Returns:
        tuple: Tuple containing the two intersection velocities per ray (as a
            ``(batch_size, n_rays, 2)`` tensor) and a mask (as a
            ``(batch_size, n_rays)`` tensor). The rays which intersect the sphere
            are marked as True, while the ones that do not are marked as False.
    """
    check_shapes_match(rays_d, spheres_o, "rays_d", "spheres_o")
    check_equal(spheres_o.shape[-1], 3, "spheres_o.shape[-1]", "3")

    # ---------------------------------------------------------------------------
    # Condition for intersecting a sphere (centered at sphere_o):
    # || ray(t) - sphere_o || ** 2 = r ** 2
    #
    # -> || ray_o + t * ray_d - sphere_o || ** 2 = r ** 2
    # -> < ray_o + t * ray_d - sphere_o, ray_o + t * ray_d - sphere_o > = r ** 2
    # -> < ray_d, ray_d > * t ** 2 + 2 * t * < ray_d, ray_o - sphere_o > +
    #        + || ray_o - sphere_o || ** 2 - r ** 2 = 0
    #
    # ---------------------------------------------------------------------------
    # Solving for t:
    # a = < ray_d, ray_d > = || ray_d || ** 2 = 1
    # b = 2 * < ray_d, ray_o - sphere_o >
    # c = || ray_o - sphere_o || ** 2 - r ** 2
    #
    # delta = b ** 2 - 4 * a * c
    #
    # When delta > 0:
    #       t1 = (-b - sqrt(delta)) / 2
    #       t2 = (-b + sqrt(delta)) / 2
    #
    # Let ray_o_prime = ray_o - sphere_o:
    #       t1 = - < ray_d, ray_o_prime >
    #            - sqrt(< ray_d, ray_o_prime > ** 2 - || ray_o_prime || ** 2 + r ** 2)
    #       t2 = - < ray_d, ray_o_prime >
    #            + sqrt(< ray_d, ray_o_prime > ** 2 - || ray_o_prime || ** 2 + r ** 2)
    # ---------------------------------------------------------------------------

    directions = rays_d.reshape(-1, 3)
    origins = spheres_o.reshape(-1, 3)

    dot_product = torch.bmm(directions[..., None, :], origins[..., None]).squeeze()
    delta = dot_product ** 2 - origins.norm(2, dim=-1) ** 2 + radius ** 2

    mask = delta > 0
    t1 = -dot_product[mask] - torch.sqrt(delta[mask])
    t2 = -dot_product[mask] + torch.sqrt(delta[mask])

    z_vals = torch.zeros((len(directions), 2), device=rays_d.device)
    z_vals[mask] = torch.stack([t1, t2], dim=-1)

    z_vals = z_vals.reshape(*rays_d.shape[:-1], 2).transpose(-1, 0)
    mask = mask.reshape(*rays_d.shape[:-1])

    return z_vals, mask


def secant(secant_fn, xs_min, xs_max, n_iterations=100):
    """
    Determines the approximate roots of a function in a given interval range by running
    the secant method.
    Args:
        secant_fn (instance): The function instance that the secant method is run for.
        xs_min (torch.Tensor): Tensor containing the minimum interval range. Its shape
            is ``(n_samples,)``.
        xs_max (torch.Tensor): Tensor containing the maximum interval range. Its shape
            is ``(n_samples,)``.
        n_iterations (int): The number of iterations to run the secant method for.

    Returns:
        torch.Tensor: Tensor containing the predicted X values. Its shape is
            ``(n_samples,)``.
    """
    check_pos_int(n_iterations, "n_iterations")
    eps = 1e-10

    ys_min, ys_max = secant_fn(torch.stack([xs_min, xs_max]))
    xs_pred = xs_min - ys_min * (xs_max - xs_min) / (ys_max - ys_min + eps)

    for _iteration in range(n_iterations):
        y_pred = secant_fn(xs_pred)

        indices_min = y_pred > 0
        if torch.any(indices_min):
            xs_min[indices_min] = xs_pred[indices_min]
            ys_min[indices_min] = y_pred[indices_min]

        indices_max = y_pred < 0
        if torch.any(indices_max):
            xs_max[indices_max] = xs_pred[indices_max]
            ys_max[indices_max] = y_pred[indices_max]

        xs_pred = xs_min - ys_min * (xs_max - xs_min) / (ys_max - ys_min + eps)

    return xs_pred
