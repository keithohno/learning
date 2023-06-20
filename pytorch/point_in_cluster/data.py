import torch


BOUNDARY = 2.5
CLUSTER_SPACING = 3.0
CLUSTER_POINTS = 10
ATTEMPTS = 10000


def random_point(boundary):
    return torch.rand(3) * 2 * boundary - boundary


def distance(p1, p2):
    return (p1 - p2).norm()


def gen_cluster_centers(attempts, spacing, boundary):
    cluster_centers = []
    for _ in range(attempts):
        new_center = random_point(boundary)
        if all([distance(new_center, c) > spacing for c in cluster_centers]):
            cluster_centers.append(new_center)

    return torch.cat(cluster_centers).view(-1, 3)


def gen_cluster_around_center(center, radius, num_points):
    return center + torch.rand(num_points, 3) * 2 * radius - radius


def gen_clusters(
    cluster_attempts=ATTEMPTS,
    cluster_spacing=CLUSTER_SPACING,
    cluster_points=CLUSTER_POINTS,
    boundary=BOUNDARY,
):
    cluster_centers = gen_cluster_centers(cluster_attempts, cluster_spacing, boundary)
    clusters = []
    labels = []
    for i, center in enumerate(cluster_centers):
        clusters.append(
            gen_cluster_around_center(center, cluster_spacing / 2, cluster_points)
        )
        labels.append(torch.ones(cluster_points) * i)

    return torch.cat(clusters), torch.cat(labels)
