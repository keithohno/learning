import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split


BOUNDARY = 2
CLUSTER_SPACING = 2.5
CLUSTER_RADIUS = 1.0
CLUSTER_POINTS = 400
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
    cluster_radius=CLUSTER_RADIUS,
    boundary=BOUNDARY,
    seed=23,
):
    torch.manual_seed(seed)
    cluster_centers = gen_cluster_centers(cluster_attempts, cluster_spacing, boundary)
    clusters = []
    labels = []
    for i, center in enumerate(cluster_centers):
        clusters.append(
            gen_cluster_around_center(center, cluster_radius, cluster_points)
        )
        labels.append(torch.ones(cluster_points, dtype=torch.long) * i)

    return torch.cat(clusters), torch.cat(labels)


class PointInClusterDataset(Dataset):
    def __init__(self, train=True, **kwargs):
        xs, ys = gen_clusters(**kwargs)
        self.num_clusters = ys[-1] + 1
        xs_train, xs_test, ys_train, ys_test = train_test_split(
            xs, ys, test_size=0.2, random_state=23
        )
        if train:
            self.xs, self.ys = xs_train, ys_train
        else:
            self.xs, self.ys = xs_test, ys_test

    def __len__(self):
        return len(self.xs)

    def __getitem__(self, idx):
        return self.xs[idx], self.ys[idx]
