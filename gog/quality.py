from sklearn.neighbors import KDTree
import numpy as np


class QualityMetric:
    def __init__(self):
        pass

    def __call__(self, *args, **kwargs):
        pass


class ShapeComplementarityMetric(QualityMetric):
    def __init__(self):
        self.w = 0.5
        self.a = 0.5

    def compute_shape_complementarity(self, hand, target, radius=0.03):

        # Find nearest neighbors for each hand contact
        kdtree = KDTree(target['points'])
        dists, nn_ids = kdtree.query(hand['points'])

        shape_metric = 0.0
        for i in range(hand['points'].shape[0]):

            # Remove duplicates or contacts with nn distance greater than a threshold
            duplicates = np.argwhere(nn_ids == nn_ids[i])[:, 0]
            if np.min(dists[duplicates]) != dists[i] or dists[i] > radius:
                continue

            # Distance error
            e_p = np.linalg.norm(hand['points'][i] - target['points'][nn_ids[i][0]])

            # Alignment error
            c_n = hand['normals'][i] # for TRO is -hand['normals']
            e_n = c_n.dot(target['normals'][nn_ids[i][0]])

            shape_metric += e_p + self.w * e_n

        return - shape_metric / len(hand['points'])

    @staticmethod
    def compute_collision_penalty(hand, collisions):
        if collisions.shape[0] == 0:
            return 0.0

        # Find nearest neighbors for each collision
        kdtree = KDTree(hand['points'])
        dists, nn_ids = kdtree.query(collisions)

        e_col = 0.0
        for i in range(collisions.shape[0]):
            e_col += np.linalg.norm(collisions[i] - hand['points'][nn_ids[i][0]])
        return e_col

    def __call__(self, hand, target, collisions):
        return self.compute_shape_complementarity(hand, target) + \
               self.a * self.compute_collision_penalty(hand, collisions)
