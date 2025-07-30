from __future__ import annotations

import colour
import numpy as np
import numpy.typing as npt


class Lab:
    @staticmethod
    def from_sRGB(image: npt.NDArray) -> npt.NDArray:
        xyz_image = colour.sRGB_to_XYZ(image)
        return colour.XYZ_to_Lab(xyz_image)

    @staticmethod
    def to_sRGB(image: npt.NDArray) -> npt.NDArray:
        xyz_image = colour.Lab_to_XYZ(image)
        return colour.XYZ_to_sRGB(xyz_image)


class Prolab:
    illuminant_XYZ = np.array((0.95047, 1.0, 1.08883))
    Q = np.array(
        (
            (75.5644333, 486.62630402, 167.39926268, 0.0),
            (617.72787198, -595.4607401, -22.26712291, 0.0),
            (48.3448951, 194.93477285, -243.27966363, 0.0),
            (0.7554, 3.8666, 1.6739, 1.0),
        )
    )

    inv_Q = np.linalg.inv(Q)

    @staticmethod
    def projective_transformation(
        points: npt.NDArray, pr_matrix: npt.NDArray
    ) -> npt.NDArray:
        homog_index = points.shape[1]
        cartesian_index = pr_matrix.shape[0] - 1
        points_homog = np.insert(points, homog_index, 1, axis=1)
        proj_points_homog = points_homog @ pr_matrix.T
        projection = proj_points_homog / proj_points_homog[:, cartesian_index:]
        return projection[:, :cartesian_index]

    @classmethod
    def from_sRGB(
        cls, image: npt.NDArray, illuminant_xyz: npt.NDArray = illuminant_XYZ
    ) -> npt.NDArray:
        h, w, c = image.shape
        xyz_image = colour.sRGB_to_XYZ(image)
        return cls.projective_transformation(
            (xyz_image / illuminant_xyz).reshape(-1, c), cls.Q
        ).reshape(h, w, c)

    @classmethod
    def to_sRGB(
        cls, image: npt.NDArray, illuminant_xyz: npt.NDArray = illuminant_XYZ
    ) -> npt.NDArray:
        h, w, c = image.shape
        xyz = (
            cls.projective_transformation(image.reshape(-1, c), cls.inv_Q)
            * illuminant_xyz
        ).reshape(h, w, c)
        return colour.XYZ_to_sRGB(xyz)
