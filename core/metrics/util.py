import torch


# from https://github.com/facebookresearch/silk
def warp_points(points: torch.Tensor, homography: torch.Tensor):
    """
    Warp the points with the given homography matrix.

    Args:
        points (tensor): the predicted points for an image in the format
            3 x num_pred_points, with a row of x coords, row of y coords, row of probs
        homography (tensor): the 3 x 3 homography matrix connecting two images

    Returns:
        cartesian_points (tensor): the points warped by the homography in the shape
            3 x num_pred_points, with a row of x coords, row of y coords, row of probs
    """
    num_points = points.shape[1]

    # convert to 2 x num_pred_points array with x coords row, y coords row
    points1 = points[:2]

    # add row of 1's for multiplication with the homography
    points1 = torch.vstack((points1, torch.ones(1, num_points, device=points1.device)))

    # calculate homogeneous coordinates by multiplying by the homography
    homogeneous_points = torch.mm(homography, points1)

    # get back to cartesian coordinates by dividing, (optional : KEEPING PROBS AS THIRD ROW)
    cartesian_points = torch.vstack(
        (
            homogeneous_points[0] / homogeneous_points[2],
            homogeneous_points[1] / homogeneous_points[2],
        )
    )
    if points.shape[0] > 2:
        cartesian_points = torch.vstack((cartesian_points, points[2]))

    return cartesian_points


# from https://github.com/facebookresearch/silk
def filter_points(points, img_shape):
    """
    Keep only the points whose coordinates are still inside the
    dimensions of img_shape.

    Args:
        points (tensor): the predicted points for an image
        img_shape (tensor): the image size

    Returns:
        points_to_keep (tensor): the points that are still inside
            the boundaries of the img_shape
    """
    # we want to get rid of any points that are not in the bounds of the second image
    # the mask will be a tensor of shape [num_points_to_keep]
    mask = (
        # ensure x coordinates are greater than 0 and less than image width
        (points[0] >= 0)
        & (points[0] < img_shape[1])
        # ensure y coordinates are greater than 0 and less than image height
        & (points[1] >= 0)
        & (points[1] < img_shape[0])
    )

    # apply the mask
    points_to_keep = points[:, mask]

    return points_to_keep, mask


# from https://github.com/facebookresearch/silk
def keep_true_points(
    points: torch.Tensor,
    homography: torch.Tensor,
    img_shape: torch.Tensor,
):
    """
    Keep only the points whose coordinates when warped by the
    homography are still inside the img_shape dimensions.

    Args:
        points (tensor): the predicted points for an image
        homography (tensor): the 3 x 3 homography matrix connecting
            two images
        img_shape (tensor): the image size (img_height, img_width)

    Returns:
        points_to_keep (tensor): the points that are still inside
            the boundaries of the img_shape after the homography is applied
    """

    # first warp the points by the homography
    warped_points = warp_points(points, homography)

    # we want to get rid of any points that are not in the bounds of the second image
    # the mask will be a tensor of shape [num_points_to_keep]
    points_to_keep, mask = filter_points(warped_points, img_shape)

    # need to warp by the inverse homography to get the original coordinates back
    points_to_keep = points[:, mask]

    return points_to_keep, mask


def select_k_best_points(points, k):
    """
    Select the k most probable points.

    Args:
        points (tensor): a 3 x num_pred_points tensor where the third row is the
            probabilities for each point
        k (int): the number of points to keep

    Returns:
        points (tensor): a 3 x k tensor with only the k best points selected in
            sorted order of the probabilities
    """
    points = points.T

    sorted_indices = torch.argsort(points[:, 2], descending=True)
    sorted_prob = points[sorted_indices]
    start = min(k, points.shape[0])

    sorted_points = sorted_prob[:start]
    sorted_indices = sorted_indices[:start]

    return sorted_points.T, sorted_indices
