from utils.metrics import keypoint_pck_accuracy, keypoint_epe, keypoint_auc
import numpy as np


def get_bb_w_and_h(gt_keypoints: np.array, bb_factor: int = 1) -> np.array:
    """
    Returns width and height of bounding box

    Args:
        gt_keypoints (np.array): GT keypoints
        bb_factor (int, optional): Bounding box margin factor. Defaults to 1.

    Returns:
        np.array: (batch_size, (bb_width, bb_height))
    """

    normalize = np.zeros((gt_keypoints.shape[0], 2))
    for idx, img in enumerate(gt_keypoints):

        xmax, ymax = img.max(axis=0)
        xmin, ymin = img.min(axis=0)

        width = xmax - xmin
        height = ymax - ymin
        normalize[idx][0] = width * bb_factor
        normalize[idx][1] = height * bb_factor

    return normalize


def batch_epe_calculation(pred_keypoints: np.array, true_keypoints: np.array, batch_mask: np.array = None, mask: np.array = None, input_img_size=128) -> float:
    """
    Calculates End Point Error (EPE) for a batch in pixels.

    Args:
        pred_keypoints (np.array): Predicted keypoints.
        true_keypoints (np.array): GT keypoints
        mask (np.array, optional): Mask with information which points to hide from calculation (0 skipped; 1 used). Defaults to None.

    Returns:
        float: EPE
    """

    if mask == None:
        mask = np.ones((true_keypoints.shape[0], 21), dtype=int)

    epe = keypoint_epe(pred_keypoints,
                       true_keypoints, mask, batch_mask)

    return epe


def batch_auc_calculation(pred_keypoints: np.array, true_keypoints: np.array, num_step: int = 20, mask: np.array = None, normalize: np.array = None):
    """
    Calculates Area Under the Curve for a batch.

    Args:
        pred_keypoints (np.array): Predicted keypoints.
        true_keypoints (np.array): GT keypoints
        num_step (int, optional): How dense is treshold. Defaults to 20.
        mask (np.array, optional): Mask with information which points to hide from calculation (0 skipped; 1 used). Defaults to None.
        normalize (np.array, optional): Width and height to normalise. Defaults to None.

    Returns:
        float: AUC
    """

    if mask == None:
        mask = np.ones((true_keypoints.shape[0], 21), dtype=int)

    if normalize == None:
        normalize = get_bb_w_and_h(true_keypoints)

    auc = keypoint_auc(pred=pred_keypoints, gt=true_keypoints,
                       mask=mask, normalize=normalize, num_step=num_step)

    return auc


def batch_pck_calculation(pred_keypoints: np.array, true_keypoints: np.array, treshold: float = 0.2, mask: np.array = None, normalize: np.array = None) -> float:
    """
    Calculates PCK for a batch.

    Args:
        pred_keypoints (np.array): Predicted keypoints.
        true_keypoints (np.array): GT keypoints
        treshold (float, optional): PCK treshold. Defaults to 0.2.
        mask (np.array, optional): Mask with information which points to hide from calculation (0 skipped; 1 used). Defaults to None.
        normalize (np.array, optional): _description_. Defaults to None.

    Returns:
        float: PCK
    """

    if mask == None:
        mask = np.ones((true_keypoints.shape[0], 21), dtype=int)

    if normalize == None:
        normalize = get_bb_w_and_h(true_keypoints)

    _, avg_acc, _ = keypoint_pck_accuracy(
        pred=pred_keypoints, gt=true_keypoints, mask=mask, thr=treshold, normalize=normalize)

    return avg_acc
