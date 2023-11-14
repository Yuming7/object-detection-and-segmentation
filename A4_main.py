import os.path
import timeit
import numpy as np
from skimage.draw import polygon

from A4_submission import detect_and_segment


def compute_classification_acc(pred, gt):
    assert pred.shape == gt.shape
    return (pred == gt).astype(int).sum() / gt.size


def compute_segmentation_acc(pred, gt):
    # pred value should be from 0 to 10, where 10 is the background.
    assert pred.shape == gt.shape

    return (pred == gt).astype(int).sum() / gt.size


def get_iou(bbox_pred, bbox_gt, L_pred, L_gt):
    """all pixel coordinates within the prediction bounding box"""
    rr, cc = polygon([bbox_pred[0], bbox_pred[0], bbox_pred[2], bbox_pred[2]],
                     [bbox_pred[1], bbox_pred[3], bbox_pred[3], bbox_pred[1]], [64, 64])
    L_pred[rr, cc] = 1

    """all pixel coordinates within the GT bounding box"""
    rr, cc = polygon([bbox_gt[0], bbox_gt[0], bbox_gt[2], bbox_gt[2]],
                     [bbox_gt[1], bbox_gt[3], bbox_gt[3], bbox_gt[1]], [64, 64])
    L_gt[rr, cc] = 1

    L_sum = L_pred + L_gt
    intersection = np.sum(L_sum == 2)
    union = np.sum(L_sum >= 1)

    iou = intersection / union

    L_pred[:, :] = 0
    L_gt[:, :] = 0

    return iou


def compute_mean_iou(bboxes_pred, bboxes_gt, classes_pred, classes_gt):
    """

    :param bboxes_pred: predicted bounding boxes, shape=(n_images,2,4)
    :param bboxes_gt: ground truth bounding boxes, shape=(n_images,2,4)
    :param classes_pred: predicted classes, shape=(n_images,2)
    :param classes_gt: ground truth classes, shape=(n_images,2)
    :return:
    """

    n_images = np.shape(bboxes_gt)[0]
    L_pred = np.zeros((64, 64))
    L_gt = np.zeros((64, 64))
    iou_sum = 0.0
    for i in range(n_images):
        iou1 = get_iou(bboxes_pred[i, 0, :], bboxes_gt[i, 0, :], L_pred, L_gt)
        iou2 = get_iou(bboxes_pred[i, 1, :], bboxes_gt[i, 1, :], L_pred, L_gt)

        iou_sum1 = iou1 + iou2

        if classes_pred[i, 0] == classes_pred[i, 1] and classes_gt[i, 0] == classes_gt[i, 1]:
            iou1 = get_iou(bboxes_pred[i, 0, :], bboxes_gt[i, 1, :], L_pred, L_gt)
            iou2 = get_iou(bboxes_pred[i, 1, :], bboxes_gt[i, 0, :], L_pred, L_gt)

            iou_sum2 = iou1 + iou2

            if iou_sum2 > iou_sum1:
                iou_sum1 = iou_sum2

        iou_sum += iou_sum1

    mean_iou = iou_sum / (2. * n_images)

    return mean_iou


class Params:
    def __init__(self):
        # self.prefix = "test"
        self.prefix = "valid"
        # self.prefix = "train"
        self.load = 1
        self.save = 1
        self.load_path = 'saved_preds.npz'
        self.vis = 0
        self.vis_size = (300, 300)
        self.show_det = 0
        self.show_seg = 1

        self.speed_thresh = 10
        self.acc_thresh = (0.7, 0.98)
        self.iou_thresh = (0.7, 0.98)
        self.seg_thresh = (0.7, 0.98)

        self.class_cols = {
            0: 'red',
            1: 'green',
            2: 'blue',
            3: 'magenta',
            4: 'cyan',
            5: 'yellow',
            6: 'purple',
            7: 'forest_green',
            8: 'orange',
            9: 'white',
            10: 'black',
        }


def compute_score(res, thresh):
    min_thres, max_thres = thresh

    if res < min_thres:
        score = 0.0
    elif res > max_thres:
        score = 100.0
    else:
        score = float(res - min_thres) / (max_thres - min_thres) * 100
    return score


def main():
    params = Params()

    try:
        import paramparse
    except ImportError:
        pass
    else:
        paramparse.process(params)

    prefix = params.prefix

    images = np.load(prefix + "_X.npy")
    gt_classes = np.load(prefix + "_Y.npy")
    gt_bboxes = np.load(prefix + "_bboxes.npy")
    gt_seg = np.load(prefix + "_seg.npy")

    n_images = images.shape[0]

    if params.load and os.path.exists(params.load_path):
        print(f'loading predictions from {params.load_path}')
        saved_preds = np.load(params.load_path)
        pred_classes = saved_preds['pred_classes']
        pred_bboxes = saved_preds['pred_bboxes']
        pred_seg = saved_preds['pred_seg']

        test_time = test_speed = 0
    else:
        print(f'running prediction on {n_images} {prefix} images')
        start_t = timeit.default_timer()
        pred_classes, pred_bboxes, pred_seg = detect_and_segment(images)
        end_t = timeit.default_timer()
        test_time = end_t - start_t
        assert test_time > 0, "test_time cannot be 0"
        test_speed = float(n_images) / test_time

        if params.save:
            np.savez_compressed(params.load_path, pred_classes=pred_classes, pred_bboxes=pred_bboxes, pred_seg=pred_seg)

    cls_acc = compute_classification_acc(pred_classes, gt_classes)
    iou = compute_mean_iou(pred_bboxes, gt_bboxes, pred_classes, gt_classes)
    seg_acc = compute_segmentation_acc(pred_seg, gt_seg)

    acc_score = compute_score(cls_acc, params.acc_thresh)
    iou_score = compute_score(iou, params.iou_thresh)
    seg_score = compute_score(seg_acc, params.seg_thresh)

    if test_speed < params.speed_thresh:
        overall_score = 0
    else:
        overall_score = ((iou_score + acc_score) / 2. + seg_score) / 2.

    print(f"Classification Accuracy: {cls_acc*100:.3f} %")
    print(f"Detection IOU: {iou*100:.3f} %")
    print(f"Segmentation Accuracy: {seg_acc*100:.3f} %")

    print(f"Test time: {test_time:.3f} seconds")
    print(f"Test speed: {test_speed:.3f} images / second")

    print(f"Classification Score: {acc_score:.3f}")
    print(f"IOU Score: {iou_score:.3f}")
    print(f"Segmentation Score: {seg_score:.3f}")
    print(f"Overall Score: {overall_score:.3f}")

    if not params.vis:
        return

    import cv2
    from A4_utils import vis_bboxes, vis_seg, annotate

    print('press spacebar to toggle pause and escape to quit')
    pause_after_frame = 1
    for img_id in range(n_images):
        src_img = images[img_id, ...].squeeze()
        src_img = src_img.reshape((64, 64, 3)).astype(np.uint8)

        vis_img = np.copy(src_img)

        bbox_1 = gt_bboxes[img_id, 0, :].squeeze().astype(np.int32)
        bbox_2 = gt_bboxes[img_id, 1, :].squeeze().astype(np.int32)
        y1, y2 = gt_classes[img_id, ...].squeeze()
        gt_classes[img_id, ...].squeeze()
        vis_img = vis_bboxes(vis_img, bbox_1, bbox_2, y1, y2, params.vis_size)
        vis_img_seg_gt = vis_seg(src_img, gt_seg, img_id, params.class_cols, params.vis_size)

        vis_img_list = [vis_img, vis_img_seg_gt]
        vis_img_labels = ['gt det', 'gt seg']

        if params.show_det:
            vis_img_det = np.copy(src_img)
            bbox_1 = pred_bboxes[img_id, 0, :].squeeze().astype(np.int32)
            bbox_2 = pred_bboxes[img_id, 1, :].squeeze().astype(np.int32)
            y1, y2 = pred_classes[img_id, ...].squeeze()
            gt_classes[img_id, ...].squeeze()
            vis_img_det = vis_bboxes(vis_img_det, bbox_1, bbox_2, y1, y2, params.vis_size)
            vis_img_list.append(vis_img_det)
            vis_img_labels.append('pred det')

        if params.show_seg:
            vis_img_seg = vis_seg(src_img, pred_seg, img_id, params.class_cols, params.vis_size)
            vis_img_list.append(vis_img_seg)
            vis_img_labels.append('pred seg')

        vis_img = annotate(vis_img_list,
                           text=f'image {img_id}',
                           img_labels=vis_img_labels, grid_size=(1, -1))
        cv2.imshow('vis_img', vis_img)

        key = cv2.waitKey(1 - pause_after_frame)
        if key == 27:
            return
        elif key == 32:
            pause_after_frame = 1 - pause_after_frame


if __name__ == '__main__':
    main()
