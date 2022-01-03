from __future__ import print_function

import numpy as np
import json
import os
from datetime import datetime

import torch
from tqdm import tqdm


def map_iou(boxes_true, boxes_pred, scores, thresholds = [0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75]):
    """
    Mean average precision at differnet intersection over union (IoU) threshold
    
    input:
        boxes_true: Mx4 numpy array of ground true bounding boxes of one image. 
                    bbox format: (x1, y1, w, h)
        boxes_pred: Nx4 numpy array of predicted bounding boxes of one image. 
                    bbox format: (x1, y1, w, h)
        scores:     length N numpy array of scores associated with predicted bboxes
        thresholds: IoU shresholds to evaluate mean average precision on
    output: 
        map: mean average precision of the image
    """
    
    # According to the introduction, images with no ground truth bboxes will not be 
    # included in the map score unless there is a false positive detection (?)
        
    # return None if both are empty, don't count the image in final evaluation (?)
    if len(boxes_true) == 0 and len(boxes_pred) == 0:
        return None
    
    # assert boxes_true.shape[1] == 4 or boxes_pred.shape[1] == 4, "boxes should be 2D arrays with shape[1]=4"
    if len(boxes_pred):
        assert len(scores) == len(boxes_pred), "boxes_pred and scores should be same length"
        # sort boxes_pred by scores in decreasing order
        boxes_pred = boxes_pred[np.argsort(scores)[::-1], :]
    
    map_total = 0
    
    # loop over thresholds
    for t in thresholds:
        matched_bt = set()
        tp, fn = 0, 0
        for i, bt in enumerate(boxes_true):
            matched = False
            for j, bp in enumerate(boxes_pred):
                miou = iou(bt, bp)
                if miou >= t and not matched and j not in matched_bt:
                    matched = True
                    tp += 1 # bt is matched for the first time, count as TP
                    matched_bt.add(j)
            if not matched:
                fn += 1 # bt has no match, count as FN
                
        fp = len(boxes_pred) - len(matched_bt) # FP is the bp that not matched to any bt
        m = tp / (tp + fn + fp)
        map_total += m
    
    return map_total / len(thresholds)

def compute_overlap(a, b):
    """
    Parameters
    ----------
    a: (N, 4) ndarray of float
    b: (K, 4) ndarray of float
    Returns
    -------
    overlaps: (N, K) ndarray of overlap between boxes and query_boxes
    """
    area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])

    iw = np.minimum(np.expand_dims(a[:, 2], axis=1), b[:, 2]) - np.maximum(np.expand_dims(a[:, 0], 1), b[:, 0])
    ih = np.minimum(np.expand_dims(a[:, 3], axis=1), b[:, 3]) - np.maximum(np.expand_dims(a[:, 1], 1), b[:, 1])

    iw = np.maximum(iw, 0)
    ih = np.maximum(ih, 0)

    ua = np.expand_dims((a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]), axis=1) + area - iw * ih

    ua = np.maximum(ua, np.finfo(float).eps)

    intersection = iw * ih

    return intersection / ua


def _compute_ap(recall, precision):
    """ Compute the average precision, given the recall and precision curves.
    Code originally from https://github.com/rbgirshick/py-faster-rcnn.
    # Arguments
        recall:    The recall curve (list).
        precision: The precision curve (list).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """
    # correct AP calculation
    # first append sentinel values at the end
    mrec = np.concatenate(([0.], recall, [1.]))
    mpre = np.concatenate(([0.], precision, [0.]))

    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def _get_detections(dataset, retinanet, score_threshold=0.05, max_detections=100, save_path=None):
    """ Get the detections from the retinanet using the generator.
    The result is a list of lists such that the size is:
        all_detections[num_images][num_classes] = detections[num_detections, 4 + num_classes]
    # Arguments
        dataset         : The generator used to run images through the retinanet.
        retinanet           : The retinanet to run on the images.
        score_threshold : The score confidence threshold to use.
        max_detections  : The maximum number of detections to use per image.
        save_path       : The path to save the images with visualized detections to.
    # Returns
        A list of lists containing the detections for each image in the generator.
    """
    all_detections = [[None for i in range(dataset.num_classes())] for j in range(len(dataset))]

    retinanet.eval()
    
    with torch.no_grad():

        for index in range(len(dataset)):
            data = dataset[index]
            scale = data['scale']

            # run network
            scores, labels, boxes = retinanet(data['img'].permute(2, 0, 1).cuda().float().unsqueeze(dim=0))
            scores = scores.cpu().numpy()
            labels = labels.cpu().numpy()
            boxes  = boxes.cpu().numpy()

            # correct boxes for image scale
            boxes /= scale

            # select indices which have a score above the threshold
            indices = np.where(scores > score_threshold)[0]
            if indices.shape[0] > 0:
                # select those scores
                scores = scores[indices]

                # find the order with which to sort the scores
                scores_sort = np.argsort(-scores)[:max_detections]

                # select detections
                image_boxes      = boxes[indices[scores_sort], :]
                image_scores     = scores[scores_sort]
                image_labels     = labels[indices[scores_sort]]
                image_detections = np.concatenate([image_boxes, np.expand_dims(image_scores, axis=1), np.expand_dims(image_labels, axis=1)], axis=1)

                # copy detections to all_detections
                for label in range(dataset.num_classes()):
                    all_detections[index][label] = image_detections[image_detections[:, -1] == label, :-1]
            else:
                # copy detections to all_detections
                for label in range(dataset.num_classes()):
                    all_detections[index][label] = np.zeros((0, 5))

            print('{}/{}'.format(index + 1, len(dataset)), end='\r')

    return all_detections


def _get_annotations(generator):
    """ Get the ground truth annotations from the generator.
    The result is a list of lists such that the size is:
        all_detections[num_images][num_classes] = annotations[num_detections, 5]
    # Arguments
        generator : The generator used to retrieve ground truth annotations.
    # Returns
        A list of lists containing the annotations for each image in the generator.
    """
    all_annotations = [[None for i in range(generator.num_classes())] for j in range(len(generator))]

    for i in range(len(generator)):
        # load the annotations
        annotations = generator.load_annotations(i)

        # copy detections to all_annotations
        for label in range(generator.num_classes()):
            all_annotations[i][label] = annotations[annotations[:, 4] == label, :4].copy()

        # print('{}/{}'.format(i + 1, len(generator)), end='\r')

    return all_annotations

def _get_predictions(dataset, retinanet):
    scores_list = []
    labels_list = []
    boxes_list = []

    print('Get network redictions...')

    retinanet.eval()
    
    with torch.no_grad():
        with tqdm(total=len(dataset)) as pbar:
            for index in range(len(dataset)):
                data = dataset[index]
                scale = data['scale']

                # run network
                scores, labels, boxes = retinanet(data['img'].permute(2, 0, 1).cuda().float().unsqueeze(dim=0))
                scores = scores.cpu().numpy()
                labels = labels.cpu().numpy()
                boxes  = boxes.cpu().numpy()

                # correct boxes for image scale
                boxes /= scale

                scores_list.append(scores)
                labels_list.append(labels)
                boxes_list.append(boxes)

                pbar.update(1)

    return scores_list, labels_list, boxes_list

def _get_scan_detections(
    scores_list,
    labels_list,
    boxes_list,
    num_classes=1,
    score_thresholds=[0.05],
    max_detections=100,
    save_path=None
):
    detections_list = []

    for score_threshold in score_thresholds:
        all_detections = [[None for i in range(num_classes)] for j in range(len(scores_list))]

        for index, (scores, labels, boxes) in enumerate(zip(scores_list, labels_list, boxes_list)):
            # select indices which have a score above the threshold
            indices = np.where(scores > score_threshold)[0]
            if indices.shape[0] > 0:
                # select those scores
                scores = scores[indices]

                # find the order with which to sort the scores
                scores_sort = np.argsort(-scores)[:max_detections]

                # select detections
                image_boxes      = boxes[indices[scores_sort], :]
                image_scores     = scores[scores_sort]
                image_labels     = labels[indices[scores_sort]]
                image_detections = np.concatenate([image_boxes, np.expand_dims(image_scores, axis=1), np.expand_dims(image_labels, axis=1)], axis=1)

                # copy detections to all_detections
                for label in range(num_classes):
                    all_detections[index][label] = image_detections[image_detections[:, -1] == label, :-1]
            else:
                # copy detections to all_detections
                for label in range(num_classes):
                    all_detections[index][label] = np.zeros((0, 5))

        detections_list.append(all_detections)
        
    return detections_list

def evaluate(
    generator,
    retinanet,
    iou_threshold=0.5,
    score_threshold=0.05,
    max_detections=100,
    save_path=None
):
    """ Evaluate a given dataset using a given retinanet.
    # Arguments
        generator       : The generator that represents the dataset to evaluate.
        retinanet           : The retinanet to evaluate.
        iou_threshold   : The threshold used to consider when a detection is positive or negative.
        score_threshold : The score confidence threshold to use for detections.
        max_detections  : The maximum number of detections to use per image.
        save_path       : The path to save images with visualized detections to.
    # Returns
        A dict mapping class names to mAP scores.
    """



    # gather all detections and annotations

    all_detections     = _get_detections(generator, retinanet, score_threshold=score_threshold, max_detections=max_detections, save_path=save_path)
    all_annotations    = _get_annotations(generator)

    average_precisions = {}

    for label in range(generator.num_classes()):
        false_positives = np.zeros((0,))
        true_positives  = np.zeros((0,))
        scores          = np.zeros((0,))
        num_annotations = 0.0

        for i in range(len(generator)):
            detections           = all_detections[i][label]
            annotations          = all_annotations[i][label]
            num_annotations     += annotations.shape[0]
            detected_annotations = []

            for d in detections:
                scores = np.append(scores, d[4])

                if annotations.shape[0] == 0:
                    false_positives = np.append(false_positives, 1)
                    true_positives  = np.append(true_positives, 0)
                    continue

                overlaps            = compute_overlap(np.expand_dims(d, axis=0), annotations)
                assigned_annotation = np.argmax(overlaps, axis=1)
                max_overlap         = overlaps[0, assigned_annotation]

                if max_overlap >= iou_threshold and assigned_annotation not in detected_annotations:
                    false_positives = np.append(false_positives, 0)
                    true_positives  = np.append(true_positives, 1)
                    detected_annotations.append(assigned_annotation)
                else:
                    false_positives = np.append(false_positives, 1)
                    true_positives  = np.append(true_positives, 0)

        # no annotations -> AP for this class is 0 (is this correct?)
        if num_annotations == 0:
            average_precisions[label] = 0, 0
            continue

        # sort by score
        indices         = np.argsort(-scores)
        false_positives = false_positives[indices]
        true_positives  = true_positives[indices]

        # compute false positives and true positives
        false_positives = np.cumsum(false_positives)
        true_positives  = np.cumsum(true_positives)

        # compute recall and precision
        recall    = true_positives / num_annotations
        precision = true_positives / np.maximum(true_positives + false_positives, np.finfo(np.float64).eps)

        # compute average precision
        average_precision  = _compute_ap(recall, precision)
        average_precisions[label] = average_precision, num_annotations
    
    print('\nmAP:')
    for label in range(generator.num_classes()):
        label_name = generator.label_to_name(label)
        print('{}: {}'.format(label_name, average_precisions[label][0]))
    
    return average_precisions

def to_pt(bbox):
    pt = [
        bbox[0],
        bbox[1],
        bbox[2] - bbox[0],
        bbox[3] - bbox[1]
    ]

    return pt

def evaluate_rsna(
    generator,
    retinanet,
    score_thresholds=[0.05],
    max_detections=100,
    save_path=None
):
    """ Evaluate a given dataset using a given retinanet.
    # Arguments
        generator       : The generator that represents the dataset to evaluate.
        retinanet           : The retinanet to evaluate.
        iou_threshold   : The threshold used to consider when a detection is positive or negative.
        score_threshold : The score confidence threshold to use for detections.
        max_detections  : The maximum number of detections to use per image.
        save_path       : The path to save images with visualized detections to.
    # Returns
        A dict mapping class names to mAP scores.
    """

    # gather all detections and annotations
    scores_list, labels_list, boxes_list = _get_predictions(generator, retinanet)
    detections_list = _get_scan_detections(
        scores_list,
        labels_list,
        boxes_list,
        score_thresholds=score_thresholds,
        max_detections=max_detections,
        save_path=save_path
    )
    all_annotations    = _get_annotations(generator)

    ap_list = []
    youden_list = []
    sensitivity_list = []
    specificity_list = []

    for all_detections in detections_list:
        average_precisions = []

        true_positive = 0
        positive = 0
        true_negative = 0
        negative = 0

        for label in range(generator.num_classes()):
            for i in range(len(generator)):
                detections           = all_detections[i][label]
                annotations          = all_annotations[i][label]

                boxes_true = [annot[:4] for annot in annotations]
                boxes_true = [to_pt(box) for box in boxes_true]
                boxes_true = np.array(boxes_true)

                boxes_pred = [det[:4] for det in detections]
                boxes_pred = [to_pt(box) for box in boxes_pred]
                boxes_pred = np.array(boxes_pred)

                scores = [det[4] for det in detections]
                scores = np.array(scores)

                mAP = map_iou(boxes_true, boxes_pred, scores)

                if mAP is not None:
                    average_precisions.append(mAP)

                    if mAP > 0.:
                        # hit
                        true_positive += 1
                        positive += 1
                    else:
                        if len(annotations) > 0:
                            # miss
                            positive += 1
                        else:
                            # false negative
                            negative += 1
                else: # mAP = None means ture negative
                    true_negative += 1
                    negative += 1
        
        ap = np.array(average_precisions).mean()
        ap_list.append(ap)

        sensitivity = true_positive / positive
        specificity = true_negative / negative
        youden_index = (sensitivity + specificity) - 1.
        youden_list.append(youden_index)
        sensitivity_list.append(sensitivity)
        specificity_list.append(specificity)
    
    return ap_list, youden_list, sensitivity_list, specificity_list

def export(
    generator,
    retinanet,
    score_thresholds=[0.05],
    max_detections=100,
    image_path=None,
    csv_path=None,
    scale=0.9,
    tag='resnet'
):
    """ Export a given dataset using a given retinanet.
    # Arguments
        generator       : The generator that represents the dataset to evaluate.
        retinanet           : The retinanet to evaluate.
        iou_threshold   : The threshold used to consider when a detection is positive or negative.
        score_threshold : The score confidence threshold to use for detections.
        max_detections  : The maximum number of detections to use per image.
        save_path       : The path to save detections.
    # Returns
        A dict mapping class names to mAP scores.
    """

    csv_dir = os.path.dirname(csv_path)
    if not os.path.exists(csv_dir):
        os.mkdir(csv_dir)

    now = datetime.now()

    # gather all detections and annotations
    scores_list, labels_list, boxes_list = _get_predictions(generator, retinanet)
    detections_list = _get_scan_detections(
        scores_list,
        labels_list,
        boxes_list,
        score_thresholds=score_thresholds,
        max_detections=max_detections
    )

    for index, all_detections in enumerate(detections_list):
        with open(csv_path.format(tag, score_thresholds[index], scale), 'w') as file:
            file.write("patientId,PredictionString\n")

            for i in range(len(generator)):
                patientId = os.path.basename(generator.image_names[i]).split('.')[0]
                csv_line = '{},'.format(patientId)

                detections = all_detections[i][0] # specific to rsna
                for d in detections:
                    # d is a_pt + score
                    left = d[0] + (1 - scale) * (d[2] - d[0]) / 2
                    upper = d[1] + (1 - scale) * (d[3] - d[1]) / 2

                    w = (d[2] - d[0]) * scale
                    h = (d[3] - d[1]) * scale
                    

                    d_element = '{} {} {} {} {}'.format(
                        d[4],
                        int(left),
                        int(upper),
                        int(w),
                        int(h)
                    )
                    csv_line = '{} {}'.format(csv_line, d_element)

                csv_line = '{}\n'.format(csv_line)
                file.write(csv_line)