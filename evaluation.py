import sys
import argparse
import os
import time
import numpy as np
import cv2
from glob import glob
import progressbar
from time import sleep
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score

def seg_multi_class(test_mask_dir, truth_mask_dir, weighting):

    test_masks = glob(os.path.join(test_mask_dir, '*.png'))
    test_masks.extend(glob(os.path.join(test_mask_dir, '*.jpg')))
    truth_masks = glob(os.path.join(truth_mask_dir, '*.png'))
    truth_masks.extend(glob(os.path.join(truth_mask_dir, '*.jpg')))

    print ('[INFO] Performing segmentation evaluation...')
    print ('[INFO] %.0f images to process' %len(test_masks))

    bar = progressbar.ProgressBar(maxval=len(test_masks), \
    widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
    bar.start()

    precision = 0.0
    recall = 0.0
    accuracy = 0.0
    f1 = 0.0

    num_evals = 0

    for i, test_img in enumerate(test_masks):

        bar.update(i+1)

        pred_im = cv2.imread(test_img, 0)

        for truth_img in truth_masks:
                if os.path.splitext(os.path.basename(truth_img))[0] == os.path.splitext(os.path.basename(test_img))[0]:
                        gt_im = cv2.imread(truth_img, 0)
                        if pred_im.size != gt_im.size:
                            raise TypeError("Images must be matching sizes")

                        if weighting == 'binary':
                            pred_im[pred_im>0] = 1
                            gt_im[gt_im>0] = 1

                        precision += precision_score(gt_im.reshape(-1, 1), pred_im.reshape(-1, 1), pos_label=1, average=weighting)
                        recall += recall_score(gt_im.reshape(-1, 1), pred_im.reshape(-1, 1), pos_label=1, average=weighting)
                        f1 += f1_score(gt_im.reshape(-1, 1), pred_im.reshape(-1, 1), pos_label=1, average=weighting)

                        accuracy += accuracy_score(gt_im.reshape(-1, 1), pred_im.reshape(-1, 1))

                        num_evals += 1

    return precision/num_evals, recall/num_evals, (accuracy/num_evals)*100, f1/num_evals


def seg_binary(test_mask_dir, truth_mask_dir):

    test_masks = glob(os.path.join(test_mask_dir, '*.png'))
    test_masks.extend(glob(os.path.join(test_mask_dir, '*.jpg')))
    truth_masks = glob(os.path.join(truth_mask_dir, '*.png'))
    truth_masks.extend(glob(os.path.join(truth_mask_dir, '*.jpg')))

    print ('[INFO] Performing segmentation evaluation...')
    print ('[INFO] %.0f images to process' %len(test_masks))

    bar = progressbar.ProgressBar(maxval=len(test_masks), \
    widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
    bar.start()

    num_evals = 0
    overall_pre = 0.0
    overall_rec = 0.0
    overall_acc = 0.0
    overall_f1 = 0.0

    for i, test_img in enumerate(test_masks):

        bar.update(i+1)

        tp = 0
        tn = 0
        fp = 0
        fn = 0
        pred_im = cv2.imread(test_img, 0)
        pred_im[pred_im>0] = 255

        for truth_img in truth_masks:
            if os.path.splitext(os.path.basename(truth_img))[0] == os.path.splitext(os.path.basename(test_img))[0]:
                gt_im = cv2.imread(truth_img, 0)
                gt_im[gt_im>0] = 255
                if pred_im.size != gt_im.size:
                    raise TypeError("Images must be matching sizes")

                tp = float(len(np.where((pred_im==255)&(gt_im==255))[0]))
                tn = float(len(np.where((pred_im==0)&(gt_im==0))[0]))
                fp = float(len(np.where((pred_im==255)&(gt_im==0))[0]))
                fn = float(len(np.where((pred_im==0)&(gt_im==255))[0]))

                if tp != 0 and tn != 0:
                    accuracy = (tp+tn)/(tp+fp+fn+tn)
                    precision = tp/(tp+fp)
                    recall = tp/(tp+fn)
                    f1 = 2*((precision*recall)/(precision+recall))
                    overall_pre += precision
                    overall_rec += recall
                    overall_acc += accuracy
                    overall_f1 += f1
                    num_evals += 1
                else:
                    accuracy = 0
                    precision = 0
                    recall = 0
                    f1 = 0
    bar.finish()

    if num_evals != 0:
        overall_pre = overall_pre / num_evals
        overall_rec = overall_rec / num_evals
        overall_acc = overall_acc / num_evals
        overall_f1 = overall_f1 / num_evals
    else:
        overall_pre = 0
        overall_rec = 0
        overall_acc = 0
        overall_f1 = 0

    return overall_pre, overall_rec, overall_acc*100, overall_f1

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("-i",  "--test_images", help="Path to test image directory", required=True)
    parser.add_argument("-g", "--ground_truth_images", help="Path to ground truth image directory", required=True)
    parser.add_argument("-m", "--multi_class", help="Multiple classes present",default=False, action="store_true")
    args = parser.parse_args()

    if not os.path.exists(args.test_images):
        raise ValueError("Test image directory does not exist")

    if not os.path.exists(args.ground_truth_images):
        raise ValueError("Ground truth image directory does not exist")#

    print ('[INFO] Evaluating Results...')

    test_mask_dir = args.test_images
    truth_mask_dir = args.ground_truth_images
    multi_class = args.multi_class

    start = time.time()

    if multi_class == False:
        precision, recall, accuracy, f1 = seg_binary(test_mask_dir=test_mask_dir, truth_mask_dir=truth_mask_dir)
    else:
        precision, recall, accuracy, f1 = seg_multi_class(test_mask_dir=test_mask_dir, truth_mask_dir=truth_mask_dir, weighting='weighted')

    print ('--------------------------------------------------------')
    print ('[RESULTS] PRECISION: %.4f' % precision)
    print ('[RESULTS] RECALL: %.4f' % recall)
    print ('[RESULTS] ACCURACY: %.4f' % accuracy, '%')
    print ('[RESULTS] F1 VALUE: %.4f' % f1)
    print ('--------------------------------------------------------')
    print ('Processing Time %.2f seconds' % float(time.time()-start))


if __name__ == '__main__':
    main()
