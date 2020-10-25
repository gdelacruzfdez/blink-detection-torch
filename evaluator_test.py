import unittest
import pandas as pd

import evaluator

class TestBlinkEvaluator(unittest.TestCase):

    def setUp(self):
        self.evaluator = evaluator.BlinkEvaluator()
        self.dataframe = pd.read_csv('test_dataframe.csv')
        self.blinks_video_1 = [
            evaluator.Blink(1,  4, False, False, 1), 
            evaluator.Blink(6, 7, False, True, 1), 
            evaluator.Blink(8, 10, False, True, 1), 
            evaluator.Blink(13, 14, True, False, 1), 
            evaluator.Blink(17, 19, False, True, 1)
        ]
        self.blinks_video_2= [
            evaluator.Blink(0, 1, False, True, 2), 
            evaluator.Blink(4, 7, False, False, 2), 
            evaluator.Blink(11, 14, False, False, 2), 
            evaluator.Blink(18, 21, False, True, 2), 
            evaluator.Blink(24, 27, False, True, 2), 
        ]

    def test_separate_left_right_eyes(self):
        self.assertEqual(len(self.dataframe), 100)
        left_eyes, right_eyes = self.evaluator.separate_left_right_eyes(self.dataframe)
        self.assertEqual(len(left_eyes), 50)
        self.assertEqual(len(right_eyes), 50)
        self.assertEqual(len(left_eyes[left_eyes['eye'] == 'LEFT']), 50)
        self.assertEqual(len(right_eyes[right_eyes['eye'] == 'RIGHT']), 50)

    def test_extract_blinks_per_video(self):
        left_eyes, right_eyes = self.evaluator.separate_left_right_eyes(self.dataframe)
        left_gt_blinks, left_pred_blinks = self.evaluator.extract_blinks_per_video(left_eyes)
        right_gt_blinks, right_pred_blinks = self.evaluator.extract_blinks_per_video(right_eyes)
        self.assertEqual(len(left_gt_blinks), 2)
        self.assertEqual(len(left_pred_blinks), 2)
        self.assertEqual(len(right_gt_blinks), 2)
        self.assertEqual(len(right_pred_blinks), 2)

        num_gt_per_video = [3, 5]
        num_pred_per_video = [3, 5]
        for video in range(2):
            self.assertEqual(len(left_gt_blinks[video]), num_gt_per_video[video])
            self.assertEqual(len(left_pred_blinks[video]), num_pred_per_video[video])
            self.assertEqual(len(right_gt_blinks[video]), num_gt_per_video[video])
            self.assertEqual(len(right_pred_blinks[video]), num_pred_per_video[video])

    def test_calculate_blinks_for_video(self):
        left_eyes, right_eyes = self.evaluator.separate_left_right_eyes(self.dataframe)
        num_gt_per_video = [3, 5]
        num_pred_per_video = [3, 5]
        for video in range(2):
            left_eyes_of_video = left_eyes[left_eyes['video'] == video + 1].reset_index()
            right_eyes_of_video = right_eyes[right_eyes['video'] == video + 1].reset_index()
            gt_left, pred_left = self.evaluator.calculate_blinks_for_video(left_eyes_of_video)
            gt_right, pred_right = self.evaluator.calculate_blinks_for_video(right_eyes_of_video)
            self.assertEqual(len(gt_left), num_gt_per_video[video])
            self.assertEqual(len(pred_left), num_pred_per_video[video])
            self.assertEqual(len(gt_right), num_gt_per_video[video])
            self.assertEqual(len(pred_right), num_pred_per_video[video])

    def test_convert_annotation_to_blinks(self):
        left_eyes, _ = self.evaluator.separate_left_right_eyes(self.dataframe)
        num_total_blinks_per_video = [5, 5]
        for video in range(2):
            left_eyes_of_video = left_eyes[left_eyes['video'] == video + 1].reset_index()
            left_blinks = self.evaluator.convert_annotation_to_blinks(left_eyes_of_video)
            self.assertEqual(len(left_blinks), num_total_blinks_per_video[video])
    
    def test_convert_predictions_to_blinks(self):
        left_eyes, _ = self.evaluator.separate_left_right_eyes(self.dataframe)
        num_total_blinks_per_video = [3, 5]
        for video in range(2):
            left_eyes_of_video = left_eyes[left_eyes['video'] == video + 1].reset_index()
            left_blinks = self.evaluator.convert_predictions_to_blinks(left_eyes_of_video)
            self.assertEqual(len(left_blinks), num_total_blinks_per_video[video])
    
    def test_delete_non_visible_blinks(self):
        self.assertEqual(len(self.blinks_video_1), 5)
        gt_visible_blinks = list(filter(lambda blink: not blink.not_visible, self.blinks_video_1))
        pred_visible_blinks = self.evaluator.delete_non_visible_blinks(self.blinks_video_1)
        
        self.assertEqual(len(gt_visible_blinks), len(pred_visible_blinks))

    def test_merge_double_blinks(self):
        merged_blinks = self.evaluator.merge_double_blinks(self.blinks_video_1)
        self.assertEqual(len(merged_blinks), 4)
        merged_blink = merged_blinks[1]
        self.assertEqual(merged_blink.start, 6)
        self.assertEqual(merged_blink.end, 10)

    def test_calculate_global_confussion_matrix(self):
       pass 




        


