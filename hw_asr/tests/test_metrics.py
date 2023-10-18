from typing import NamedTuple
import unittest

from hw_asr.metric.utils import calc_cer, calc_wer


class TextsMetricsTestCase(NamedTuple):
    target: str
    predicted: str
    metric_value: float


class TestMetrics(unittest.TestCase):
    def test_cer(self):
        test_cases = [
            TextsMetricsTestCase(
                target='',
                predicted='not empty',
                metric_value=1.,
            ),
            TextsMetricsTestCase(
                target='not empty',
                predicted='',
                metric_value=1.,
            ),
            TextsMetricsTestCase(
                target='aaaaa',
                predicted='ababa',
                metric_value=0.4,
            ),
            TextsMetricsTestCase(
                target='a',
                predicted='aaa',
                metric_value=2.,
            )
        ]

        for test_case in test_cases:
            calculated_metric_value = calc_cer(test_case.target, test_case.predicted)
            self.assertAlmostEqual(calculated_metric_value, test_case.metric_value)

    def test_wer(self):
        test_cases = [
            TextsMetricsTestCase(
                target='',
                predicted='not empty',
                metric_value=1.,
            ),
            TextsMetricsTestCase(
                target='not empty',
                predicted='',
                metric_value=1.,
            ),
            TextsMetricsTestCase(
                target='I am a student',
                predicted='I a good stjudent',
                metric_value=0.75,
            ),
            TextsMetricsTestCase(
                target='a',
                predicted='a b c d',
                metric_value=3.,
            )
        ]

        for test_case in test_cases:
            calculated_metric_value = calc_wer(test_case.target, test_case.predicted)
            self.assertAlmostEqual(calculated_metric_value, test_case.metric_value)
