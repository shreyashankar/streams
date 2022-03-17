import unittest

from streams import STREAMSDataset


class TestDataset(unittest.TestCase):
    def setUp(self) -> None:
        self.T = 100
        self.inference_window = 10

    def testDatasetProperties(self) -> None:
        ds = STREAMSDataset(
            "test", T=self.T, inference_window=self.inference_window
        )

        self.assertTrue(ds.step == 1)
        self.assertTrue(len(ds.permutation) == self.T)
        self.assertTrue(len(ds) == self.T)

    def testCreateBadDataset(self) -> None:
        with self.assertRaises(ValueError):
            STREAMSDataset("bad_dataset")

    def testAdvance(self) -> None:
        ds = STREAMSDataset(
            "test", T=self.T, inference_window=self.inference_window
        )

        ds.advance(1)
        ds.advance(99)

        with self.assertRaises(ValueError):
            ds.advance(100)

        with self.assertRaises(ValueError):
            ds.advance(-1)

        self.assertTrue(ds.step == (self.T - self.inference_window))

    def testPeekIntoFuture(self) -> None:
        ds = STREAMSDataset(
            "test", T=self.T, inference_window=self.inference_window
        )

        ds.advance(10)

        with self.assertRaises(ValueError):
            ds[80]

        # This should pass
        ds.get(90, future_ok=True)

    def testGetData(self) -> None:
        ds = STREAMSDataset(
            "test", T=self.T, inference_window=self.inference_window
        )

        train_data, test_data = ds.get_data(include_test=True)
        self.assertTrue(len(train_data) == ds.step)
        self.assertTrue(len(test_data) == ds.inference_window)

        self.assertTrue((train_data[0][0] == ds[0][0]).all())

    def testGet(self) -> None:
        ds = STREAMSDataset(
            "test", T=self.T, inference_window=self.inference_window
        )

        # This should work
        ds.get(0)
        ds.get(50, future_ok=True)

        # This should not work
        with self.assertRaises(ValueError):
            ds.get(self.T - 1)

        # After stepping, this should work
        ds.advance(self.T)
        ds.get(50)
