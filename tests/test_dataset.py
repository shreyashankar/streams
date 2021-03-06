import unittest

from streams import STREAMSDataset


class TestDataset(unittest.TestCase):
    def setUp(self) -> None:
        self.T = 10
        self.inference_window = 1

    def testDatasetProperties(self) -> None:
        ds = STREAMSDataset("test", T=self.T, inference_window=self.inference_window)

        self.assertTrue(ds.step == 0)
        self.assertTrue(len(ds.sample_history) == self.T)
        self.assertTrue(len(ds) == self.T)

    def testSignalStart(self) -> None:
        starting_time_steps = { (0, 9): 5, (0, 0): 9 }
        ds = STREAMSDataset("test", T=10, log_step=1, start_max=10, duration=1,
                starting_time_steps=starting_time_steps)

        complete_sample_history = []

        for t in range(ds._T):
            for (i,j) in starting_time_steps:
                if starting_time_steps[(i,j)] > t:
                    self.assertNotIn(j, ds.sample_history[t])

            complete_sample_history.extend(ds.sample_history[t])

        self.assertIn(5, complete_sample_history)
        self.assertIn(9, complete_sample_history)

    def testTrainTestLeakage(self) -> None:
        ds = STREAMSDataset("test", T=self.T, inference_window=self.inference_window)

        for t in range(ds._T):
            self.assertEqual(set([]), set(ds.sample_history[t]) & set(ds.oracle_training_data[t]))

    def testCreateBadDataset(self) -> None:
        with self.assertRaises(ValueError):
            STREAMSDataset("bad_dataset")

    def testAdvance(self) -> None:
        ds = STREAMSDataset("test", T=self.T, inference_window=self.inference_window)

        ds.advance(8)

        # can only go up to 8; time step 9 has to be left for inference
        with self.assertRaises(ValueError):
            ds.advance(1)

        with self.assertRaises(ValueError):
            ds.advance(-1)

        self.assertTrue(ds.step == (self.T - 1 - self.inference_window))

    def testPeekIntoFuture(self) -> None:
        ds = STREAMSDataset("test", T=self.T, inference_window=self.inference_window)

        ds.advance(1)

        with self.assertRaises(ValueError):
            ds[[8]]

        # This should pass
        ds.get([8], future_ok=True)

    def testGetData(self) -> None:
        ds = STREAMSDataset("test", T=self.T, inference_window=self.inference_window)

        train_data, test_data = ds.get_data(include_test=True)
        self.assertTrue((train_data[0][0] == ds.get([ds.step])[0][0]).all())
        self.assertTrue(
            (
                test_data[0][0]
                == ds.get([ds.step + self.inference_window], future_ok=True)[0][0]
            ).all()
        )

    def testGet(self) -> None:
        ds = STREAMSDataset("test", T=self.T, inference_window=self.inference_window)

        # This should work
        ds.get([0])
        ds.get([5], future_ok=True)

        # This should not work
        with self.assertRaises(ValueError):
            ds.get([5])

        # After stepping, this should work
        ds.advance(5)
        ds.get([5])


if __name__ == "__main__":
    unittest.main()
