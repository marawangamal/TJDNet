import unittest

from utils.utils import group_arr


class TestGroupResults(unittest.TestCase):
    def setUp(self):
        # Sample test data
        self.test_data = [
            {
                "name": "model1",
                "model_head": "cp",
                "horizon": 1,
                "rank": 2,
                "value": 10,
            },
            {
                "name": "model2",
                "model_head": "cp",
                "horizon": 1,
                "rank": 4,
                "value": 20,
            },
            {
                "name": "model3",
                "model_head": "cp",
                "horizon": 2,
                "rank": 2,
                "value": 30,
            },
            {
                "name": "model4",
                "model_head": "ucp",
                "horizon": 1,
                "rank": 2,
                "value": 40,
            },
            {
                "name": "model5",
                "model_head": "ucp",
                "horizon": 2,
                "rank": 4,
                "value": 50,
            },
            {
                "name": "model6",
                "model_head": "mps",
                "horizon": 1,
                "rank": 2,
                "value": 60,
            },
        ]

    def test_single_level_grouping(self):
        # Test grouping by only model_head
        grouped = group_arr(self.test_data, lambda x: x["model_head"])

        # Verify structure
        self.assertEqual(set(grouped.keys()), {"cp", "ucp", "mps"})
        self.assertEqual(len(grouped["cp"]), 3)
        self.assertEqual(len(grouped["ucp"]), 2)
        self.assertEqual(len(grouped["mps"]), 1)

        # Check content
        self.assertEqual(grouped["cp"][0]["name"], "model1")
        self.assertEqual(grouped["ucp"][0]["name"], "model4")
        self.assertEqual(grouped["mps"][0]["name"], "model6")

    def test_two_level_grouping(self):
        # Test grouping by model_head and horizon
        grouped = group_arr(
            self.test_data, lambda x: x["model_head"], lambda x: f"h={x['horizon']}"
        )

        # Verify structure
        self.assertEqual(set(grouped.keys()), {"cp", "ucp", "mps"})
        self.assertEqual(set(grouped["cp"].keys()), {"h=1", "h=2"})
        self.assertEqual(set(grouped["ucp"].keys()), {"h=1", "h=2"})
        self.assertEqual(set(grouped["mps"].keys()), {"h=1"})

        # Check content
        self.assertEqual(len(grouped["cp"]["h=1"]), 2)
        self.assertEqual(len(grouped["cp"]["h=2"]), 1)
        self.assertEqual(grouped["cp"]["h=1"][0]["name"], "model1")
        self.assertEqual(grouped["cp"]["h=1"][1]["name"], "model2")
        self.assertEqual(grouped["mps"]["h=1"][0]["value"], 60)

    def test_three_level_grouping(self):
        # Test grouping by model_head, horizon, and rank
        grouped = group_arr(
            self.test_data,
            lambda x: x["model_head"],
            lambda x: f"h={x['horizon']}",
            lambda x: f"r={x['rank']}",
        )

        # Verify structure
        self.assertEqual(set(grouped["cp"]["h=1"].keys()), {"r=2", "r=4"})
        self.assertEqual(set(grouped["cp"]["h=2"].keys()), {"r=2"})

        # Check content
        self.assertEqual(len(grouped["cp"]["h=1"]["r=2"]), 1)
        self.assertEqual(grouped["cp"]["h=1"]["r=2"][0]["name"], "model1")
        self.assertEqual(grouped["ucp"]["h=2"]["r=4"][0]["value"], 50)

    def test_empty_input(self):
        # Test with empty list
        grouped = group_arr([], lambda x: x["model_head"])
        self.assertEqual(grouped, {})

    def test_custom_grouping(self):
        # Test with custom grouping function
        grouped = group_arr(
            self.test_data, lambda x: "high" if x["value"] > 30 else "low"
        )

        self.assertEqual(set(grouped.keys()), {"high", "low"})
        self.assertEqual(len(grouped["high"]), 3)
        self.assertEqual(len(grouped["low"]), 3)

    def test_missing_key(self):
        # Test with a grouping function that might cause KeyError
        with self.assertRaises(KeyError):
            group_arr(self.test_data, lambda x: x["non_existent_key"])


if __name__ == "__main__":
    unittest.main()
