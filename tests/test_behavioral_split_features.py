import unittest

import pandas as pd

from src.features import validate_behavioral_join


class BehavioralSplitFeatureTests(unittest.TestCase):
    def test_rolling_window_uses_only_events_in_the_current_split(self):
        timestamps = pd.to_datetime(
            ["2017-07-03 09:00:00", "2017-07-03 09:00:30", "2017-07-03 09:02:00"]
        )
        behavioral = pd.DataFrame(
            {
                "flow_marker": [1.0, 2.0, 3.0],
                "Label": ["BENIGN", "BENIGN", "BENIGN"],
                "Source IP": ["192.0.2.1"] * 3,
                "Timestamp": timestamps,
                "Destination Port": [80, 80, 443],
                "Protocol": [6, 6, 6],
                "RST Flag Count": [0, 0, 0],
                "SYN Flag Count": [1, 1, 1],
                "ACK Flag Count": [0, 0, 0],
                "Total Length of Fwd Packets": [60, 60, 60],
                "Total Length of Bwd Packets": [0, 0, 0],
                "day_group": ["Monday"] * 3,
            }
        )
        target_split = behavioral.loc[[1], ["flow_marker", "Label"]].reset_index(drop=True)

        result, dropped = validate_behavioral_join.apply_join(
            target_split, behavioral, ["flow_marker", "Label"], "test"
        )

        self.assertEqual(dropped, 0)
        self.assertEqual(len(result), 1)
        self.assertEqual(int(result.loc[0, "conn_count_1m"]), 1)
        self.assertEqual(int(result.loc[0, "uniq_dport_count_1m"]), 1)


if __name__ == "__main__":
    unittest.main()
