import os
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import pandas as pd

from src.evaluation import split_robustness as SR


def make_population(n_groups=120, rows_per_group=6, seed=7):
    rng = np.random.default_rng(seed)
    group_label = rng.choice(["BENIGN", "DoS Hulk", "PortScan"], size=n_groups, p=[0.6, 0.25, 0.15])
    groups = np.repeat(np.arange(n_groups), rows_per_group)
    labels = np.repeat(group_label, rows_per_group)
    return labels, groups


class SplitRobustnessTests(unittest.TestCase):
    def setUp(self):
        self.labels, self.groups = make_population()
        self.n = len(self.labels)

    def test_each_row_is_in_exactly_one_outer_test_fold(self):
        for scheme in SR.SCHEMES:
            with self.subTest(scheme=scheme):
                ids = SR.assign_outer_folds(self.labels, self.groups, scheme)
                self.assertEqual(len(ids), self.n)
                self.assertEqual(set(ids.tolist()), set(range(SR.OUTER_SPLITS)))
                counts = np.bincount(ids, minlength=SR.OUTER_SPLITS)
                self.assertEqual(int(counts.sum()), self.n)

    def test_train_val_test_are_disjoint_and_cover_all_rows(self):
        for scheme in SR.SCHEMES:
            ids = SR.assign_outer_folds(self.labels, self.groups, scheme)
            for k in range(SR.OUTER_SPLITS):
                with self.subTest(scheme=scheme, fold=k):
                    train, val, test = SR.split_fold(self.labels, self.groups, ids, k, scheme)
                    parts = [set(train.tolist()), set(val.tolist()), set(test.tolist())]
                    self.assertEqual(sum(len(p) for p in parts), self.n)
                    self.assertEqual(set().union(*parts), set(range(self.n)))
                    self.assertTrue(all(len(p) > 0 for p in parts))

    def test_test_fold_is_exactly_the_outer_fold_rows(self):
        for scheme in SR.SCHEMES:
            ids = SR.assign_outer_folds(self.labels, self.groups, scheme)
            for k in range(SR.OUTER_SPLITS):
                _, _, test = SR.split_fold(self.labels, self.groups, ids, k, scheme)
                self.assertEqual(sorted(test.tolist()), np.flatnonzero(ids == k).tolist())

    def test_grouped_never_shares_a_group_across_partitions(self):
        ids = SR.assign_outer_folds(self.labels, self.groups, "grouped")
        for k in range(SR.OUTER_SPLITS):
            with self.subTest(fold=k):
                train, val, test = SR.split_fold(self.labels, self.groups, ids, k, "grouped")
                g_train, g_val, g_test = (set(self.groups[i].tolist()) for i in (train, val, test))
                self.assertFalse(g_train & g_val)
                self.assertFalse(g_train & g_test)
                self.assertFalse(g_val & g_test)
                self.assertEqual(SR.pct_rows_sharing_groups(test, train, self.groups), 0.0)
                self.assertEqual(SR.pct_rows_sharing_groups(test, val, self.groups), 0.0)

    def test_random_shares_groups_between_train_and_test(self):
        ids = SR.assign_outer_folds(self.labels, self.groups, "random")
        for k in range(SR.OUTER_SPLITS):
            with self.subTest(fold=k):
                train, _, test = SR.split_fold(self.labels, self.groups, ids, k, "random")
                shared = set(self.groups[train].tolist()) & set(self.groups[test].tolist())
                self.assertGreater(len(shared), 0)
                self.assertGreater(SR.pct_rows_sharing_groups(test, train, self.groups), 0.0)

    def test_group_ids_follow_day_ip_and_five_minute_block(self):
        df = pd.DataFrame({
            "day_group": ["Monday", "Monday", "Monday", "Tuesday"],
            "Source IP": ["10.0.0.1", "10.0.0.1", "10.0.0.1", "10.0.0.1"],
            "Timestamp": pd.to_datetime([
                "2017-07-03 09:01:00", "2017-07-03 09:04:59",
                "2017-07-03 09:05:00", "2017-07-04 09:01:00",
            ]),
        })
        ids = SR.build_group_ids(df)
        self.assertEqual(ids[0], ids[1])
        self.assertNotEqual(ids[1], ids[2])
        self.assertNotEqual(ids[0], ids[3])


FINGERPRINT = {"n_rows": 10, "label_hash": "a", "network_cols": ["x"]}
AE_CFG = {"hidden_dims": [4], "bottleneck_dim": 2}
FUSION_CFG = {"confidence_threshold": 0.95, "fused_decision_threshold": 0.03}


def make_rows(row_ids, labels, in_train=None, seed=0):
    rng = np.random.default_rng(seed)
    n = len(row_ids)
    labels = np.asarray(labels)
    y = (labels != "BENIGN").astype(int)
    data = {"row_id": np.asarray(row_ids), "source_row": np.asarray(row_ids) * 10,
            "Label": labels, "y_true": y}
    for model in SR.MODEL_NAMES:
        data[f"{model}_score"] = np.clip(y * 0.6 + rng.random(n) * 0.4, 0, 1)
        data[f"{model}_pred"] = y
    data["class_in_train"] = np.ones(n, dtype=bool) if in_train is None else np.asarray(in_train)
    return pd.DataFrame(data)


def make_meta(fold, rows, signature="sig", scheme="random"):
    metrics, per_class = SR.evaluate_rows(rows)
    for model in SR.MODEL_NAMES:
        for stats in per_class[model].values():
            stats["n_train"] = 5
    return {
        "run_signature": signature, "scheme": scheme, "fold": fold, "n_test": len(rows),
        "n_test_attack": int(rows["y_true"].sum()), "metrics": metrics, "per_class": per_class,
        "pct_test_rows_group_in_train": 0.0, "pct_test_rows_group_in_val": 0.0,
        "pct_train_rows_group_in_val": 0.0, "held_out_classes": {},
        "class_counts_train": {"BENIGN": 5, "DoS Hulk": 5},
    }


class CheckpointTests(unittest.TestCase):
    def test_signature_changes_with_test_rows_and_data(self):
        a = SR.run_signature("random", 0, np.array([1, 2, 3]), FINGERPRINT, AE_CFG, FUSION_CFG)
        b = SR.run_signature("random", 0, np.array([1, 2, 4]), FINGERPRINT, AE_CFG, FUSION_CFG)
        c = SR.run_signature("random", 0, np.array([1, 2, 3]), {**FINGERPRINT, "label_hash": "b"},
                             AE_CFG, FUSION_CFG)
        again = SR.run_signature("random", 0, np.array([1, 2, 3]), FINGERPRINT, AE_CFG, FUSION_CFG)
        self.assertEqual(a, again)
        self.assertEqual(len({a, b, c}), 3)

    def test_load_rejects_checkpoint_for_different_test_rows(self):
        test_idx = np.array([3, 5, 8])
        rows = make_rows(test_idx, ["BENIGN", "DoS Hulk", "BENIGN"])
        meta = make_meta(0, rows, signature="abc")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            SR.save_checkpoint(root, meta, rows)
            self.assertIsNotNone(SR.load_checkpoint(root, "random", 0, "abc", test_idx))
            self.assertIsNone(SR.load_checkpoint(root, "random", 0, "abc", np.array([3, 5, 9])))
            self.assertIsNone(SR.load_checkpoint(root, "random", 0, "other", test_idx))

    def test_checkpoints_of_different_signatures_do_not_overwrite(self):
        rows = make_rows([1, 2], ["BENIGN", "DoS Hulk"])
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            SR.save_checkpoint(root, make_meta(0, rows, signature="s1"), rows)
            SR.save_checkpoint(root, make_meta(0, rows, signature="s2"), rows)
            self.assertIsNotNone(SR.load_checkpoint(root, "random", 0, "s1", [1, 2]))
            self.assertIsNotNone(SR.load_checkpoint(root, "random", 0, "s2", [1, 2]))


class AggregationTests(unittest.TestCase):
    def test_pooling_counts_each_row_once(self):
        rows0 = make_rows([0, 1, 2, 3], ["BENIGN", "DoS Hulk", "BENIGN", "DoS Hulk"])
        rows1 = make_rows([4, 5, 6], ["BENIGN", "BENIGN", "DoS Hulk"], seed=1)
        results = {"random": [(make_meta(0, rows0), rows0), (make_meta(1, rows1), rows1)]}
        exclude_ids, by_class = SR.excluded_rows(results)
        fold_table, class_table = SR.build_tables(results, exclude_ids)
        summary = SR.build_summary(results, fold_table, exclude_ids, by_class)
        block = summary["schemes"]["random"]
        self.assertEqual(block["n_pooled_rows"], 7)
        self.assertEqual(block["folds"], [0, 1])
        pooled_class = class_table[(class_table["fold"] == "pooled") & (class_table["model"] == "rf")]
        self.assertEqual(int(pooled_class["n"].iloc[0]), 3)
        self.assertEqual(len(fold_table), 2 * len(SR.MODEL_NAMES))

    def test_absent_classes_are_excluded_from_seen_metrics(self):
        in_train = [True, True, False, False]
        rows = make_rows([0, 1, 2, 3], ["BENIGN", "DoS Hulk", "Rare", "Rare"], in_train=in_train)
        rows["rf_pred"] = [0, 1, 0, 0]
        _, per_class = SR.evaluate_rows(rows)
        self.assertEqual(per_class["rf"]["Rare"]["n"], 2)
        self.assertEqual(per_class["rf"]["Rare"]["n_seen"], 0)
        self.assertTrue(np.isnan(per_class["rf"]["Rare"]["recall_seen"]))
        all_metrics, _ = SR.evaluate_rows(rows)
        seen_metrics, _ = SR.evaluate_rows(rows, seen_only=True)
        self.assertLess(all_metrics["rf"]["recall"], seen_metrics["rf"]["recall"])


class ArgsAndSubsampleTests(unittest.TestCase):
    def test_duplicate_folds_and_schemes_are_deduplicated(self):
        args = SR.parse_args(["--folds", "2", "0", "2", "0", "--schemes", "grouped", "random", "grouped"])
        self.assertEqual(args.folds, [0, 2])
        self.assertEqual(args.schemes, ["random", "grouped"])

    def test_subsample_keeps_min_rows_per_class_and_source_positions(self):
        labels = ["BENIGN"] * 900 + ["DoS Hulk"] * 90 + ["Rare"] * 8
        df = pd.DataFrame({"Label": labels, "v": np.arange(len(labels))})
        out = SR.stratified_subsample(df, 100)
        counts = out["Label"].value_counts()
        self.assertEqual(counts["Rare"], 8)
        self.assertGreaterEqual(counts["DoS Hulk"], SR.MIN_ROWS_PER_CLASS_SMOKE)
        self.assertTrue((df.loc[out.index, "v"].to_numpy() == out["v"].to_numpy()).all())
        self.assertTrue(out.index.is_unique)


class HeldOutClassTests(unittest.TestCase):
    def test_single_group_class_is_flagged_absent_from_train_in_its_test_fold(self):
        labels, groups = make_population()
        labels = np.concatenate([labels.astype(object), np.array(["Rare"] * 10, dtype=object)])
        groups = np.concatenate([groups, np.full(10, groups.max() + 1)])
        ids = SR.assign_outer_folds(labels, groups, "grouped")
        flagged = []
        for k in range(SR.OUTER_SPLITS):
            train, _, test = SR.split_fold(labels, groups, ids, k, "grouped")
            held = SR.held_out_classes(SR.class_counts(labels, train), SR.class_counts(labels, test))
            if held:
                flagged.append((k, held))
        self.assertEqual(len(flagged), 1)
        self.assertEqual(flagged[0][1], {"Rare": 10})

    def test_fail_fast_on_class_with_one_train_row(self):
        with self.assertRaises(ValueError):
            SR.check_train_class_counts({"BENIGN": 50, "Rare": 1})
        SR.check_train_class_counts({"BENIGN": 50, "Rare": 0, "Other": 2})


class CommonSeenPopulationTests(unittest.TestCase):
    def make_results(self):
        labels = {i: "BENIGN" if i % 3 else "DoS Hulk" for i in range(12)}
        labels[4] = labels[5] = "Rare"

        def fold_rows(ids, absent_ids):
            ids = list(ids)
            rows = make_rows(ids, [labels[i] for i in ids],
                             in_train=[i not in absent_ids for i in ids])
            rare = rows["Label"] == "Rare"
            for model in SR.MODEL_NAMES:
                rows.loc[rare, f"{model}_pred"] = 0
            return rows

        grouped = [fold_rows(range(0, 6), {4, 5}), fold_rows(range(6, 12), set())]
        random_ = [fold_rows(range(0, 12, 2), set()), fold_rows(range(1, 12, 2), set())]
        return {
            "grouped": [(make_meta(k, r, scheme="grouped"), r) for k, r in enumerate(grouped)],
            "random": [(make_meta(k, r), r) for k, r in enumerate(random_)],
        }

    def test_both_schemes_exclude_the_same_rows(self):
        results = self.make_results()
        exclude_ids, by_class = SR.excluded_rows(results)
        self.assertEqual(sorted(exclude_ids.tolist()), [4, 5])
        self.assertEqual(by_class, {"Rare": 2})
        fold_table, class_table = SR.build_tables(results, exclude_ids)
        summary = SR.build_summary(results, fold_table, exclude_ids, by_class)
        schemes = summary["schemes"]
        self.assertEqual(schemes["random"]["n_rows_in_seen_population"], 10)
        self.assertEqual(schemes["grouped"]["n_rows_in_seen_population"], 10)
        self.assertEqual(summary["common_excluded_rows"], {"n": 2, "by_class": {"Rare": 2}})

    def test_delta_seen_is_computed_on_the_common_rows(self):
        results = self.make_results()
        exclude_ids, by_class = SR.excluded_rows(results)
        fold_table, _ = SR.build_tables(results, exclude_ids)
        summary = SR.build_summary(results, fold_table, exclude_ids, by_class)
        delta = summary["pooled_delta_grouped_minus_random"]["excluding_absent_classes"]
        for model in SR.MODEL_NAMES:
            self.assertEqual(delta[model]["recall"], 0.0)
        random_pooled = pd.concat([rows for _, rows in results["random"]], ignore_index=True)
        own_flag, _ = SR.evaluate_rows(random_pooled, seen_only=True)
        common, _ = SR.evaluate_rows(random_pooled, exclude_ids, seen_only=True)
        self.assertLess(own_flag["rf"]["recall"], common["rf"]["recall"])

    def test_per_class_seen_counts_use_the_common_set_in_both_schemes(self):
        results = self.make_results()
        exclude_ids, _ = SR.excluded_rows(results)
        _, class_table = SR.build_tables(results, exclude_ids)
        pooled = class_table[(class_table["fold"] == "pooled") & (class_table["model"] == "rf")]
        rare = pooled[pooled["label"] == "Rare"].set_index("scheme")
        self.assertEqual(int(rare.loc["random", "n_excluded_common"]), 2)
        self.assertEqual(int(rare.loc["grouped", "n_excluded_common"]), 2)


class ComparisonValidityTests(unittest.TestCase):
    def summarize(self, results):
        exclude_ids, by_class = SR.excluded_rows(results)
        fold_table, _ = SR.build_tables(results, exclude_ids)
        return SR.build_summary(results, fold_table, exclude_ids, by_class)

    def scheme_entries(self, folds_ids, scheme):
        out = []
        for k, ids in folds_ids:
            rows = make_rows(ids, ["BENIGN" if i % 2 else "DoS Hulk" for i in ids])
            out.append((make_meta(k, rows, scheme=scheme), rows))
        return out

    def test_partial_and_mismatched_folds_omit_deltas(self):
        results = {
            "random": self.scheme_entries([(0, [0, 1, 2, 3])], "random"),
            "grouped": self.scheme_entries([(0, [4, 5, 6, 7, 8])], "grouped"),
        }
        summary = self.summarize(results)
        self.assertFalse(summary["comparison_valid"])
        self.assertNotIn("pooled_delta_grouped_minus_random", summary)
        self.assertIn("different row sets", summary["comparison_reason"])
        _, class_table = SR.build_tables(results, SR.excluded_rows(results)[0])
        text = "\n".join(SR.comparison_text(summary, class_table))
        self.assertIn("WARNING", text)
        self.assertNotIn("delta_seen", text)

    def test_full_coverage_emits_deltas(self):
        results = {
            "random": self.scheme_entries([(0, [0, 2, 4]), (1, [1, 3, 5])], "random"),
            "grouped": self.scheme_entries([(0, [0, 1, 2]), (1, [3, 4, 5])], "grouped"),
        }
        summary = self.summarize(results)
        self.assertTrue(summary["comparison_valid"])
        self.assertIn("pooled_delta_grouped_minus_random", summary)
        _, class_table = SR.build_tables(results, SR.excluded_rows(results)[0])
        text = "\n".join(SR.comparison_text(summary, class_table))
        self.assertIn("delta_seen", text)
        self.assertNotIn("WARNING", text)

    def test_single_scheme_is_not_comparable(self):
        summary = self.summarize({"random": self.scheme_entries([(0, [0, 1, 2])], "random")})
        self.assertFalse(summary["comparison_valid"])
        self.assertNotIn("pooled_delta_grouped_minus_random", summary)


if __name__ == "__main__":
    unittest.main()
