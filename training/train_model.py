from __future__ import annotations

import argparse
import math
from pathlib import Path

from poker44_ml.inference import Poker44Model
from training.build_dataset import (
    DEFAULT_BENCHMARK_PATHS,
    DEFAULT_BOT_PATHS,
    DEFAULT_HUMAN_PATHS,
    build_training_dataframe,
    load_json_or_gz,
    load_public_benchmark_rows,
    resolve_existing_path,
)
from training.evaluate import evaluate_predictions, format_metrics

try:
    import joblib
except ImportError:  # pragma: no cover - surfaced only in incomplete runtime envs.
    joblib = None

try:
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.ensemble import ExtraTreesClassifier, VotingClassifier
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.model_selection import train_test_split
except ImportError:  # pragma: no cover - surfaced only in incomplete runtime envs.
    CalibratedClassifierCV = None
    ExtraTreesClassifier = None
    VotingClassifier = None
    HistGradientBoostingClassifier = None
    train_test_split = None

try:
    from xgboost import XGBClassifier
except ImportError:  # pragma: no cover - surfaced only in incomplete runtime envs.
    XGBClassifier = None


REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a fast chunk-level Poker44 miner model.")
    parser.add_argument("--human-path", type=str, default=None)
    parser.add_argument("--bot-path", type=str, default=None)
    parser.add_argument("--benchmark-path", type=str, default=None)
    parser.add_argument("--chunk-size", type=int, default=80)
    parser.add_argument("--min-chunk-size", type=int, default=40)
    parser.add_argument("--stride", type=int, default=40)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-estimators", type=int, default=300)
    parser.add_argument("--max-depth", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--subsample", type=float, default=0.9)
    parser.add_argument("--colsample-bytree", type=float, default=0.9)
    parser.add_argument("--calibration", choices=("auto", "isotonic", "sigmoid", "none"), default="auto")
    parser.add_argument(
        "--benchmark-oversample",
        type=int,
        default=3,
        help="How many times to repeat public benchmark train rows in the training mix.",
    )
    parser.add_argument(
        "--selection-objective",
        choices=("balanced", "low_fpr"),
        default="low_fpr",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(REPO_ROOT / "models" / "poker44_xgb_calibrated.joblib"),
    )
    parser.add_argument(
        "--search",
        action="store_true",
        help="Evaluate a small set of chunk/model configurations and keep the best artifact.",
    )
    parser.add_argument(
        "--search-budget",
        type=int,
        default=6,
        help="Maximum number of candidate configurations to evaluate when --search is enabled.",
    )
    return parser.parse_args()


def choose_calibration(method: str, train_size: int) -> str | None:
    if method == "none":
        return None
    if method == "auto":
        return "isotonic" if train_size >= 800 else "sigmoid"
    return method


def model_selection_score(metrics: dict[str, float], objective: str) -> float:
    if objective == "balanced":
        return (
            0.55 * metrics["validator_reward"]
            + 0.15 * metrics["roc_auc"]
            + 0.10 * metrics["pr_auc"]
            + 0.10 * metrics["validator_bot_recall"]
            - 0.10 * metrics["log_loss"]
            - 0.08 * metrics["brier_score"]
            - 0.15 * metrics["fpr_at_threshold_0_5"]
            - 0.10 * metrics["fpr_at_recall"]
        )
    return (
        0.45 * metrics["validator_reward"]
        + 0.10 * metrics["roc_auc"]
        + 0.10 * metrics["pr_auc"]
        - 0.10 * metrics["log_loss"]
        - 0.08 * metrics["brier_score"]
        - 0.85 * metrics["fpr_at_threshold_0_5"]
        - 0.95 * metrics["fpr_at_recall"]
    )


def apply_probability_postprocess(
    probs: list[float],
    *,
    bias: float,
    temperature: float,
) -> list[float]:
    adjusted: list[float] = []
    safe_temperature = max(0.25, float(temperature))
    for prob in probs:
        clipped = min(max(float(prob), 1e-6), 1.0 - 1e-6)
        logit = math.log(clipped / (1.0 - clipped))
        transformed = 1.0 / (1.0 + math.exp(-((logit / safe_temperature) + float(bias))))
        adjusted.append(float(transformed))
    return adjusted


def choose_best_probability_postprocess(
    *,
    y_true: list[int],
    y_prob: list[float],
    objective: str,
) -> tuple[list[float], dict[str, float], dict[str, float]]:
    candidate_settings = [
        {"bias": -1.00, "temperature": 1.00},
        {"bias": -0.75, "temperature": 1.00},
        {"bias": -0.50, "temperature": 1.00},
        {"bias": -0.35, "temperature": 1.00},
        {"bias": -0.25, "temperature": 1.10},
        {"bias": -0.15, "temperature": 1.05},
        {"bias": 0.00, "temperature": 1.00},
        {"bias": 0.10, "temperature": 0.95},
    ]

    best_probs = list(y_prob)
    best_metrics = evaluate_predictions(y_true=y_true, y_prob=y_prob)
    best_config = {"bias": 0.0, "temperature": 1.0}
    best_score = model_selection_score(best_metrics, objective)

    for setting in candidate_settings:
        transformed = apply_probability_postprocess(
            y_prob,
            bias=setting["bias"],
            temperature=setting["temperature"],
        )
        metrics = evaluate_predictions(y_true=y_true, y_prob=transformed)
        score = model_selection_score(metrics, objective)
        if score > best_score:
            best_score = score
            best_probs = transformed
            best_metrics = metrics
            best_config = dict(setting)

    return best_probs, best_metrics, best_config


def build_search_configs(args: argparse.Namespace) -> list[dict[str, float | int | str]]:
    base = {
        "chunk_size": args.chunk_size,
        "min_chunk_size": args.min_chunk_size,
        "stride": args.stride,
        "repeats": args.repeats,
        "n_estimators": args.n_estimators,
        "max_depth": args.max_depth,
        "learning_rate": args.learning_rate,
        "subsample": args.subsample,
        "colsample_bytree": args.colsample_bytree,
        "calibration": args.calibration,
    }
    candidates = [
        base,
        {
            **base,
            "chunk_size": max(64, args.chunk_size - 16),
            "min_chunk_size": max(32, args.min_chunk_size - 8),
            "stride": max(24, args.stride - 8),
            "repeats": max(args.repeats, 4),
            "learning_rate": 0.03,
            "n_estimators": max(args.n_estimators, 450),
            "max_depth": max(args.max_depth, 4),
        },
        {
            **base,
            "chunk_size": args.chunk_size + 16,
            "min_chunk_size": args.min_chunk_size + 8,
            "stride": args.stride + 8,
            "repeats": max(args.repeats, 4),
            "learning_rate": 0.04,
            "n_estimators": max(args.n_estimators, 500),
            "max_depth": args.max_depth + 1,
        },
        {
            **base,
            "chunk_size": max(60, args.chunk_size - 20),
            "min_chunk_size": max(30, args.min_chunk_size - 10),
            "stride": max(20, args.stride - 12),
            "repeats": max(args.repeats, 5),
            "learning_rate": 0.025,
            "n_estimators": max(args.n_estimators, 650),
            "max_depth": max(args.max_depth, 6),
            "subsample": min(1.0, max(args.subsample, 0.95)),
            "colsample_bytree": min(1.0, max(args.colsample_bytree, 0.95)),
        },
        {
            **base,
            "chunk_size": args.chunk_size,
            "min_chunk_size": args.min_chunk_size,
            "stride": max(20, args.stride // 2),
            "repeats": max(args.repeats, 6),
            "learning_rate": 0.035,
            "n_estimators": max(args.n_estimators, 700),
            "max_depth": max(args.max_depth, 6),
        },
        {
            **base,
            "chunk_size": args.chunk_size + 24,
            "min_chunk_size": args.min_chunk_size + 12,
            "stride": args.stride,
            "repeats": max(args.repeats, 4),
            "learning_rate": 0.02,
            "n_estimators": max(args.n_estimators, 800),
            "max_depth": max(args.max_depth, 7),
            "calibration": "sigmoid" if args.calibration == "auto" else args.calibration,
        },
    ]
    budget = max(1, int(args.search_budget))
    return candidates[:budget]


def _fit_single_candidate(
    *,
    args: argparse.Namespace,
    candidate_config: dict[str, float | int | str],
    human_hands: list[dict],
    bot_hands: list[dict],
    benchmark_train_rows: list[dict[str, float]],
    benchmark_validation_rows: list[dict[str, float]],
) -> tuple[object, list[str], dict[str, float], dict[str, float | int | str], str]:
    raw_rows = build_training_dataframe(
        human_hands=human_hands,
        bot_hands=bot_hands,
        chunk_size=int(candidate_config["chunk_size"]),
        min_chunk_size=int(candidate_config["min_chunk_size"]),
        stride=int(candidate_config["stride"]),
        repeats=int(candidate_config["repeats"]),
        seed=args.seed,
    )
    benchmark_multiplier = max(1, int(args.benchmark_oversample))
    oversampled_benchmark_rows = benchmark_train_rows * benchmark_multiplier
    rows = list(raw_rows) + list(oversampled_benchmark_rows)
    if not rows:
        raise RuntimeError("Training dataframe is empty. Verify your human/bot hand sources.")

    feature_names = sorted(key for key in rows[0].keys() if key != "label")
    if benchmark_validation_rows:
        X_train = [[float(row.get(name, 0.0)) for name in feature_names] for row in rows]
        y_train = [int(row["label"]) for row in rows]
        X_test = [[float(row.get(name, 0.0)) for name in feature_names] for row in benchmark_validation_rows]
        y_test = [int(row["label"]) for row in benchmark_validation_rows]
    else:
        X = [[float(row.get(name, 0.0)) for name in feature_names] for row in rows]
        y = [int(row["label"]) for row in rows]
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=args.test_size,
            random_state=args.seed,
            stratify=y,
        )

    if XGBClassifier is not None:
        booster_model = XGBClassifier(
            n_estimators=int(candidate_config["n_estimators"]),
            max_depth=int(candidate_config["max_depth"]),
            learning_rate=float(candidate_config["learning_rate"]),
            subsample=float(candidate_config["subsample"]),
            colsample_bytree=float(candidate_config["colsample_bytree"]),
            eval_metric="logloss",
            random_state=args.seed,
            n_jobs=1,
        )
        forest_model = ExtraTreesClassifier(
            n_estimators=max(200, int(candidate_config["n_estimators"])),
            max_depth=int(candidate_config["max_depth"]) + 2,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=args.seed,
            n_jobs=1,
        )
        base_model = VotingClassifier(
            estimators=[("xgb", booster_model), ("et", forest_model)],
            voting="soft",
            weights=[2, 1],
        )
        framework_name = "xgboost+extra-trees+sklearn-calibration"
    else:
        booster_model = HistGradientBoostingClassifier(
            learning_rate=float(candidate_config["learning_rate"]),
            max_depth=int(candidate_config["max_depth"]),
            max_iter=int(candidate_config["n_estimators"]),
            random_state=args.seed,
        )
        forest_model = ExtraTreesClassifier(
            n_estimators=max(300, int(candidate_config["n_estimators"])),
            max_depth=int(candidate_config["max_depth"]) + 2,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=args.seed,
            n_jobs=1,
        )
        base_model = VotingClassifier(
            estimators=[("hgb", booster_model), ("et", forest_model)],
            voting="soft",
            weights=[2, 1],
        )
        framework_name = "sklearn-hist-gradient-boosting+extra-trees+calibration"

    requested_calibration = choose_calibration(str(candidate_config["calibration"]), len(X_train))
    calibration_mode = str(candidate_config["calibration"])
    candidate_calibrations = (
        [requested_calibration] if calibration_mode != "auto" else ["sigmoid", "isotonic", None]
    )

    best_model = None
    best_metrics = None
    best_calibration = None
    best_postprocess = {"bias": 0.0, "temperature": 1.0}
    best_selection_score = float("-inf")

    for calibration_method in candidate_calibrations:
        candidate_model = base_model
        if calibration_method is not None:
            candidate_model = CalibratedClassifierCV(base_model, method=calibration_method, cv=3)
        candidate_model.fit(X_train, y_train)
        if hasattr(candidate_model, "predict_proba"):
            candidate_probs = candidate_model.predict_proba(X_test)[:, 1].tolist()
        else:
            candidate_probs = [float(value) for value in candidate_model.predict(X_test)]
        adjusted_probs, candidate_metrics, postprocess_config = choose_best_probability_postprocess(
            y_true=y_test,
            y_prob=candidate_probs,
            objective=args.selection_objective,
        )
        candidate_score = model_selection_score(candidate_metrics, args.selection_objective)
        print(
            "candidate",
            f"chunk_size={candidate_config['chunk_size']}",
            f"stride={candidate_config['stride']}",
            f"repeats={candidate_config['repeats']}",
            f"n_estimators={candidate_config['n_estimators']}",
            f"max_depth={candidate_config['max_depth']}",
            f"learning_rate={candidate_config['learning_rate']}",
            f"calibration={calibration_method or 'none'}",
            f"prob_bias={postprocess_config['bias']:.2f}",
            f"prob_temp={postprocess_config['temperature']:.2f}",
            f"selection_score={candidate_score:.6f}",
            format_metrics(candidate_metrics),
        )
        if candidate_score > best_selection_score:
            best_selection_score = candidate_score
            best_model = candidate_model
            best_metrics = candidate_metrics
            best_calibration = calibration_method
            best_postprocess = dict(postprocess_config)

    result_config = dict(candidate_config)
    result_config["calibration"] = best_calibration or "none"
    result_config["probability_bias"] = float(best_postprocess["bias"])
    result_config["probability_temperature"] = float(best_postprocess["temperature"])
    result_config["raw_rows"] = float(len(raw_rows))
    result_config["train_rows"] = float(len(X_train))
    result_config["test_rows"] = float(len(X_test))
    return best_model, feature_names, dict(best_metrics or {}), result_config, framework_name


def train_model(args: argparse.Namespace) -> tuple[object, list[str], dict[str, float]]:
    if joblib is None:
        raise RuntimeError(
            "Training dependencies are missing. Install scikit-learn and joblib first."
        )
    if (
        CalibratedClassifierCV is None
        or train_test_split is None
        or HistGradientBoostingClassifier is None
        or ExtraTreesClassifier is None
        or VotingClassifier is None
    ):
        raise RuntimeError("scikit-learn is required to train and calibrate the miner model.")

    human_path = resolve_existing_path(args.human_path, DEFAULT_HUMAN_PATHS)
    bot_path = resolve_existing_path(args.bot_path, DEFAULT_BOT_PATHS)
    human_hands = load_json_or_gz(human_path)
    bot_hands = load_json_or_gz(bot_path)
    benchmark_path = None
    try:
        benchmark_path = resolve_existing_path(args.benchmark_path, DEFAULT_BENCHMARK_PATHS)
    except FileNotFoundError:
        benchmark_path = None

    benchmark_train_rows: list[dict[str, float]] = []
    benchmark_validation_rows: list[dict[str, float]] = []
    if benchmark_path is not None:
        benchmark_train_rows = load_public_benchmark_rows(benchmark_path, split_filter="train")
        benchmark_validation_rows = load_public_benchmark_rows(benchmark_path, split_filter="validation")

    search_configs = build_search_configs(args) if args.search else [{
        "chunk_size": args.chunk_size,
        "min_chunk_size": args.min_chunk_size,
        "stride": args.stride,
        "repeats": args.repeats,
        "n_estimators": args.n_estimators,
        "max_depth": args.max_depth,
        "learning_rate": args.learning_rate,
        "subsample": args.subsample,
        "colsample_bytree": args.colsample_bytree,
        "calibration": args.calibration,
    }]

    best_model = None
    best_metrics = None
    best_feature_names = None
    best_config = None
    best_framework_name = ""
    best_selection_score = float("-inf")
    for search_index, candidate_config in enumerate(search_configs, start=1):
        print(f"search_candidate={search_index}/{len(search_configs)} config={candidate_config}")
        model, feature_names, metrics, result_config, framework_name = _fit_single_candidate(
            args=args,
            candidate_config=candidate_config,
            human_hands=human_hands,
            bot_hands=bot_hands,
            benchmark_train_rows=benchmark_train_rows,
            benchmark_validation_rows=benchmark_validation_rows,
        )
        candidate_score = model_selection_score(metrics, args.selection_objective)
        if candidate_score > best_selection_score:
            best_selection_score = candidate_score
            best_model = model
            best_metrics = metrics
            best_feature_names = feature_names
            best_config = result_config
            best_framework_name = framework_name

    model = best_model
    feature_names = list(best_feature_names or [])
    final_config = dict(best_config or {})

    artifact_meta = {
        "chunk_size": float(final_config.get("chunk_size", args.chunk_size)),
        "min_chunk_size": float(final_config.get("min_chunk_size", args.min_chunk_size)),
        "stride": float(final_config.get("stride", args.stride)),
        "repeats": float(final_config.get("repeats", args.repeats)),
        "n_estimators": float(final_config.get("n_estimators", args.n_estimators)),
        "max_depth": float(final_config.get("max_depth", args.max_depth)),
        "learning_rate": float(final_config.get("learning_rate", args.learning_rate)),
        "subsample": float(final_config.get("subsample", args.subsample)),
        "colsample_bytree": float(final_config.get("colsample_bytree", args.colsample_bytree)),
        "calibration": str(final_config.get("calibration", args.calibration)),
        "selection_objective": args.selection_objective,
        "framework": best_framework_name,
        "human_path": str(human_path),
        "bot_path": str(bot_path),
        "benchmark_path": str(benchmark_path) if benchmark_path is not None else "",
        "benchmark_train_rows": float(len(benchmark_train_rows)),
        "benchmark_validation_rows": float(len(benchmark_validation_rows)),
        "benchmark_oversample": float(args.benchmark_oversample),
        "raw_rows": float(final_config.get("raw_rows", 0.0)),
        "train_rows": float(final_config.get("train_rows", 0.0)),
        "test_rows": float(final_config.get("test_rows", 0.0)),
        "probability_bias": float(final_config.get("probability_bias", 0.0)),
        "probability_temperature": float(final_config.get("probability_temperature", 1.0)),
        "search_enabled": 1.0 if args.search else 0.0,
        "search_candidates": float(len(search_configs)),
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": model,
            "feature_names": feature_names,
            "metadata": artifact_meta,
        },
        output_path,
    )

    loaded = Poker44Model(output_path)
    latency = loaded.benchmark_latency([human_hands[: args.chunk_size], bot_hands[: args.chunk_size]])
    metrics = dict(best_metrics or {})
    metrics["latency_per_chunk_ms"] = latency["latency_per_chunk_ms"]
    return model, feature_names, metrics


def main() -> None:
    args = parse_args()
    _, feature_names, metrics = train_model(args)
    print(f"Saved model to {args.output}")
    print(f"Feature count: {len(feature_names)}")
    print(format_metrics(metrics))


if __name__ == "__main__":
    main()
