def print_eval_results(metrics, class_names):
    g_ood = metrics["global_with_ood"]
    global_sentence_ood = f"Global results: TP={g_ood['tp']}, FP={g_ood['fp']}, FN={g_ood['fn']}"

    g_no_ood = metrics["global_without_ood"]
    global_sentence_no_ood = f"Global results without ood: TP={g_no_ood['tp']}, FP={g_no_ood['fp']}, FN={g_no_ood['fn']}"

    per_class_parts = []
    for cls_id, stats in metrics["per_class"].items():
        cls_name = class_names[int(cls_id)] if int(cls_id) < len(class_names) else str(cls_id)
        precision = stats['tp'] / (stats['tp'] + stats.get('fp', 0)) if stats['tp'] + stats.get('fp', 0) > 0 else 0
        recall = stats['tp'] / (stats['tp'] + stats['fn']) if stats['tp'] + stats['fn'] > 0 else 0
        per_class_parts.append(
            f"{cls_name}: TP={stats['tp']}, FN={stats['fn']}, Precision={precision:.3f}, Recall={recall:.3f}")
    per_class_sentence = "Per-class results: " + "; ".join(per_class_parts)

    print(global_sentence_ood)
    print(global_sentence_no_ood)
    print(per_class_sentence)
