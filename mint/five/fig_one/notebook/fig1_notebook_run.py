# =============================================================================
# BLOCK 3: RUN (execute after Block 1 + Block 2)
# =============================================================================
# Notebook cell — runs last. All names from Block 1 (init) and Block 2 (peft)
# are already in scope. Do NOT add imports for symbols defined in other blocks.
#
# Runs softmax probe + XGBoost + MINT linear probes (SVM, LR) + equivalence search
# for all vitals outcomes, separately for each hospital in hospital_splits.
# Writes results to output/<hospital>/results.jsonl and output/results_summary.csv
#
# NOTE: The MINT linear probe specifications (SVM, LR) in this file must stay
# aligned with mint/five/fig_one/fig1.py. Both use model embeddings (not
# bag-of-words), StandardScaler, and identical hyperparameters. If you change
# the probe methodology here, update fig1.py and vice versa.
#
# If PEFT_BACKBONE=True, fine-tunes the model per hospital then evaluates the
# fine-tuned model via both softmax probe (Softmax_FT) and linear probes (SVM_FT, LR_FT).
#
# Testing:
#   Run the integration test after any changes to this file:
#     TEST_OUTCOMES=tachypnea python -m mint.five.fig_one.notebook.test_fig1_notebook
#
#   Environment variables:
#     TEST_OUTCOMES  - comma-separated outcomes (default: hypoxia,tachypnea)
#     TEST_N_ENCOUNTERS - number of encounters to use (default: 100)
# =============================================================================

all_results = []

logger.info(f"Running {len(OUTCOMES)} outcomes x {len(hospital_splits)} hospitals")
logger.info(f"Lookahead: {LOOKAHEAD_MIN} min | Max len: {MAX_LEN}")

for hospital_name, splits in hospital_splits.items():
    logger.info("#" * 70)
    logger.info(f"HOSPITAL: {hospital_name}")
    logger.info("#" * 70)

    hosp_dir = OUTPUT_DIR / hospital_name
    hosp_dir.mkdir(exist_ok=True, parents=True)

    results_file = hosp_dir / "results.jsonl"
    if results_file.exists():
        results_file.unlink()

    tokens_pure_train = splits["tokens_pure_train"]
    tokens_val = splits["tokens_val"]
    tokens_test = splits["tokens_test"]

    # ─── Backbone Fine-Tuning (per hospital, before outcome loop) ────────
    peft_model = None
    if PEFT_BACKBONE:
        # Ensure token_id column exists for PEFT dataset
        train_with_ids = tokens_pure_train.copy()
        val_with_ids = tokens_val.copy()
        if "token_id" not in train_with_ids.columns:
            train_with_ids["token_id"] = train_with_ids["name"].map(name_to_id)
        if "token_id" not in val_with_ids.columns:
            val_with_ids["token_id"] = val_with_ids["name"].map(name_to_id)

        peft_model = backbone_finetune(
            model, train_with_ids, val_with_ids,
            output_dir=hosp_dir,
            logger=logger,
            device=DEVICE,
            max_len=MAX_LEN,
        )

    # Pre-compute encounter-to-age for each split
    logger.info(f"  Computing encounter ages for {hospital_name}...")
    enc_age_train = get_encounter_ages(tokens_pure_train)
    enc_age_val = get_encounter_ages(tokens_val)
    enc_age_test = get_encounter_ages(tokens_test)

    for outcome in OUTCOMES:
        logger.info("=" * 60)
        logger.info(f"  [{hospital_name}] OUTCOME: {outcome} (lookahead={LOOKAHEAD_MIN}min)")
        logger.info("=" * 60)
        t0 = time.time()

        token_maps = TASK_TOKEN_MAPS[outcome]
        age_to_pos_id = token_maps["pos"]
        age_to_neg_id = token_maps["neg"]

        # ─── Build cases ─────────────────────────────────────────────────
        logger.info(f"    Building train cases...")
        train_cases = build_cases(tokens_pure_train, outcome, enc_age_train, use_first_only=True)

        logger.info(f"    Building val cases...")
        val_cases = build_cases(tokens_val, outcome, enc_age_val, use_first_only=True)

        logger.info(f"    Building test cases...")
        test_cases = build_cases(tokens_test, outcome, enc_age_test, use_first_only=True)

        n_test_pos = len(test_cases["positive"])
        n_test_neg = len(test_cases["negative"])
        if n_test_pos == 0:
            logger.warning(f"    No positive test cases for {outcome}, skipping.")
            continue

        # Canonical encounter ordering shared by every method's test predictions:
        # softmax_probe (shuffle=False loader), build_bag_of_words, and
        # build_triage_features all iterate positive + negative in this order.
        # Saved with every CSV so results can be re-bootstrapped per encounter.
        test_enc_keys = [c["encounter_key"] for c in test_cases["positive"] + test_cases["negative"]]

        # ─── Softmax Probe (Delphi model) ─────────────────────────────────
        logger.info(f"    Running softmax probe on test set ({n_test_pos + n_test_neg} cases)...")
        test_loader = build_loader(test_cases, age_to_pos_id, age_to_neg_id, MAX_LEN, BATCH_SIZE)

        sm_probs, sm_labels, sm_enc_keys, emb_test = softmax_probe(model, test_loader)
        sm_metrics = compute_metrics(sm_probs, sm_labels, "Softmax", outcome)

        # Save Softmax predictions
        pd.DataFrame({
            "probs": sm_probs,
            "labels": sm_labels,
            "encounter_key": sm_enc_keys,
        }).to_csv(hosp_dir / f"{outcome}_Softmax.csv", index=False)

        # ─── XGBoost ─────────────────────────────────────────────────────
        logger.info(f"    Building bag-of-words features...")
        vocab_size = len(vocab)
        X_train, y_train = build_bag_of_words(train_cases, MAX_LEN, vocab_size)
        X_val, y_val = build_bag_of_words(val_cases, MAX_LEN, vocab_size)
        X_test, y_test = build_bag_of_words(test_cases, MAX_LEN, vocab_size)

        n_pos = y_train.sum()
        n_neg = len(y_train) - n_pos
        scale_pos_weight = n_neg / n_pos if n_pos > 0 else 1.0
        incidence = float(y_train.mean())

        # Two-class guard for XGBoost
        xgb_valid, xgb_stats = check_two_classes(y_train, y_test, "XGBoost", outcome, hospital_name, logger)
        logger.info(f"    XGBoost dataset sizes: train_n={xgb_stats['train_n_total']}, train_pos={xgb_stats['train_n_pos']}, "
                    f"test_n={xgb_stats['test_n_total']}, test_pos={xgb_stats['test_n_pos']}")

        if xgb_valid:
            logger.info(f"    Training XGBoost (incidence={incidence:.4f}, scale_pos_weight={scale_pos_weight:.1f})...")

            clf = xgb.XGBClassifier(
                random_state=42,
                n_jobs=1,
                eval_metric="aucpr",
                objective="binary:logistic",
                scale_pos_weight=scale_pos_weight,
            )
            clf.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

            xgb_probs = clf.predict_proba(X_test)[:, 1]
            xgb_metrics = compute_metrics(xgb_probs, y_test, "XGBoost", outcome)

            # Save XGBoost predictions
            pd.DataFrame({
                "probs": xgb_probs,
                "labels": y_test,
                "encounter_key": test_enc_keys,
            }).to_csv(hosp_dir / f"{outcome}_XGBoost.csv", index=False)
        else:
            xgb_metrics = {f"XGBoost_{k}": None for k in ["auprc", "auroc", "improvement"]}

        # ─── Global XGBoost (pre-trained, test-only) ────────────────────
        if GLOBAL_XGBOOST and outcome in global_xgb_json:
            logger.info(f"    Loading Global XGBoost for {outcome}...")
            global_clf = xgb.XGBClassifier()
            global_clf.load_model(str(global_xgb_json[outcome]))
            global_xgb_probs = global_clf.predict_proba(X_test)[:, 1]
            global_xgb_metrics = compute_metrics(global_xgb_probs, y_test, "GlobalXGBoost", outcome)

            pd.DataFrame({
                "probs": global_xgb_probs,
                "labels": y_test,
                "encounter_key": test_enc_keys,
            }).to_csv(hosp_dir / f"{outcome}_GlobalXGBoost.csv", index=False)
        else:
            global_xgb_metrics = {}

        # ─── Triage baseline (logistic regression on first-30min triage tokens) ─
        if TRIAGE:
            logger.info(f"    Building triage-time features (first {TRIAGE_WINDOW_MIN}min)...")
            Xt_train, yt_train = build_triage_features(train_cases, vocab_size)
            Xt_test, yt_test = build_triage_features(test_cases, vocab_size)

            # Two-class guard for Triage LR
            triage_valid, triage_stats = check_two_classes(yt_train, yt_test, "Triage", outcome, hospital_name, logger)
            logger.info(f"    Triage dataset sizes: train_n={triage_stats['train_n_total']}, train_pos={triage_stats['train_n_pos']}, "
                        f"test_n={triage_stats['test_n_total']}, test_pos={triage_stats['test_n_pos']}")

            if triage_valid:
                triage_scaler = StandardScaler()
                Xt_train_scaled = triage_scaler.fit_transform(Xt_train)
                Xt_test_scaled = triage_scaler.transform(Xt_test)

                logger.info(f"    Training Triage logistic regression ({Xt_train.shape[1]} features)...")
                triage_clf = LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    random_state=SEED,
                    solver="lbfgs",
                )
                triage_clf.fit(Xt_train_scaled, yt_train)
                triage_probs = triage_clf.predict_proba(Xt_test_scaled)[:, 1]
                triage_metrics = compute_metrics(triage_probs, yt_test, "Triage", outcome)

                pd.DataFrame({
                    "probs": triage_probs,
                    "labels": yt_test,
                    "encounter_key": test_enc_keys,
                }).to_csv(hosp_dir / f"{outcome}_Triage.csv", index=False)
            else:
                triage_metrics = {f"Triage_{k}": None for k in ["auprc", "auroc", "improvement"]}
        else:
            triage_metrics = {}

        # ─── MINT Linear Probes (SVM + Logistic Regression) ──────────────
        logger.info(f"    Extracting MINT train embeddings for linear probes...")
        train_loader = build_loader(train_cases, age_to_pos_id, age_to_neg_id, MAX_LEN, BATCH_SIZE)
        _, y_emb_train, _, emb_train = softmax_probe(model, train_loader)
        y_emb_test = sm_labels

        # Two-class guard for MINT linear probes
        mint_probe_valid, mint_probe_stats = check_two_classes(y_emb_train, y_emb_test, "MINT_LinearProbes", outcome, hospital_name, logger)
        logger.info(f"    MINT LinearProbes dataset sizes: train_n={mint_probe_stats['train_n_total']}, train_pos={mint_probe_stats['train_n_pos']}, "
                    f"test_n={mint_probe_stats['test_n_total']}, test_pos={mint_probe_stats['test_n_pos']}")

        if not mint_probe_valid:
            svm_metrics = {f"MINT_SVM_{k}": None for k in ["auprc", "auroc", "improvement"]}
            lr_metrics = {f"MINT_LR_{k}": None for k in ["auprc", "auroc", "improvement"]}
        else:
            scaler = StandardScaler()
            emb_train_scaled = scaler.fit_transform(emb_train)
            emb_test_scaled = scaler.transform(emb_test)

            # Linear SVM probe (calibrated for probability outputs)
            logger.info(f"    Training Linear SVM probe...")
            svm_base = LinearSVC(
                max_iter=5000,
                class_weight="balanced",
                random_state=SEED,
                dual="auto",
            )
            n_min_class = min(mint_probe_stats["train_n_pos"], mint_probe_stats["train_n_neg"])
            svm_cv = min(3, n_min_class)
            if svm_cv >= 2:
                svm_cal = CalibratedClassifierCV(svm_base, cv=svm_cv, method="sigmoid")
                svm_cal.fit(emb_train_scaled, y_emb_train)
                svm_probs = svm_cal.predict_proba(emb_test_scaled)[:, 1]
            else:
                svm_base.fit(emb_train_scaled, y_emb_train)
                svm_probs = svm_base.decision_function(emb_test_scaled)
                svm_probs = 1.0 / (1.0 + np.exp(-svm_probs))
            svm_metrics = compute_metrics(svm_probs, y_emb_test, "MINT_SVM", outcome)

            pd.DataFrame({
                "probs": svm_probs,
                "labels": y_emb_test,
                "encounter_key": test_enc_keys,
            }).to_csv(hosp_dir / f"{outcome}_MINT_SVM.csv", index=False)

            # Logistic Regression probe
            logger.info(f"    Training Logistic Regression probe...")
            lr_clf = LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
                random_state=SEED,
                solver="lbfgs",
            )
            lr_clf.fit(emb_train_scaled, y_emb_train)
            lr_probs = lr_clf.predict_proba(emb_test_scaled)[:, 1]
            lr_metrics = compute_metrics(lr_probs, y_emb_test, "MINT_LR", outcome)

            pd.DataFrame({
                "probs": lr_probs,
                "labels": y_emb_test,
                "encounter_key": test_enc_keys,
            }).to_csv(hosp_dir / f"{outcome}_MINT_LR.csv", index=False)

        # ─── Fine-tuned model evaluation (Softmax_FT + linear probes) ─────
        if PEFT_BACKBONE and peft_model is not None:
            logger.info(f"    Running fine-tuned softmax probe on test set...")
            ft_sm_probs, ft_sm_labels, _, ft_emb_test = softmax_probe(peft_model, test_loader)
            ft_sm_metrics = compute_metrics(ft_sm_probs, ft_sm_labels, "Softmax_FT", outcome)

            pd.DataFrame({
                "probs": ft_sm_probs,
                "labels": ft_sm_labels,
                "encounter_key": test_enc_keys,
            }).to_csv(hosp_dir / f"{outcome}_Softmax_FT.csv", index=False)

            # Fine-tuned linear probes
            logger.info(f"    Extracting fine-tuned train embeddings...")
            _, ft_y_train, _, ft_emb_train = softmax_probe(peft_model, train_loader)

            # Two-class guard for FT linear probes
            ft_probe_valid, ft_probe_stats = check_two_classes(ft_y_train, ft_sm_labels, "MINT_FT_LinearProbes", outcome, hospital_name, logger)
            logger.info(f"    MINT FT LinearProbes dataset sizes: train_n={ft_probe_stats['train_n_total']}, train_pos={ft_probe_stats['train_n_pos']}, "
                        f"test_n={ft_probe_stats['test_n_total']}, test_pos={ft_probe_stats['test_n_pos']}")

            if not ft_probe_valid:
                ft_svm_metrics = {f"MINT_SVM_FT_{k}": None for k in ["auprc", "auroc", "improvement"]}
                ft_lr_metrics = {f"MINT_LR_FT_{k}": None for k in ["auprc", "auroc", "improvement"]}
            else:
                ft_scaler = StandardScaler()
                ft_emb_train_scaled = ft_scaler.fit_transform(ft_emb_train)
                ft_emb_test_scaled = ft_scaler.transform(ft_emb_test)

                # FT SVM
                logger.info(f"    Training FT Linear SVM probe...")
                ft_svm_base = LinearSVC(max_iter=5000, class_weight="balanced", random_state=SEED, dual="auto")
                ft_n_min = min(ft_probe_stats["train_n_pos"], ft_probe_stats["train_n_neg"])
                ft_svm_cv = min(3, ft_n_min)
                if ft_svm_cv >= 2:
                    ft_svm_cal = CalibratedClassifierCV(ft_svm_base, cv=ft_svm_cv, method="sigmoid")
                    ft_svm_cal.fit(ft_emb_train_scaled, ft_y_train)
                    ft_svm_probs = ft_svm_cal.predict_proba(ft_emb_test_scaled)[:, 1]
                else:
                    ft_svm_base.fit(ft_emb_train_scaled, ft_y_train)
                    ft_svm_probs = ft_svm_base.decision_function(ft_emb_test_scaled)
                    ft_svm_probs = 1.0 / (1.0 + np.exp(-ft_svm_probs))
                ft_svm_metrics = compute_metrics(ft_svm_probs, ft_sm_labels, "MINT_SVM_FT", outcome)

                pd.DataFrame({"probs": ft_svm_probs, "labels": ft_sm_labels, "encounter_key": test_enc_keys}).to_csv(
                    hosp_dir / f"{outcome}_MINT_SVM_FT.csv", index=False)

                # FT LR
                logger.info(f"    Training FT Logistic Regression probe...")
                ft_lr_clf = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=SEED, solver="lbfgs")
                ft_lr_clf.fit(ft_emb_train_scaled, ft_y_train)
                ft_lr_probs = ft_lr_clf.predict_proba(ft_emb_test_scaled)[:, 1]
                ft_lr_metrics = compute_metrics(ft_lr_probs, ft_sm_labels, "MINT_LR_FT", outcome)

                pd.DataFrame({"probs": ft_lr_probs, "labels": ft_sm_labels, "encounter_key": test_enc_keys}).to_csv(
                    hosp_dir / f"{outcome}_MINT_LR_FT.csv", index=False)
        else:
            ft_sm_metrics = {}
            ft_svm_metrics = {}
            ft_lr_metrics = {}
            ft_probe_stats = None

        # ─── Classification Head Fine-Tuning (backbone + head, per outcome) ─
        if PEFT_CLASSIFICATION:
            clf_head_model = classification_head_finetune(
                model, train_cases, val_cases, MAX_LEN,
                output_dir=hosp_dir, outcome=outcome,
                logger=logger, device=DEVICE,
            )
            if clf_head_model is not None:
                clf_head_probs, clf_head_labels = classification_head_predict(
                    clf_head_model, test_loader, device=DEVICE)
                clf_head_metrics = compute_metrics(clf_head_probs, clf_head_labels, "ClassHead", outcome)
                pd.DataFrame({
                    "probs": clf_head_probs,
                    "labels": clf_head_labels,
                    "encounter_key": test_enc_keys,
                }).to_csv(hosp_dir / f"{outcome}_ClassHead.csv", index=False)
                del clf_head_model
                gc.collect()
            else:
                clf_head_metrics = {f"ClassHead_{k}": None for k in ["auprc", "auroc", "improvement"]}
        else:
            clf_head_metrics = {}

        # ─── Equivalence search ──────────────────────────────────────────
        logger.info(f"    Running equivalence search ({EQUIVALENCE_SEEDS} seeds)...")
        equiv_results = []
        for seed_i in range(EQUIVALENCE_SEEDS):
            result = equivalence_search(
                X_train, y_train, X_val, y_val, X_test, y_test,
                mint_auroc=sm_metrics["Softmax_auroc"],
                task_name=outcome,
                seed=seed_i,
            )
            equiv_results.append(result)
            logger.info(f"      Seed {seed_i}: {result}")

        # ─── Combine and save ────────────────────────────────────────────
        elapsed = time.time() - t0

        # Dataset statistics for interpretability (n_total and n_pos for train/test by method)
        dataset_stats = {
            # XGBoost (bag-of-words features)
            "xgb_train_n_total": xgb_stats["train_n_total"],
            "xgb_train_n_pos": xgb_stats["train_n_pos"],
            "xgb_test_n_total": xgb_stats["test_n_total"],
            "xgb_test_n_pos": xgb_stats["test_n_pos"],
            # MINT Linear Probes (embeddings)
            "mint_probe_train_n_total": mint_probe_stats["train_n_total"],
            "mint_probe_train_n_pos": mint_probe_stats["train_n_pos"],
            "mint_probe_test_n_total": mint_probe_stats["test_n_total"],
            "mint_probe_test_n_pos": mint_probe_stats["test_n_pos"],
        }
        # Triage stats (if enabled)
        if TRIAGE:
            dataset_stats.update({
                "triage_train_n_total": triage_stats["train_n_total"],
                "triage_train_n_pos": triage_stats["train_n_pos"],
                "triage_test_n_total": triage_stats["test_n_total"],
                "triage_test_n_pos": triage_stats["test_n_pos"],
            })
        # FT probe stats (if PEFT enabled and stats available)
        if PEFT_BACKBONE and peft_model is not None and ft_probe_stats is not None:
            dataset_stats.update({
                "ft_probe_train_n_total": ft_probe_stats["train_n_total"],
                "ft_probe_train_n_pos": ft_probe_stats["train_n_pos"],
                "ft_probe_test_n_total": ft_probe_stats["test_n_total"],
                "ft_probe_test_n_pos": ft_probe_stats["test_n_pos"],
            })

        combined = {
            "hospital": hospital_name,
            "task": outcome,
            "lookahead_min": LOOKAHEAD_MIN,
            "incidence": incidence,
            "n_train": int(len(y_train)),
            "n_test": int(len(y_test)),
            "n_pos_train": int(y_train.sum()),
            "n_pos_test": int(y_test.sum()),
            **dataset_stats,
            **sm_metrics,
            **xgb_metrics,
            **global_xgb_metrics,
            **triage_metrics,
            **svm_metrics,
            **lr_metrics,
            **ft_sm_metrics,
            **ft_svm_metrics,
            **ft_lr_metrics,
            **clf_head_metrics,
            "xgb_total_samples": int(len(X_train)),
            "equivalence_samples_multiple": equiv_results,
            "elapsed_sec": round(elapsed, 1),
        }

        # Serialize — convert tuples to lists for JSON
        combined_json = {}
        for k, v in combined.items():
            if isinstance(v, tuple):
                combined_json[k] = list(v)
            else:
                combined_json[k] = v

        with open(results_file, "a") as f:
            f.write(json.dumps(combined_json) + "\n")

        all_results.append(combined)
        logger.info(f"    DONE [{hospital_name}] {outcome} in {elapsed:.1f}s")

# ─── Final summary ───────────────────────────────────────────────────────────
logger.info("=" * 60)
logger.info("ALL HOSPITALS / OUTCOMES COMPLETE")
logger.info("=" * 60)

summary_df = pd.DataFrame(all_results)
summary_df.to_csv(OUTPUT_DIR / "results_summary.csv", index=False)
logger.info(f"Saved summary to {OUTPUT_DIR / 'results_summary.csv'}")

# Print summary table
has_ft = any("Softmax_FT_auprc" in r for r in all_results)
if has_ft:
    print("\n" + "=" * 160)
    print(f"{'Hospital':<15} {'Task':<15} {'Softmax':<8} {'SM_FT':<8} {'XGB':<8} {'Triage':<8} {'SVM':<8} {'SVM_FT':<8} {'LR':<8} {'LR_FT':<8} {'Incidence':<10} {'Equiv'}")
    print("-" * 160)
    for r in all_results:
        def _fmt(key):
            v = r.get(key)
            return f"{v:<8.4f}" if v is not None else f"{'N/A':<8}"
        equiv_str = str(r.get("equivalence_samples_multiple", []))[:20]
        print(f"{r['hospital']:<15} {r['task']:<15} {_fmt('Softmax_auprc')} {_fmt('Softmax_FT_auprc')} "
              f"{_fmt('XGBoost_auprc')} {_fmt('Triage_auprc')} {_fmt('MINT_SVM_auprc')} {_fmt('MINT_SVM_FT_auprc')} "
              f"{_fmt('MINT_LR_auprc')} {_fmt('MINT_LR_FT_auprc')} {r['incidence']:<10.4f} {equiv_str}")
    print("=" * 160)
else:
    print("\n" + "=" * 120)
    print(f"{'Hospital':<15} {'Task':<15} {'SM AUPRC':<12} {'XGB AUPRC':<12} {'Triage AUPRC':<14} {'SVM AUPRC':<12} {'LR AUPRC':<12} {'Incidence':<12} {'Equiv Samples'}")
    print("-" * 120)
    for r in all_results:
        equiv_str = str(r.get("equivalence_samples_multiple", []))[:20]
        sm_str = f"{r['Softmax_auprc']:<12.4f}" if r.get("Softmax_auprc") is not None else f"{'N/A':<12}"
        xgb_str = f"{r['XGBoost_auprc']:<12.4f}" if r.get("XGBoost_auprc") is not None else f"{'N/A':<12}"
        svm_str = f"{r['MINT_SVM_auprc']:<12.4f}" if r.get("MINT_SVM_auprc") is not None else f"{'N/A':<12}"
        lr_str = f"{r['MINT_LR_auprc']:<12.4f}" if r.get("MINT_LR_auprc") is not None else f"{'N/A':<12}"
        triage_str = f"{r['Triage_auprc']:<14.4f}" if r.get("Triage_auprc") is not None else f"{'N/A':<14}"
        print(f"{r['hospital']:<15} {r['task']:<15} {sm_str} {xgb_str} {triage_str} {svm_str} {lr_str} {r['incidence']:<12.4f} {equiv_str}")
    print("=" * 120)

# ─── Alignment audit ──────────────────────────────────────────────────────────
# Re-read every per-case prediction CSV and confirm that, for each hospital ×
# outcome, all method files agree on both the encounter_key column AND the
# labels column, row for row. Every method's test predictions are produced by
# iterating test_cases positive+negative in the same order (shuffle=False
# loaders, build_bag_of_words, build_triage_features), so the encounter_key and
# labels columns must be byte-for-byte identical across methods. If they are,
# a downstream join/re-bootstrap on encounter_key will line probs up with the
# correct labels. This asserts hard so a misalignment can never pass silently.
logger.info("=" * 60)
logger.info("ALIGNMENT AUDIT: verifying encounter_key + labels across method CSVs")
logger.info("=" * 60)

# Prediction CSVs are named "{outcome}_{Method}.csv"; the summary lives at the
# output root, so it is never globbed here.
audited = 0
for hospital_name in hospital_splits:
    hosp_dir = OUTPUT_DIR / hospital_name
    if not hosp_dir.exists():
        continue
    for outcome in OUTCOMES:
        method_files = sorted(hosp_dir.glob(f"{outcome}_*.csv"))
        if len(method_files) < 2:
            continue  # nothing to cross-check for this outcome

        ref_path = method_files[0]
        ref = pd.read_csv(ref_path)
        assert "encounter_key" in ref.columns, f"{ref_path} missing encounter_key column"
        assert "labels" in ref.columns, f"{ref_path} missing labels column"
        ref_keys = ref["encounter_key"].astype(str).tolist()
        ref_labels = ref["labels"].tolist()

        for path in method_files[1:]:
            df = pd.read_csv(path)
            assert "encounter_key" in df.columns, f"{path} missing encounter_key column"
            assert "labels" in df.columns, f"{path} missing labels column"
            assert len(df) == len(ref), (
                f"Row count mismatch for {outcome} in {hospital_name}: "
                f"{path.name}={len(df)} vs {ref_path.name}={len(ref)}"
            )
            assert df["encounter_key"].astype(str).tolist() == ref_keys, (
                f"encounter_key misalignment: {path.name} != {ref_path.name} "
                f"({hospital_name}/{outcome})"
            )
            assert df["labels"].tolist() == ref_labels, (
                f"labels misalignment: {path.name} != {ref_path.name} "
                f"({hospital_name}/{outcome})"
            )
        audited += 1
        logger.info(f"  OK [{hospital_name}] {outcome}: {len(method_files)} method files aligned "
                    f"({len(ref)} rows)")

logger.info(f"ALIGNMENT AUDIT PASSED: {audited} hospital×outcome group(s) verified")
