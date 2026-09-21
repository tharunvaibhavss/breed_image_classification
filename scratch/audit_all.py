"""
Independent Comprehensive Reproducibility and Integrity Audit Script.
Audit of Final Model (EfficientNet-B0 V3: Real + Synthetic Augmentation).
DO NOT RETRAIN. DO NOT MODIFY DATASET. DO NOT MODIFY METRICS.
"""

import os
import sys
import json
import hashlib
import time
import platform
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms, models
import onnx
import onnxruntime as ort
import imagehash
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    classification_report, confusion_matrix
)

WORKSPACE = Path("c:/Users/HP/Desktop/MCA Project/MCA Project AI Breed")
sys.path.insert(0, str(WORKSPACE))
MODELS_V3 = WORKSPACE / "models" / "synthetic_82_breeds_v3"
REPORTS_V3 = WORKSPACE / "reports" / "synthetic_v3"
REPORTS_V2 = WORKSPACE / "reports" / "dataset_expansion_v2"
SPLIT_MANIFEST_REAL = REPORTS_V2 / "split_manifest_v2.csv"
AUG_MANIFEST_V3 = REPORTS_V3 / "augmented_split_manifest_v3.csv"
CLASS_MAP_PATH = MODELS_V3 / "class_mapping_v3.json"
MODEL_PATH = MODELS_V3 / "best_model_v3.pth"
ONNX_MODEL_PATH = MODELS_V3 / "efficientnet_b0_v3.onnx"
SYNTH_METADATA_PATH = REPORTS_V3 / "synthetic_metadata_v3.csv"


def run_audit():
    print("=" * 70)
    print("RUNNING INDEPENDENT FINAL MODEL INTEGRITY & REPRODUCIBILITY AUDIT")
    print("=" * 70)
    
    results = {}

    # -------------------------------------------------------------
    # 1. DATASET AUDIT
    # -------------------------------------------------------------
    print("\n--- 1. DATASET AUDIT ---")
    aug_manifest = pd.read_csv(AUG_MANIFEST_V3)
    real_manifest = pd.read_csv(SPLIT_MANIFEST_REAL)
    
    # Counts in augmented manifest
    train_df = aug_manifest[aug_manifest["split"] == "train"].reset_index(drop=True)
    val_df = aug_manifest[aug_manifest["split"] == "validation"].reset_index(drop=True)
    test_df = aug_manifest[aug_manifest["split"] == "test"].reset_index(drop=True)
    
    real_train = train_df[train_df["source_type"] == "REAL"]
    synth_train = train_df[train_df["source_type"] == "SYNTHETIC"]
    real_val = val_df[val_df["source_type"] == "REAL"]
    synth_val = val_df[val_df["source_type"] == "SYNTHETIC"]
    real_test = test_df[test_df["source_type"] == "REAL"]
    synth_test = test_df[test_df["source_type"] == "SYNTHETIC"]
    
    total_classes = aug_manifest["breed_id"].nunique()
    cattle_classes = aug_manifest[aug_manifest["species"].str.lower() == "cattle"]["breed_id"].nunique()
    buffalo_classes = aug_manifest[aug_manifest["species"].str.lower() == "buffalo"]["breed_id"].nunique()
    
    print(f"Total training images: {len(train_df)}")
    print(f"  Real training images: {len(real_train)}")
    print(f"  Synthetic training images: {len(synth_train)}")
    print(f"Validation images: {len(val_df)} (Real: {len(real_val)}, Synthetic: {len(synth_val)})")
    print(f"Testing images: {len(test_df)} (Real: {len(real_test)}, Synthetic: {len(synth_test)})")
    print(f"Total classes: {total_classes} (Cattle: {cattle_classes}, Buffalo: {buffalo_classes})")
    
    val_is_100_real = (len(synth_val) == 0 and len(real_val) == len(val_df))
    test_is_100_real = (len(synth_test) == 0 and len(real_test) == len(test_df))
    print(f"Validation is 100% real: {val_is_100_real}")
    print(f"Test is 100% real: {test_is_100_real}")

    results["dataset"] = {
        "real_train_count": len(real_train),
        "synth_train_count": len(synth_train),
        "total_train_count": len(train_df),
        "validation_count": len(val_df),
        "validation_real_count": len(real_val),
        "validation_synth_count": len(synth_val),
        "test_count": len(test_df),
        "test_real_count": len(real_test),
        "test_synth_count": len(synth_test),
        "total_classes": total_classes,
        "cattle_classes": cattle_classes,
        "buffalo_classes": buffalo_classes,
        "val_is_100_real": val_is_100_real,
        "test_is_100_real": test_is_100_real,
    }

    # -------------------------------------------------------------
    # 2. TEST SET INDEPENDENCE (Exact & Near Duplicates)
    # -------------------------------------------------------------
    print("\n--- 2. TEST SET INDEPENDENCE ---")
    # Compute sha256 and dhash for test, train, val
    def get_hashes(df):
        shas = []
        dhashes = []
        paths = []
        for _, r in df.iterrows():
            p = WORKSPACE / r["relative_path"]
            paths.append(str(p))
            with open(p, "rb") as f:
                shas.append(hashlib.sha256(f.read()).hexdigest())
            with Image.open(p) as im:
                dhashes.append(imagehash.dhash(im))
        return shas, dhashes, paths

    test_shas, test_dhashes, test_paths = get_hashes(test_df)
    real_train_shas, real_train_dhashes, _ = get_hashes(real_train)
    val_shas, val_dhashes, _ = get_hashes(val_df)
    synth_train_shas, synth_train_dhashes, _ = get_hashes(synth_train)

    # 1. Real Test <-> Real Train
    test_train_exact = len(set(test_shas).intersection(set(real_train_shas)))
    test_train_near = 0
    for th in test_dhashes:
        for trh in real_train_dhashes:
            if th - trh <= 3:
                test_train_near += 1
                break

    # 2. Real Test <-> Real Validation
    test_val_exact = len(set(test_shas).intersection(set(val_shas)))
    test_val_near = 0
    for th in test_dhashes:
        for vh in val_dhashes:
            if th - vh <= 3:
                test_val_near += 1
                break

    # 3. Synthetic Train <-> Real Test
    synth_test_exact = len(set(synth_train_shas).intersection(set(test_shas)))
    synth_test_near = 0
    for th in test_dhashes:
        for sh in synth_train_dhashes:
            if th - sh <= 3:
                synth_test_near += 1
                break

    print(f"Real Test <-> Real Train: exact={test_train_exact}, near (dhash<=3)={test_train_near}")
    print(f"Real Test <-> Real Val:   exact={test_val_exact}, near (dhash<=3)={test_val_near}")
    print(f"Synth Train <-> Real Test: exact={synth_test_exact}, near (dhash<=3)={synth_test_near}")

    results["independence"] = {
        "test_train_exact": test_train_exact,
        "test_train_near": test_train_near,
        "test_val_exact": test_val_exact,
        "test_val_near": test_val_near,
        "synth_test_exact": synth_test_exact,
        "synth_test_near": synth_test_near,
    }

    # -------------------------------------------------------------
    # 3. SYNTHETIC REFERENCE LEAKAGE
    # -------------------------------------------------------------
    print("\n--- 3. SYNTHETIC REFERENCE LEAKAGE ---")
    # Check how synthetic images were created and if any test image was used as base_path
    # In generate_synthetic_dataset_v3.py, let's trace which breeds used test images as base_path
    synth_meta = pd.read_csv(SYNTH_METADATA_PATH) if SYNTH_METADATA_PATH.exists() else None
    
    # Check which breeds had 0 real train images in real_manifest
    breeds_with_no_real_train = []
    real_train_manifest = real_manifest[real_manifest["split"] == "train"]
    real_test_manifest = real_manifest[real_manifest["split"] == "test"]
    
    all_breeds = real_manifest["breed_id"].unique()
    for b in all_breeds:
        tr_count = len(real_train_manifest[real_train_manifest["breed_id"] == b])
        if tr_count == 0:
            te_count = len(real_test_manifest[real_test_manifest["breed_id"] == b])
            breeds_with_no_real_train.append((b, te_count))

    print(f"Breeds with 0 real training images: {breeds_with_no_real_train}")
    
    # Check how many synthetic images were created for these zero-train breeds
    synth_from_zero_train_breeds = 0
    zero_train_breed_ids = [b[0] for b in breeds_with_no_real_train]
    if synth_meta is not None:
        synth_from_zero_train_breeds = len(synth_meta[synth_meta["breed_id"].isin(zero_train_breed_ids)])
    print(f"Synthetic images generated for zero-train breeds: {synth_from_zero_train_breeds}")
    
    # In generate_synthetic_dataset_v3.py:
    # If available_exemplars for breed_id was empty from real_train, it fell back to:
    # [ROOT / r["relative_path"] for _, r in manifest_df[manifest_df["breed_id"] == b_id].iterrows()]
    # For cow_bargur, manifest_df only has 1 row, which is in test!
    # So all synthetic images for cow_bargur used that 1 test image as base exemplar!
    print(f"Potential synthetic-to-test reference lineage leakage: {synth_from_zero_train_breeds} / {len(synth_train)}")
    results["synthetic_reference_leakage"] = {
        "breeds_with_zero_real_train": breeds_with_no_real_train,
        "synth_images_from_zero_train_breeds": synth_from_zero_train_breeds,
        "total_synthetic_images": len(synth_train),
        "potential_leakage_count": synth_from_zero_train_breeds,
    }

    # -------------------------------------------------------------
    # 4. CLASS MAPPING AUDIT
    # -------------------------------------------------------------
    print("\n--- 4. CLASS MAPPING AUDIT ---")
    with open(CLASS_MAP_PATH, "r", encoding="utf-8") as f:
        class_map = json.load(f)
    idx_to_class = class_map["idx_to_class"]
    idx_to_breed_name = class_map["idx_to_breed_name"]
    class_to_idx = {v: int(k) for k, v in idx_to_class.items()}
    num_classes = len(idx_to_class)
    print(f"Class mapping output classes: {num_classes}")
    
    # Verify correspondence to model architecture
    device = torch.device("cpu")
    model = models.efficientnet_b0(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    
    ckpt = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    
    model_output_features = model.classifier[1].out_features
    print(f"PyTorch model classifier out_features: {model_output_features}")
    assert model_output_features == num_classes == 82, "Class count mismatch!"

    # -------------------------------------------------------------
    # 5. METRIC REPRODUCTION (Strict on 123 Real Test Images)
    # -------------------------------------------------------------
    print("\n--- 5. METRIC REPRODUCTION ---")
    eval_tf = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    test_predictions = []
    y_true = []
    y_pred = []
    top3_correct = 0
    all_probs = []

    with torch.no_grad():
        for _, row in test_df.iterrows():
            img_path = WORKSPACE / row["relative_path"]
            true_breed_id = row["breed_id"]
            true_idx = class_to_idx[true_breed_id]
            species = row["species"]

            with Image.open(img_path) as im:
                img_tensor = eval_tf(im.convert("RGB")).unsqueeze(0).to(device)

            logits = model(img_tensor)
            probs = torch.softmax(logits, dim=1).squeeze(0)

            top3_prob, top3_indices = torch.topk(probs, 3)
            pred_idx = top3_indices[0].item()
            pred_breed_id = idx_to_class[str(pred_idx)]
            top3_idx_list = [t.item() for t in top3_indices]

            is_top1 = (pred_idx == true_idx)
            is_top3 = (true_idx in top3_idx_list)
            if is_top3:
                top3_correct += 1

            y_true.append(true_idx)
            y_pred.append(pred_idx)
            all_probs.append(probs.numpy())
            
            test_predictions.append({
                "breed_id": true_breed_id,
                "species": species,
                "pred_breed_id": pred_breed_id,
                "is_top1": is_top1,
                "is_top3": is_top3
            })

    total_test = len(test_df)
    acc = accuracy_score(y_true, y_pred)
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(y_true, y_pred, average='macro', zero_division=0)
    weighted_p, weighted_r, weighted_f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted', zero_division=0)
    top3_acc = top3_correct / total_test
    unique_pred = len(set(y_pred))

    print(f"Independently Calculated Metrics (N={total_test}):")
    print(f"  Accuracy (Top-1):    {acc*100:.2f}% (Expected: 36.59%)")
    print(f"  Top-3 Accuracy:      {top3_acc*100:.2f}% (Expected: 50.41%)")
    print(f"  Macro Precision:     {macro_p*100:.2f}% (Expected: 20.34%)")
    print(f"  Macro Recall:        {macro_r*100:.2f}% (Expected: 24.37%)")
    print(f"  Macro F1-Score:      {macro_f1*100:.2f}% (Expected: 21.41%)")
    print(f"  Weighted Precision:  {weighted_p*100:.2f}% (Expected: 28.54%)")
    print(f"  Weighted Recall:     {weighted_r*100:.2f}% (Expected: 36.59%)")
    print(f"  Weighted F1-Score:   {weighted_f1*100:.2f}% (Expected: 30.88%)")
    print(f"  Unique Predicted:    {unique_pred} / 82 (Expected: 40/82)")

    results["metrics"] = {
        "acc": acc,
        "top3_acc": top3_acc,
        "macro_p": macro_p,
        "macro_r": macro_r,
        "macro_f1": macro_f1,
        "weighted_p": weighted_p,
        "weighted_r": weighted_r,
        "weighted_f1": weighted_f1,
        "unique_pred": unique_pred,
    }

    # -------------------------------------------------------------
    # 6. PER-CLASS REPORT
    # -------------------------------------------------------------
    print("\n--- 6. PER-CLASS REPORT ---")
    labels_all = list(range(num_classes))
    target_names = [idx_to_class[str(i)] for i in labels_all]
    
    p_per, r_per, f1_per, s_per = precision_recall_fscore_support(
        y_true, y_pred, labels=labels_all, zero_division=0
    )
    
    per_class_df = pd.DataFrame({
        "class_index": labels_all,
        "breed_id": target_names,
        "breed_name": [idx_to_breed_name[str(i)] for i in labels_all],
        "precision": p_per,
        "recall": r_per,
        "f1": f1_per,
        "support": s_per
    })
    
    small_support = per_class_df[per_class_df["support"] <= 1]
    zero_support = per_class_df[per_class_df["support"] == 0]
    print(f"Classes with support == 0: {len(zero_support)}")
    print(f"Classes with support == 1: {len(small_support) - len(zero_support)}")
    print(f"Classes with support >= 2: {len(per_class_df[per_class_df['support'] >= 2])}")

    # Check match with saved classification report
    saved_cls_rep = pd.read_csv(REPORTS_V3 / "classification_report_v3.csv")
    print(f"Saved classification report classes: {len(saved_cls_rep)}")

    # -------------------------------------------------------------
    # 7. CONFUSION MATRIX VERIFICATION
    # -------------------------------------------------------------
    print("\n--- 7. CONFUSION MATRIX ---")
    cm = confusion_matrix(y_true, y_pred, labels=labels_all)
    saved_cm = pd.read_csv(REPORTS_V3 / "confusion_matrix_v3.csv", index_col=0).values
    cm_diff = np.max(np.abs(cm - saved_cm))
    print(f"Max difference with saved confusion matrix: {cm_diff}")
    assert cm_diff == 0, "Confusion matrix does not match saved file!"

    # -------------------------------------------------------------
    # 8. CATTLE / BUFFALO METRICS
    # -------------------------------------------------------------
    print("\n--- 8. CATTLE / BUFFALO METRICS ---")
    pred_df = pd.DataFrame(test_predictions)
    
    # Cattle
    c_df = pred_df[pred_df["species"].str.lower() == "cattle"]
    c_acc = c_df["is_top1"].mean()
    c_top3 = c_df["is_top3"].mean()
    
    # Buffalo
    b_df = pred_df[pred_df["species"].str.lower() == "buffalo"]
    b_acc = b_df["is_top1"].mean()
    b_top3 = b_df["is_top3"].mean()

    # Per species macro metrics
    c_indices = [class_to_idx[r["breed_id"]] for _, r in c_df.iterrows()]
    c_preds = [class_to_idx[r["pred_breed_id"]] for _, r in c_df.iterrows()]
    c_p, c_r, c_f1, _ = precision_recall_fscore_support(c_indices, c_preds, average='macro', zero_division=0)

    b_indices = [class_to_idx[r["breed_id"]] for _, r in b_df.iterrows()]
    b_preds = [class_to_idx[r["pred_breed_id"]] for _, r in b_df.iterrows()]
    b_p, b_r, b_f1, _ = precision_recall_fscore_support(b_indices, b_preds, average='macro', zero_division=0)

    print(f"Cattle (N={len(c_df)}):  Acc={c_acc*100:.2f}% (Expected: 35.56%), Top-3={c_top3*100:.2f}%, Macro F1={c_f1*100:.2f}%")
    print(f"Buffalo (N={len(b_df)}): Acc={b_acc*100:.2f}% (Expected: 39.39%), Top-3={b_top3*100:.2f}%, Macro F1={b_f1*100:.2f}%")

    results["species"] = {
        "cattle_n": len(c_df),
        "cattle_acc": c_acc,
        "cattle_top3": c_top3,
        "cattle_p": c_p,
        "cattle_r": c_r,
        "cattle_f1": c_f1,
        "buffalo_n": len(b_df),
        "buffalo_acc": b_acc,
        "buffalo_top3": b_top3,
        "buffalo_p": b_p,
        "buffalo_r": b_r,
        "buffalo_f1": b_f1,
    }

    # -------------------------------------------------------------
    # 9. REAL-ONLY TEST VERIFICATION
    # -------------------------------------------------------------
    print("\n--- 9. REAL-ONLY TEST VERIFICATION ---")
    val_synth = val_df[val_df["source_type"] != "REAL"]
    test_synth = test_df[test_df["source_type"] != "REAL"]
    print(f"Synthetic images in validation split: {len(val_synth)}")
    print(f"Synthetic images in test split: {len(test_synth)}")

    # -------------------------------------------------------------
    # 10. PYTORCH / ONNX CONSISTENCY
    # -------------------------------------------------------------
    print("\n--- 10. PYTORCH / ONNX CONSISTENCY ---")
    ort_session = ort.InferenceSession(str(ONNX_MODEL_PATH), providers=["CPUExecutionProvider"])
    ort_input_name = ort_session.get_inputs()[0].name
    
    onnx_preds = []
    max_prob_diff = 0.0

    for idx, (_, row) in enumerate(test_df.iterrows()):
        img_path = WORKSPACE / row["relative_path"]
        with Image.open(img_path) as im:
            t = eval_tf(im.convert("RGB")).unsqueeze(0).numpy()
        
        ort_out = ort_session.run(None, {ort_input_name: t})[0]
        ort_pred_idx = int(np.argmax(ort_out, axis=1)[0])
        onnx_preds.append(ort_pred_idx)
        
        pt_prob = all_probs[idx]
        ort_prob = np.exp(ort_out) / np.sum(np.exp(ort_out))
        diff = np.max(np.abs(pt_prob - ort_prob))
        if diff > max_prob_diff:
            max_prob_diff = diff

    identical = sum(1 for pt, ox in zip(y_pred, onnx_preds) if pt == ox)
    differing = len(y_pred) - identical
    agreement_pct = (identical / len(y_pred)) * 100.0

    print(f"Total test samples: {len(y_pred)}")
    print(f"Identical predictions: {identical}")
    print(f"Differing predictions: {differing}")
    print(f"Agreement percentage: {agreement_pct:.2f}%")
    print(f"Max softmax probability difference: {max_prob_diff:.6e}")

    results["onnx_parity"] = {
        "identical": identical,
        "differing": differing,
        "agreement_pct": agreement_pct,
        "max_prob_diff": float(max_prob_diff)
    }

    # -------------------------------------------------------------
    # 11. APPLICATION MODEL VERIFICATION
    # -------------------------------------------------------------
    print("\n--- 11. APPLICATION MODEL VERIFICATION ---")
    import app.api.predict as predict_module
    import app.core.config as config_module
    
    # Check default model in predict.py
    import inspect
    sig = inspect.signature(predict_module.get_pipeline)
    default_model_version = sig.parameters["model_version"].default
    print(f"app.api.predict get_pipeline default model_version: '{default_model_version}'")
    
    # Check what pipelines can be created
    pipe_default = predict_module.get_inference_pipeline()
    loaded_classes = len(pipe_default.breed_predictor.class_mapping)
    print(f"Default pipeline loaded class count: {loaded_classes}")

    # Check if V3 is configured in predict.py
    with open(WORKSPACE / "app" / "api" / "predict.py", "r", encoding="utf-8") as pf:
        predict_code = pf.read()
    v3_in_code = ("synthetic_82_breeds_v3" in predict_code or "best_model_v3.pth" in predict_code)
    print(f"V3 model configured in app/api/predict.py: {v3_in_code}")
    results["application"] = {
        "default_model_version": default_model_version,
        "loaded_classes": loaded_classes,
        "v3_configured_in_api": v3_in_code,
        "preprocessor_target_size": [224, 224],
        "normalization_mean": [0.485, 0.456, 0.406],
        "normalization_std": [0.229, 0.224, 0.225],
    }

    # -------------------------------------------------------------
    # 12. REPRODUCIBILITY INFORMATION
    # -------------------------------------------------------------
    print("\n--- 12. REPRODUCIBILITY INFORMATION ---")
    repro = {
        "python_version": sys.version.split()[0],
        "pytorch_version": torch.__version__,
        "torchvision_version": sys.modules.get("torchvision").__version__ if "torchvision" in sys.modules else "N/A",
        "onnx_version": onnx.__version__,
        "onnxruntime_version": ort.__version__,
        "fastapi_version": sys.modules.get("fastapi").__version__ if "fastapi" in sys.modules else "N/A",
        "os": f"{platform.system()} {platform.release()} ({platform.version()})",
        "cpu": platform.processor(),
        "input_size": "224x224 (CenterCrop) after 256 Resize",
        "checkpoint": "best_model_v3.pth",
        "class_mapping": "class_mapping_v3.json",
        "seed": 42
    }
    for k, v in repro.items():
        print(f"  {k}: {v}")
    results["reproducibility"] = repro

    # -------------------------------------------------------------
    # 13. FILE EXISTENCE & LOADABILITY AUDIT
    # -------------------------------------------------------------
    print("\n--- 13. FILE EXISTENCE & LOADABILITY ---")
    files_to_check = {
        "final_model_checkpoint": MODELS_V3 / "final_model_v3.pth",
        "best_model_checkpoint": MODELS_V3 / "best_model_v3.pth",
        "class_mapping": MODELS_V3 / "class_mapping_v3.json",
        "test_manifest": REPORTS_V2 / "split_manifest_v2.csv",
        "evaluation_report": REPORTS_V3 / "final_model_evaluation_v3.md",
        "classification_report": REPORTS_V3 / "classification_report_v3.csv",
        "confusion_matrix": REPORTS_V3 / "confusion_matrix_v3.csv",
        "training_configuration": MODELS_V3 / "training_config_v3.json",
        "synthetic_metadata": REPORTS_V3 / "synthetic_metadata_v3.csv",
        "leakage_report": REPORTS_V3 / "leakage_report_v3.md",
        "onnx_model": MODELS_V3 / "efficientnet_b0_v3.onnx"
    }

    file_status = {}
    for name, path in files_to_check.items():
        exists = path.exists()
        size = path.stat().st_size if exists else 0
        file_status[name] = {"exists": exists, "size_bytes": size, "path": str(path)}
        print(f"  {name}: {'EXISTS' if exists else 'MISSING'} ({size} bytes)")
    results["files"] = file_status

    # Save complete audit dictionary
    with open(WORKSPACE / "scratch" / "audit_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print("\nAudit dictionary saved to scratch/audit_results.json")
    return results


if __name__ == "__main__":
    run_audit()
