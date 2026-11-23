import os
import shutil
import subprocess
from ultralytics import YOLO

MODEL_PATH = "yolo11m-cls_best.pt"
DATA_POOL_DIR = "v5_egitim_havuzu"
RETRAIN_THRESHOLD = 50

def retrain_model():
    print("Starting incremental training for V5...")
    try:
        model = YOLO(MODEL_PATH)
        # Incremental train: epochs=15, imgsz=224
        # Assuming dataset yaml was prepared or we map folders to classes
        # In YOLO classification, passing the directory root usually suffices
        results = model.train(data=DATA_POOL_DIR, epochs=15, imgsz=224, project="scanvitals_v5", name="incremental_run")
        
        # Validate improvement
        metrics = model.val()
        current_acc = metrics.top1
        
        # Backup and Replace if improved
        backup_path = MODEL_PATH + ".bak"
        shutil.copy(MODEL_PATH, backup_path)
        
        # Move best weight back to primary path
        best_weight = os.path.join("scanvitals_v5", "incremental_run", "weights", "best.pt")
        if os.path.exists(best_weight):
            shutil.copy(best_weight, MODEL_PATH)
            print(f"Model updated! New Accuracy: {current_acc:.4f}")
        else:
            print("Training completed but best weights not found.")
            
    except Exception as e:
        print(f"Auto-retraining failed: {e}")

if __name__ == "__main__":
    # Check pool size
    total_examples = sum([len(files) for r, d in os.walk(DATA_POOL_DIR) for files in [d]])
    if total_examples >= RETRAIN_THRESHOLD:
        retrain_model()
    else:
        print(f"Threshold not reached. Current samples: {total_examples}/{RETRAIN_THRESHOLD}")
