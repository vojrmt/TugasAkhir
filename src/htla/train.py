import os
import json
import argparse
import numpy as np
import torch
import pandas as pd
from torch.utils.data import DataLoader, Subset
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from tqdm import tqdm
from dataset import PANDORADataset
from model import HTLA

# Get the directory where train.py is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ── Args ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--debug", action="store_true",
                    help="Run 2 batches only to verify code correctness")

# Set default path relative to train.py's location. 
# Added a trailing slash "/" so your existing DATA_PATH + "filename.csv" code still works perfectly!
parser.add_argument("--data_path", default=os.path.join(BASE_DIR, "../../data/processed/"))
parser.add_argument("--epochs",    type=int, default=10)
parser.add_argument("--batch_size",type=int, default=4)
parser.add_argument("--lr",          type=float, default=1e-5)
parser.add_argument("--accum_steps", type=int,   default=4,
                    help="Gradient accumulation steps. Effective batch = batch_size * accum_steps. "
                         "Default 4 gives effective batch of 16, which stabilizes extraversion "
                         "gradients (35%% positive rate is too noisy at bare batch_size=4).")
args = parser.parse_args()

# ── Config ────────────────────────────────────────────────────────────────────
DATA_PATH    = args.data_path
BATCH_SIZE   = 2 if args.debug else args.batch_size
NUM_EPOCHS   = 1 if args.debug else args.epochs
LR           = args.lr
ACCUM_STEPS  = 1 if args.debug else args.accum_steps
WARMUP_STEPS = 100
DROPOUT      = 0.1
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TRAIT_NAMES  = ["agreeableness", "openness", "conscientiousness",
                "extraversion", "neuroticism"]

print(f"Device: {DEVICE}")
print(f"Debug mode: {args.debug}")
print(f"Effective batch size: {BATCH_SIZE} x {ACCUM_STEPS} accum = {BATCH_SIZE * ACCUM_STEPS}")
print("-" * 60)

# ── Load data ─────────────────────────────────────────────────────────────────
print("Loading data...")
profiles = pd.read_csv(DATA_PATH + "profiles_labeled.csv")
comments = pd.read_csv(DATA_PATH + "comments_capped.csv")

with open(DATA_PATH + "splits.json") as f:
    splits = json.load(f)
with open(DATA_PATH + "pos_weights.json") as f:
    pw          = json.load(f)
    pos_weights = torch.tensor(pw["pos_weights"], dtype=torch.float).to(DEVICE)

# ── Datasets ──────────────────────────────────────────────────────────────────
print("\nBuilding train dataset...")
train_dataset = PANDORADataset(splits["train"], profiles, comments)

print("Building val dataset...")
val_dataset   = PANDORADataset(splits["val"],   profiles, comments)

# In debug mode: only use first 4 users from each split
if args.debug:
    train_dataset = Subset(train_dataset, list(range(4)))
    val_dataset   = Subset(val_dataset,   list(range(4)))
    print("DEBUG: using 4 users per split, 2 batches max")

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                          shuffle=True,  num_workers=0)
val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE,
                          shuffle=False, num_workers=0)

import torch.nn.functional as F

class AdaptiveFocalLoss(torch.nn.Module):
    def __init__(self, pos_weights):
        super().__init__()
        self.pos_weights = pos_weights
        # Trainable gamma parameter (initialized to roughly 2.0)
        # We use a log-parameter so we can use torch.exp() to guarantee gamma stays > 0
        self.log_gamma = torch.nn.Parameter(torch.tensor(0.693)) 

    def forward(self, logits, targets):
        # 1. Calculate standard unweighted BCE for the focal factor (Z_BCE in the paper)
        pure_bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        
        # 2. Calculate the weighted BCE (w * Z_BCE)
        weighted_bce = F.binary_cross_entropy_with_logits(
            logits, targets, pos_weight=self.pos_weights, reduction='none'
        )
        
        gamma = torch.exp(self.log_gamma)
        
        # 3. Calculate Focal Factor: (1 - e^(-Z_BCE))^gamma
        focal_factor = torch.pow(1.0 - torch.exp(-pure_bce), gamma)
        
        # 4. FBCE-T: focal_factor * weighted_bce + regularization on gamma
        # Bug in original: "+ self.log_gamma" gets averaged across all batch*dim
        # terms, shrinking the effective gradient to ~1e-5 — gamma barely moves.
        # Fix: add as a scaled scalar so the gradient magnitude is meaningful.
        sample_loss = focal_factor * weighted_bce
        reg  = 0.01 * self.log_gamma
        loss = sample_loss + reg

        # 5. FBCE-M: Average across all dimensions and batch samples
        return loss.mean()

# ── Model, loss, optimizer ────────────────────────────────────────────────────
print("\nLoading model...")
model     = HTLA(dropout=DROPOUT).to(DEVICE)

# Use the new Multi-Dimensional Adaptive Focal Loss
criterion = AdaptiveFocalLoss(pos_weights=pos_weights).to(DEVICE)

# Separate LR for gamma: BERT grads are ~1e-5 scale, but gamma is a single
# scalar that needs a much higher LR to move meaningfully.
# Using the same LR buries gamma's gradient under BERT's parameter mass.
optimizer = AdamW(
    [
        {"params": model.parameters(),     "lr": LR,     "weight_decay": 0.01},
        {"params": criterion.parameters(), "lr": 1e-3,   "weight_decay": 0.0},
    ]
)

total_steps = len(train_loader) * NUM_EPOCHS // ACCUM_STEPS
scheduler   = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=min(WARMUP_STEPS, total_steps),
    num_training_steps=total_steps
)

# ── Eval ──────────────────────────────────────────────────────────────────────
def evaluate(loader, desc="Evaluating"):
    model.eval()
    all_preds, all_labels = [], []
    total_loss = 0.0

    with torch.no_grad():
        for batch in tqdm(loader, desc=desc, leave=False):
            ids   = batch["input_ids"].to(DEVICE)
            masks = batch["attention_masks"].to(DEVICE)
            cmask = batch["comment_mask"].to(DEVICE)
            lbls  = batch["labels"].to(DEVICE)

            logits, _ = model(ids, masks, cmask)
            loss      = criterion(logits, lbls)
            total_loss += loss.item()

            preds = (torch.sigmoid(logits) >= 0.5).int().cpu().numpy()
            all_preds.append(preds)
            all_labels.append(lbls.int().cpu().numpy())

    all_preds  = np.vstack(all_preds)
    all_labels = np.vstack(all_labels)

    results = {"loss": total_loss / len(loader)}
    for i, trait in enumerate(TRAIT_NAMES):
        results[trait] = {
            "accuracy":  round(accuracy_score (all_labels[:, i], all_preds[:, i]), 4),
            "f1":        round(f1_score       (all_labels[:, i], all_preds[:, i], zero_division=0), 4),
            "precision": round(precision_score(all_labels[:, i], all_preds[:, i], zero_division=0), 4),
            "recall":    round(recall_score   (all_labels[:, i], all_preds[:, i], zero_division=0), 4),
        }
    return results

# ── Training loop ─────────────────────────────────────────────────────────────
best_val_f1      = 0.0
history          = []
patience_counter = 0
PATIENCE_LIMIT   = 3

print(f"\nStarting training | {len(train_loader)} batches/epoch | {NUM_EPOCHS} epoch(s)")
print(f"Gradient accumulation: every {ACCUM_STEPS} steps (effective batch = {BATCH_SIZE * ACCUM_STEPS})")
print("=" * 60)

for epoch in range(1, NUM_EPOCHS + 1):
    model.train()
    train_loss = 0.0
    train_bar  = tqdm(train_loader, desc=f"Epoch {epoch}/{NUM_EPOCHS} [train]")

    optimizer.zero_grad()  # zero once before the loop, not inside

    for step, batch in enumerate(train_bar):
        ids   = batch["input_ids"].to(DEVICE)
        masks = batch["attention_masks"].to(DEVICE)
        cmask = batch["comment_mask"].to(DEVICE)
        lbls  = batch["labels"].to(DEVICE)

        logits, _ = model(ids, masks, cmask)
        # Scale loss by accum steps so gradients average correctly
        loss = criterion(logits, lbls) / ACCUM_STEPS
        loss.backward()

        train_loss += loss.item() * ACCUM_STEPS  # unscale for logging

        # Only step when we've accumulated enough gradients
        if (step + 1) % ACCUM_STEPS == 0 or (step + 1) == len(train_loader):
            # Clip grads for BOTH model and criterion (log_gamma included)
            torch.nn.utils.clip_grad_norm_(
                list(model.parameters()) + list(criterion.parameters()),
                max_norm=1.0
            )
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        train_bar.set_postfix(loss=f"{train_loss/(step+1):.4f}")

        # Debug mode: stop after 2 batches
        if args.debug and step >= 1:
            print("\nDEBUG: 2 batches completed successfully. Stopping train loop.")
            break

    val_results = evaluate(val_loader, desc=f"Epoch {epoch}/{NUM_EPOCHS} [val]")
    avg_f1      = sum(val_results[t]["f1"] for t in TRAIT_NAMES) / 5

    # Extract the current value of the trainable focal parameter (gamma)
    # We use torch.exp because it was stored as log_gamma to keep it positive
    current_gamma = torch.exp(criterion.log_gamma).item()

    print(f"\nEpoch {epoch} — train_loss: {train_loss/(step+1):.4f} "
          f"| val_loss: {val_results['loss']:.4f} | avg_f1: {avg_f1:.4f} | gamma: {current_gamma:.4f}")
    
    for trait in TRAIT_NAMES:
        r = val_results[trait]
        print(f"  {trait:<20} acc={r['accuracy']}  f1={r['f1']}  "
              f"prec={r['precision']}  rec={r['recall']}")

    # Save gamma to history so you can plot it later if needed
    history.append({"epoch": epoch, "val": val_results, "avg_f1": avg_f1, "gamma": current_gamma})

    # Early Stopping & Model Saving Logic
    if not args.debug:
        if avg_f1 > best_val_f1:
            best_val_f1 = avg_f1
            patience_counter = 0  # Reset patience because we improved!
            torch.save(model.state_dict(), "best_model.pt")
            print(f"  ✓ Saved best model (avg F1 = {best_val_f1:.4f})")
        else:
            patience_counter += 1
            print(f"  ! No improvement. Early stopping patience: {patience_counter}/{PATIENCE_LIMIT}")
            
            if patience_counter >= PATIENCE_LIMIT:
                print(f"\n  [!] Validation F1 hasn't improved for {PATIENCE_LIMIT} epochs.")
                print("  [!] Early stopping triggered. Halting training to save time.")
                break  # Kills the epoch loop

    print("-" * 60)

    if args.debug:
        print("\nDEBUG run complete. No errors. Code is ready for GPU.")
        break

if not args.debug:
    with open("training_history.json", "w") as f:
        json.dump(history, f, indent=2)
    print(f"\nDone. Best val avg F1: {best_val_f1:.4f}")