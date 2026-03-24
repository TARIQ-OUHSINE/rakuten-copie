"""
Phase 4 — OpenAI GPT-4o Vision Fine-Tuning (text + image multimodal)

Fine-tunes gpt-4o-2024-08-06 on 1,000 stratified samples for product classification.

Usage:
    python models/openai_finetune.py prepare
        → Builds data/openai_train.jsonl (1,000 stratified samples)

    python models/openai_finetune.py train
        → Uploads JSONL and launches fine-tuning job (~$10)

    python models/openai_finetune.py train --status JOB_ID
        → Checks status of an existing fine-tuning job

    python models/openai_finetune.py predict --model ft:gpt-4o-2024-08-06:org::id
        → Runs inference on test set, saves data/openai_test_predictions.csv (~$9)

Requirements:
    pip install openai pillow pandas scikit-learn tqdm python-dotenv
    OPENAI_API_KEY must be set in .env at the project root

Input:
    data/train_cleaned.pkl
    data/test_cleaned.pkl
    images/image_train/image_{imageid}_product_{productid}.jpg
    images/image_test/image_{imageid}_product_{productid}.jpg

Output:
    data/openai_train.jsonl       (1,000-sample fine-tuning file)
    data/openai_test_predictions.csv
"""

import argparse
import base64
import io
import json
import os
import time

from dotenv import load_dotenv
import pandas as pd

# Load OPENAI_API_KEY from .env at project root
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"))
from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")
IMG_TRAIN_DIR = os.path.join(ROOT, "images", "image_train")
IMG_TEST_DIR = os.path.join(ROOT, "images", "image_test")

TRAIN_JSONL = os.path.join(DATA_DIR, "openai_train.jsonl")
PRED_CSV = os.path.join(DATA_DIR, "openai_test_predictions.csv")

MODEL_ID = "gpt-4o-2024-08-06"
SAMPLE_SIZE = 500
RANDOM_STATE = 42

SYSTEM_PROMPT = (
    "You are an e-commerce product classifier. "
    "Given a product title, description, and image, output only the product "
    "category code as a single integer. Do not explain."
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def encode_image(image_path: str) -> str:
    """Load image, resize to 512x512, return base64-encoded JPEG string."""
    with Image.open(image_path).convert("RGB") as img:
        img = img.resize((512, 512), Image.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=85)
        return base64.b64encode(buf.getvalue()).decode("utf-8")


def image_path(row, split: str) -> str:
    img_dir = IMG_TRAIN_DIR if split == "train" else IMG_TEST_DIR
    return os.path.join(img_dir, f"image_{row.imageid}_product_{row.productid}.jpg")


def build_message(row, split: str, label=None) -> dict:
    """Build a single chat message dict for fine-tuning or inference."""
    description = row.description if pd.notna(row.description) and str(row.description).strip() else ""
    text = f"Title: {row.designation}"
    if description:
        text += f"\nDescription: {description}"

    b64 = encode_image(image_path(row, split))

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": text},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{b64}",
                        "detail": "low",
                    },
                },
            ],
        },
    ]

    if label is not None:
        messages.append({"role": "assistant", "content": str(label)})

    return {"messages": messages}


# ── Subcommands ───────────────────────────────────────────────────────────────

def cmd_prepare(args):
    """Build JSONL fine-tuning file from 1,000 stratified training samples."""
    print(f"Loading data/train_cleaned.pkl ...")
    train = pd.read_pickle(os.path.join(DATA_DIR, "train_cleaned.pkl"))

    print(f"Sampling {SAMPLE_SIZE} stratified examples across 27 classes ...")
    _, sample = train_test_split(
        train,
        test_size=SAMPLE_SIZE,
        stratify=train["prdtypecode"],
        random_state=RANDOM_STATE,
    )

    print(f"Building JSONL → {TRAIN_JSONL}")
    failed = 0
    with open(TRAIN_JSONL, "w", encoding="utf-8") as f:
        for row in tqdm(sample.itertuples(), total=len(sample)):
            path = image_path(row, "train")
            if not os.path.exists(path):
                failed += 1
                continue
            try:
                msg = build_message(row, "train", label=row.prdtypecode)
                f.write(json.dumps(msg, ensure_ascii=False) + "\n")
            except Exception as e:
                failed += 1
                print(f"  Warning: skipped {path}: {e}")

    written = SAMPLE_SIZE - failed
    print(f"Done. {written} examples written ({failed} skipped). → {TRAIN_JSONL}")


def cmd_train(args):
    """Upload JSONL and launch (or check) a fine-tuning job."""
    from openai import OpenAI
    client = OpenAI()

    if args.status:
        job = client.fine_tuning.jobs.retrieve(args.status)
        print(f"Job {job.id}: status={job.status}")
        
        # Show error details if job failed
        if job.status == "failed":
            print(f"\n❌ Error details:")
            if hasattr(job, 'error') and job.error:
                error = job.error
                if hasattr(error, 'message'):
                    print(f"   Message: {error.message}")
                if hasattr(error, 'code'):
                    print(f"   Code: {error.code}")
                else:
                    print(f"   Error: {error}")
            else:
                print("   (No error details available)")
        
        # Show additional useful info
        if hasattr(job, 'trained_tokens') and job.trained_tokens:
            print(f"Trained tokens: {job.trained_tokens}")
        if hasattr(job, 'training_file'):
            print(f"Training file: {job.training_file}")
        if hasattr(job, 'validation_file') and job.validation_file:
            print(f"Validation file: {job.validation_file}")
            
        if job.status == "succeeded":
            print(f"\n✅ Fine-tuned model: {job.fine_tuned_model}")
        elif job.status == "failed":
            print(f"\n💡 To investigate further:")
            print(f"   1. Check the training file format: python models/openai_finetune.py prepare")
            print(f"   2. Verify JSONL is valid: head -1 {TRAIN_JSONL} | python3 -m json.tool")
            print(f"   3. Check OpenAI dashboard: https://platform.openai.com/finetune")
        return

    if not os.path.exists(TRAIN_JSONL):
        print(f"ERROR: {TRAIN_JSONL} not found. Run 'prepare' first.")
        return

    print(f"Uploading {TRAIN_JSONL} ...")
    with open(TRAIN_JSONL, "rb") as f:
        upload = client.files.create(file=f, purpose="fine-tune")
    print(f"Uploaded file ID: {upload.id}")

    print(f"Creating fine-tuning job (model={MODEL_ID}) ...")
    job = client.fine_tuning.jobs.create(
        training_file=upload.id,
        model=MODEL_ID,
    )
    print(f"Job created: {job.id}")
    print(f"Check progress: python models/openai_finetune.py train --status {job.id}")


def build_text_only_message(row) -> dict:
    """Build a text-only chat message (no image) for inference."""
    description = row.description if pd.notna(row.description) and str(row.description).strip() else ""
    text = f"Title: {row.designation}"
    if description:
        text += f"\nDescription: {description}"
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": text},
        ]
    }


def cmd_predict(args):
    """Run inference on test_cleaned.pkl with the fine-tuned model."""
    from openai import OpenAI
    client = OpenAI()

    if not args.model:
        print("ERROR: --model required (e.g. ft:gpt-4o-2024-08-06:org::id)")
        return

    if args.no_image:
        print("Mode: text-only (--no-image)")

    print(f"Loading data/test_cleaned.pkl ...")
    test = pd.read_pickle(os.path.join(DATA_DIR, "test_cleaned.pkl"))

    predictions = []
    errors = 0

    print(f"Running inference on {len(test)} examples with {args.model} ...")
    for row in tqdm(test.itertuples(), total=len(test)):
        if args.no_image:
            msg = build_text_only_message(row)
        else:
            path = image_path(row, "test")
            if not os.path.exists(path):
                predictions.append(None)
                errors += 1
                continue
            msg = build_message(row, "test")

        success = False
        for attempt in range(3):
            try:
                response = client.chat.completions.create(
                    model=args.model,
                    messages=msg["messages"],
                    max_tokens=10,
                    temperature=0,
                )
                raw = response.choices[0].message.content.strip()
                predictions.append(int(raw))
                success = True
                break
            except Exception as e:
                err_str = str(e)
                if "image_url is only supported by certain models" in err_str:
                    print(
                        "\nERROR: This fine-tuned model does not support images.\n"
                        "Re-run with --no-image flag:\n"
                        f"  python models/openai_finetune.py predict --model {args.model} --no-image"
                    )
                    return
                wait = 2 ** attempt
                print(f"  Retry {attempt + 1}/3 after {wait}s: {e}")
                time.sleep(wait)

        if not success:
            predictions.append(None)
            errors += 1

        # Rate limiting
        time.sleep(0.1)

    out = pd.DataFrame({
        "index": test.index,
        "productid": test["productid"].values,
        "prdtypecode": predictions,
    })
    out.to_csv(PRED_CSV, index=False)
    print(f"\nDone. {len(test) - errors}/{len(test)} predictions saved → {PRED_CSV}")
    if errors:
        print(f"  {errors} rows have None (missing image or repeated API failure)")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="OpenAI GPT-4o fine-tuning for Rakuten product classification"
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("prepare", help="Build openai_train.jsonl (1,000 samples)")

    p_train = sub.add_parser("train", help="Upload JSONL and launch fine-tuning")
    p_train.add_argument("--status", metavar="JOB_ID", help="Check status of an existing job")

    p_pred = sub.add_parser("predict", help="Run inference on test set")
    p_pred.add_argument("--model", required=True, metavar="MODEL_ID",
                        help="Fine-tuned model ID (ft:gpt-4o-2024-08-06:...)")
    p_pred.add_argument("--no-image", action="store_true",
                        help="Text-only inference (use if model was fine-tuned without images)")

    args = parser.parse_args()

    if args.cmd == "prepare":
        cmd_prepare(args)
    elif args.cmd == "train":
        cmd_train(args)
    elif args.cmd == "predict":
        cmd_predict(args)


if __name__ == "__main__":
    main()
