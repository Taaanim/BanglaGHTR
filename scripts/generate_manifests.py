#!/usr/bin/env python3
"""
scripts/generate_manifests.py
Generates unified, leakage-free metadata manifests for dataset_1 (BN-HTR) and dataset_2 (Ekush Character dataset).
Features:
- Document-independent split for BN-HTR (105 train / 20 val / 25 test).
- Writer-independent split for dataset_2 (80% train / 10% val / 10% test).
- Full Unicode NFC normalization for all Bengali text.
- Case-insensitive image discovery (.jpg, .png, .jpeg, etc.).
- Automatic handling of known quirks (e.g., 60.xl.xlsx).
- Relative image paths stored for portability across local/remote environments.
"""

import os
import random
import unicodedata
import zipfile
import xml.etree.ElementTree as ET
import pandas as pd
from tqdm import tqdm

WORKSPACE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D1_ROOT = os.path.join(WORKSPACE_ROOT, "Dataset", "Raw_dataset", "BN-HTRd A Benchmark Dataset for Document Level Offline Bangla Handwritten Text Recognition (HTR)")
D2_ROOT = os.path.join(WORKSPACE_ROOT, "Dataset", "Raw_dataset", "dataset_Char")
MANIFEST_DIR = os.path.join(WORKSPACE_ROOT, "datasets", "manifests")
os.makedirs(MANIFEST_DIR, exist_ok=True)

IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.JPG', '.PNG', '.JPEG')

def get_xlsx_words(xlsx_path):
    """Extracts Id and Word pairs from .xlsx using lightweight zipfile + xml parsing."""
    entries = {}
    try:
        with zipfile.ZipFile(xlsx_path) as z:
            strings = []
            if 'xl/sharedStrings.xml' in z.namelist():
                tree = ET.fromstring(z.read('xl/sharedStrings.xml'))
                for si in tree.findall('{http://schemas.openxmlformats.org/spreadsheetml/2006/main}si'):
                    t = si.find('{http://schemas.openxmlformats.org/spreadsheetml/2006/main}t')
                    if t is not None and t.text:
                        strings.append(t.text)
                    else:
                        strings.append(''.join([c.text for c in si.iter('{http://schemas.openxmlformats.org/spreadsheetml/2006/main}t') if c.text]))

            tree = ET.fromstring(z.read('xl/worksheets/sheet1.xml'))
            rows = tree.findall('{http://schemas.openxmlformats.org/spreadsheetml/2006/main}sheetData/{http://schemas.openxmlformats.org/spreadsheetml/2006/main}row')
            for r in rows:
                cells = r.findall('{http://schemas.openxmlformats.org/spreadsheetml/2006/main}c')
                row_vals = []
                for c in cells:
                    t_attr = c.attrib.get('t')
                    v = c.find('{http://schemas.openxmlformats.org/spreadsheetml/2006/main}v')
                    if v is not None and v.text is not None:
                        val = v.text
                        if t_attr == 's' and val.isdigit() and int(val) < len(strings):
                            val = strings[int(val)]
                        row_vals.append(val)
                if len(row_vals) >= 2 and row_vals[0] != 'Id':
                    w_id = row_vals[0].strip()
                    word = unicodedata.normalize('NFC', str(row_vals[1]).strip())
                    entries[w_id] = word
    except Exception as e:
        print(f"Warning reading {xlsx_path}: {e}")
    return entries

def build_dataset_1_manifests():
    print("\n--- Building manifests for dataset_1 (BN-HTR) ---")
    bn_root = os.path.join(D1_ROOT, "BN-HTR_Dataset")
    gt_dir = os.path.join(bn_root, "Recognition_Ground_Truth_Texts")
    lines_dir = os.path.join(bn_root, "Segmentation_Images", "Lines")
    words_dir = os.path.join(bn_root, "Segmentation_Images", "Words")

    # 1. Load ground truth word texts
    doc_words = {}
    for d in range(1, 151):
        dp = os.path.join(gt_dir, str(d))
        if not os.path.exists(dp):
            continue
        xlsx_candidates = [f for f in os.listdir(dp) if f.endswith('.xlsx')]
        if xlsx_candidates:
            xlsx_path = os.path.join(dp, xlsx_candidates[0])
            doc_words[str(d)] = get_xlsx_words(xlsx_path)

    # 2. Document-Independent Split (105 train / 20 val / 25 test)
    random.seed(42)
    all_docs = [str(i) for i in range(1, 151)]
    random.shuffle(all_docs)
    train_docs = set(all_docs[:105])
    val_docs = set(all_docs[105:125])
    test_docs = set(all_docs[125:])

    word_records = []
    line_records = {}

    for doc_id in tqdm(range(1, 151), desc="Processing BN-HTR Documents"):
        d_str = str(doc_id)
        split_name = "train" if d_str in train_docs else ("val" if d_str in val_docs else "test")
        words_doc_dir = os.path.join(words_dir, d_str)
        lines_doc_dir = os.path.join(lines_dir, d_str)

        if not os.path.exists(words_doc_dir):
            continue

        page_folders = sorted([f for f in os.listdir(words_doc_dir) if os.path.isdir(os.path.join(words_doc_dir, f))])
        for page_id in page_folders:
            p_dir = os.path.join(words_doc_dir, page_id)
            line_folders = sorted([f for f in os.listdir(p_dir) if os.path.isdir(os.path.join(p_dir, f))])
            for line_id in line_folders:
                l_dir = os.path.join(p_dir, line_id)
                word_files = sorted([f for f in os.listdir(l_dir) if f.lower().endswith(IMAGE_EXTENSIONS)])

                line_words_list = []
                for wf in word_files:
                    wid = os.path.splitext(wf)[0]
                    text = doc_words.get(d_str, {}).get(wid, "")
                    abs_w_path = os.path.join(l_dir, wf)
                    rel_w_path = os.path.relpath(abs_w_path, WORKSPACE_ROOT)

                    word_records.append({
                        "word_id": wid,
                        "line_id": line_id,
                        "page_id": page_id,
                        "doc_id": d_str,
                        "split": split_name,
                        "image_path": rel_w_path,
                        "text": text
                    })
                    if text:
                        line_words_list.append((wid, text))

                # Match cropped line image
                line_img_candidates = [
                    os.path.join(lines_doc_dir, page_id, f"{line_id}{ext}")
                    for ext in ['.jpg', '.png', '.jpeg', '.JPG', '.PNG', '.JPEG']
                ]
                line_img_path = next((p for p in line_img_candidates if os.path.exists(p)), "")
                rel_line_path = os.path.relpath(line_img_path, WORKSPACE_ROOT) if line_img_path else ""

                # Reconstruct full line text by ordering words by index
                def extract_word_idx(item):
                    parts = item[0].split('_')
                    return int(parts[-1]) if (parts and parts[-1].isdigit()) else 0

                sorted_words = [w[1] for w in sorted(line_words_list, key=extract_word_idx)]
                line_text = " ".join(sorted_words)

                line_records[line_id] = {
                    "line_id": line_id,
                    "page_id": page_id,
                    "doc_id": d_str,
                    "split": split_name,
                    "image_path": rel_line_path,
                    "text": line_text,
                    "num_words": len(sorted_words)
                }

    df_words = pd.DataFrame(word_records)
    words_path = os.path.join(MANIFEST_DIR, "dataset_1_words.csv")
    df_words.to_csv(words_path, index=False)
    print(f"Saved {len(df_words)} words to {words_path}")

    df_lines = pd.DataFrame(list(line_records.values()))
    lines_path = os.path.join(MANIFEST_DIR, "dataset_1_lines.csv")
    df_lines.to_csv(lines_path, index=False)
    print(f"Saved {len(df_lines)} lines to {lines_path}")

def build_dataset_2_manifests():
    print("\n--- Building manifests for dataset_2 (Ekush Characters) ---")
    if not os.path.exists(D2_ROOT):
        print(f"Error: {D2_ROOT} not found!")
        return

    # Build Class Taxonomy Mapping
    vowel_kars = [' া', ' ি', ' ী', ' ু', ' ূ', ' ৃ', ' ে', ' ৈ', ' ো', ' ৌ']
    basic_vowels = ['অ', 'আ', 'ই', 'ঈ', 'উ', 'ঊ', 'ঋ', 'এ', 'ঐ', 'ও', 'ঔ']
    consonants = ['ক', 'খ', 'গ', 'ঘ', 'ঙ', 'চ', 'ছ', 'জ', 'ঝ', 'ঞ', 'ট', 'ঠ', 'ড', 'ঢ', 'ণ',
                  'ত', 'থ', 'দ', 'ধ', 'ন', 'প', 'ফ', 'ব', 'ভ', 'ম', 'য', 'র', 'ল', 'শ', 'ষ',
                  'স', 'হ', 'ড়', 'ঢ়', 'য়', 'ৎ', 'ং', 'ঃ', 'ঁ']
    numerals = ['০', '১', '২', '৩', '৪', '৫', '৬', '৭', '৮', '৯']

    taxonomy = []
    for c_id in range(122):
        if 0 <= c_id <= 9:
            cat = "Vowel Diacritic (Kar)"
            char = vowel_kars[c_id]
        elif 10 <= c_id <= 20:
            cat = "Independent Vowel"
            char = basic_vowels[c_id - 10]
        elif 21 <= c_id <= 59:
            cat = "Consonant"
            char = consonants[c_id - 21]
        elif 60 <= c_id <= 111:
            cat = "Compound Character (Yuktakshar)"
            char = f"যুক্তবর্ণ_{c_id}"
        elif 112 <= c_id <= 121:
            cat = "Bengali Numeral"
            char = numerals[c_id - 112]
        else:
            cat = "Unknown"
            char = ""
        taxonomy.append({
            "class_id": c_id,
            "category": cat,
            "char_sample": unicodedata.normalize('NFC', char)
        })

    df_tax = pd.DataFrame(taxonomy)
    tax_path = os.path.join(MANIFEST_DIR, "char_classes_122.csv")
    df_tax.to_csv(tax_path, index=False)
    print(f"Saved 122 class taxonomy to {tax_path}")

    # Discover images and writer IDs
    records = []
    writers = set()

    c_folders = sorted(os.listdir(D2_ROOT), key=lambda x: int(x) if x.isdigit() else 999)
    for c_dir in tqdm(c_folders, desc="Processing Character Folders"):
        cp = os.path.join(D2_ROOT, c_dir)
        if not os.path.isdir(cp):
            continue
        c_id = int(c_dir) if c_dir.isdigit() else -1

        for fn in os.listdir(cp):
            if fn.lower().endswith(IMAGE_EXTENSIONS):
                parts = os.path.splitext(fn)[0].split('_')
                if len(parts) >= 5:
                    gender, dist, age, form, writer_id = parts[:5]
                    w_uid = f"{gender}_{dist}_{writer_id}"
                else:
                    gender, dist, age, form, writer_id = "0", "UNKNOWN", "0", "0", parts[0]
                    w_uid = fn

                writers.add(w_uid)
                abs_p = os.path.join(cp, fn)
                rel_p = os.path.relpath(abs_p, WORKSPACE_ROOT)

                records.append({
                    "filename": fn,
                    "class_id": c_id,
                    "gender": int(gender) if gender.isdigit() else 0,
                    "district": dist,
                    "age": int(age) if age.isdigit() else 0,
                    "form_id": form,
                    "writer_id": w_uid,
                    "image_path": rel_p
                })

    # Writer-Independent Split (80% train / 10% val / 10% test)
    random.seed(42)
    writers_list = sorted(list(writers))
    random.shuffle(writers_list)
    n_w = len(writers_list)
    train_w = set(writers_list[:int(n_w * 0.8)])
    val_w = set(writers_list[int(n_w * 0.8):int(n_w * 0.9)])

    for r in records:
        w = r["writer_id"]
        r["split"] = "train" if w in train_w else ("val" if w in val_w else "test")

    df_chars = pd.DataFrame(records)
    chars_path = os.path.join(MANIFEST_DIR, "dataset_2_chars.csv")
    df_chars.to_csv(chars_path, index=False)
    print(f"Saved {len(df_chars)} character instances across {len(writers)} writers to {chars_path}")

if __name__ == "__main__":
    build_dataset_1_manifests()
    build_dataset_2_manifests()
    print("\nAll manifests successfully generated!")
