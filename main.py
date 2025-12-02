# Tom Richards
from datasets import load_dataset
import pandas as pd
import csv
import re
from tqdm import tqdm
import unicodedata

DATASET = "hao-li/AIDev"
OUT_TASK1 = "task1_all_pull_request.csv"
OUT_TASK2 = "task2_all_repository.csv"
OUT_TASK3 = "task3_pr_task_type.csv"
OUT_TASK4 = "task4_pr_commit_details.csv"
OUT_TASK5 = "task5_summary.csv"

SECURITY_KEYWORDS = [
 "race", "racy", "buffer", "overflow", "stack", "integer", "signedness", "underflow",
 "improper", "unauthenticated", "gain access", "permission", "cross site", "css", "xss",
 "denial service", "dos", "crash", "deadlock", "injection", "request forgery", "csrf",
 "xsrf", "forged", "security", "vulnerability", "vulnerable", "exploit", "attack",
 "bypass", "backdoor", "threat", "expose", "breach", "violate", "fatal", "blacklist",
 "overrun", "insecure"
]
SEC_KEYWORDS_RE = re.compile(r'|'.join(re.escape(k) for k in SECURITY_KEYWORDS), flags=re.IGNORECASE)


def clean_patch(patch_text):
    if patch_text is None:
        return ""
    s = unicodedata.normalize("NFKC", patch_text)
    s = re.sub(r'[^\x09\x0A\x0D\x20-\x7E\u0080-\uFFFF]', ' ', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s


print("Loading dataset metadata (no heavy download)...")
ds_info = load_dataset(DATASET, split="train", streaming=True)


def stream_table(table_name):
    print(f"Streaming table: {table_name}")
    return load_dataset(DATASET, name=table_name, split="train", streaming=True)


# ---------- Task 1: all_pull_request -> CSV ----------
def task1():
    cols = ["TITLE", "ID", "AGENTNAME", "BODYSTRING", "REPOID", "REPOURL"]
    it = stream_table("all_pull_request")
    with open(OUT_TASK1, "w", newline='', encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(cols)
        for ex in tqdm(it, desc="task1"):
            title = ex.get("title", "")
            _id = ex.get("id", "")
            agent = ex.get("agent", "")
            body = ex.get("body", "")
            repo_id = ex.get("repo_id", "")
            repo_url = ex.get("repo_url", "")
            writer.writerow([title, _id, agent, body, repo_id, repo_url])
    print("Task1 finished ->", OUT_TASK1)


# ---------- Task 2: all_repository -> CSV ----------
def task2():
    cols = ["REPOID", "LANG", "STARS", "REPOURL"]
    it = stream_table("all_repository")
    with open(OUT_TASK2, "w", newline='', encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(cols)
        for ex in tqdm(it, desc="task2"):
            _id = ex.get("id", "")
            lang = ex.get("language", "")
            stars = ex.get("stars", "")
            url = ex.get("url", "")
            writer.writerow([_id, lang, stars, url])
    print("Task2 finished ->", OUT_TASK2)


# ---------- Task 3: pr_task_type -> CSV ----------
def task3():
    cols = ["PRID", "PRTITLE", "PRREASON", "PRTYPE", "CONFIDENCE"]
    it = stream_table("pr_task_type")
    with open(OUT_TASK3, "w", newline='', encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(cols)
        for ex in tqdm(it, desc="task3"):
            _id = ex.get("id", "")
            title = ex.get("title", "")
            reason = ex.get("reason", "")
            thetype = ex.get("type", "")
            confidence = ex.get("confidence", "")
            writer.writerow([_id, title, reason, thetype, confidence])
    print("Task3 finished ->", OUT_TASK3)


# ---------- Task 4: pr_commit_details -> CSV (clean patch) ----------
def task4():
    cols = ["PRID", "PRSHA", "PRCOMMITMESSAGE", "PRFILE", "PRSTATUS", "PRADDS", "PRDELSS", "PRCHANGECOUNT", "PRDIFF"]
    it = stream_table("pr_commit_details")
    with open(OUT_TASK4, "w", newline='', encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(cols)
        for ex in tqdm(it, desc="task4"):
            prid = ex.get("pr_id", "")
            sha = ex.get("sha", "")
            message = ex.get("message", "")
            filename = ex.get("filename", "")
            status = ex.get("status", "")
            additions = ex.get("additions", "")
            deletions = ex.get("deletions", "")
            changes = ex.get("changes", "")
            patch = clean_patch(ex.get("patch", ""))
            writer.writerow([prid, sha, message, filename, status, additions, deletions, changes, patch])
    print("Task4 finished ->", OUT_TASK4)


# ---------- Task 5: Join & SECURITY flag ----------
def task5():
    print("Loading Task3 into memory (pr_task_type) for joins...")
    df3 = pd.read_csv(OUT_TASK3, dtype=str)
    df3 = df3.fillna("")
    pr_type_map = df3.set_index("PRID").to_dict(orient="index")

    print("Reading Task1 and building Task5...")
    with open(OUT_TASK1, newline='', encoding="utf-8") as fh_in, \
         open(OUT_TASK5, "w", newline='', encoding="utf-8") as fh_out:
        r = csv.DictReader(fh_in)
        writer = csv.writer(fh_out)
        writer.writerow(["ID", "AGENT", "TYPE", "CONFIDENCE", "SECURITY"])
        for row in tqdm(r, desc="task5"):
            pr_id = str(row["ID"])
            agent = row["AGENTNAME"]
            title = row["TITLE"] or ""
            body = row["BODYSTRING"] or ""
            typ = ""
            conf = ""
            info = pr_type_map.get(pr_id)
            if info:
                typ = info.get("PRTYPE", "")
                conf = info.get("CONFIDENCE", "")
            combined = (title + " " + body).lower()
            security_flag = 1 if SECURITY_KEYWORDS and SEC_KEYWORDS_RE.search(combined) else 0
            writer.writerow([pr_id, agent, typ, conf, security_flag])
    print("Task5 finished ->", OUT_TASK5)


# ---------- Run all tasks ----------
if __name__ == "__main__":
    task1()
    task2()
    task3()
    task4()
    task5()
    print("All tasks 1-5 done.")
