import pandas as pd


REVIEW_PATH = "data/evidence_review_50.csv"


def main():

    df = pd.read_csv(REVIEW_PATH)

    # Remove rows that haven't been reviewed yet
    reviewed = df[
        df["useful_evidence"]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.upper()
        .isin(["YES", "NO"])
    ].copy()

    if reviewed.empty:
        print("No reviewed examples found.")
        print("Fill useful_evidence with YES or NO first.")
        return

    yes_count = (
        reviewed["useful_evidence"]
        .str.strip()
        .str.upper()
        .eq("YES")
        .sum()
    )

    no_count = (
        reviewed["useful_evidence"]
        .str.strip()
        .str.upper()
        .eq("NO")
        .sum()
    )

    total = len(reviewed)

    evidence_recall_at_5 = yes_count / total

    print("\n" + "=" * 60)
    print("EVIDENCE EVALUATION")
    print("=" * 60)

    print(f"\nReviewed examples : {total}")
    print(f"Useful evidence   : {yes_count}")
    print(f"Not useful        : {no_count}")

    print(
        f"\nEvidence Recall@5 : "
        f"{evidence_recall_at_5:.4f}"
    )

    print(
        f"Evidence Recall@5 : "
        f"{evidence_recall_at_5 * 100:.1f}%"
    )


if __name__ == "__main__":
    main()