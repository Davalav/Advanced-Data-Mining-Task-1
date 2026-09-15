import itertools
import numpy as np
import lib
records =lib.train_val_records

NUM_CLASSES = 5




# ---------------------------------------------------
# Precompute class counts per record
# ---------------------------------------------------

record_counts = {}

for rec in records:
    ds = lib.MITBIHDataset([rec], window_size=256)

    counts = np.bincount(
        ds.labels,
        minlength=NUM_CLASSES
    )

    record_counts[rec] = counts

    total = counts.sum()

    print(
        f"{rec}:",
        np.round(100 * counts / total, 3)
    )


# ---------------------------------------------------
# Global target distribution
# ---------------------------------------------------

global_counts = np.sum(
    list(record_counts.values()),
    axis=0
)

global_dist = global_counts / global_counts.sum()

print("\nGlobal distribution:")
print(np.round(100 * global_dist, 3))


# ---------------------------------------------------
# Score split
# ---------------------------------------------------

def score_split(train_records):
    val_records = [
        r for r in records
        if r not in train_records
    ]

    train_counts = np.sum(
        [record_counts[r] for r in train_records],
        axis=0
    )

    val_counts = np.sum(
        [record_counts[r] for r in val_records],
        axis=0
    )

    train_dist = train_counts / train_counts.sum()
    val_dist = val_counts / val_counts.sum()

    # train-vs-val similarity
    distribution_loss = np.sum(
        (train_dist - val_dist) ** 2
    )

    return (
        distribution_loss,
        train_dist,
        val_dist,
        train_counts,
        val_counts
    )


# ---------------------------------------------------
# Search all 80/20 splits
# ---------------------------------------------------
split=0.91
n_train = int(round(len(records) * split))

best_score = float("inf")
best_result = None

num_checked = 0

for train_records in itertools.combinations(
    records,
    n_train
):

    score, train_dist, val_dist, train_counts, val_counts = (
        score_split(train_records)
    )

    if score < best_score:
        best_score = score
        best_result = (
            train_records,
            train_dist,
            val_dist,
            train_counts,
            val_counts
        )

    num_checked += 1

print(f"\nChecked {num_checked:,} splits")


# ---------------------------------------------------
# Display best split
# ---------------------------------------------------

(
    best_train,
    best_train_dist,
    best_val_dist,
    best_train_counts,
    best_val_counts
) = best_result

best_val = [
    r for r in records
    if r not in best_train
]

print("\nBEST TRAIN SET")
print(list(best_train))

print("\nBEST VAL SET")
print(best_val)

print("\nTrain distribution (%)")
print(np.round(100 * best_train_dist, 3))

print("\nVal distribution (%)")
print(np.round(100 * best_val_dist, 3))

print("\nAbs difference (%)")
print(
    np.round(
        100 * np.abs(
            best_train_dist - best_val_dist
        ),
        3
    )
)