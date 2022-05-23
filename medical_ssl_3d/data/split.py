from sklearn.model_selection import KFold


def kfold(ids, num_splits, random_state=42):
    kfold = KFold(num_splits, shuffle=True, random_state=random_state)
    for train_idx, test_idx in kfold.split(ids):
        yield [ids[i] for i in train_idx], [ids[i] for i in test_idx]
