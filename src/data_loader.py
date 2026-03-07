from sklearn.datasets import fetch_20newsgroups


def load_dataset():

    dataset = fetch_20newsgroups(
        subset="all",
        remove=("headers", "footers", "quotes")
    )

    documents = dataset.data
    labels = dataset.target

    return documents, labels