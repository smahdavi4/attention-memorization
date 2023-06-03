import logging
import os
import numpy as np
import torch
from torch.utils.data import Dataset, random_split
import datasets
import transformers
import tokenizers


class TaskDataset(Dataset):
    def __init__(self, n_dims, seed, n_datapoints, dataset_size) -> None:
        super().__init__()
        self.rng = np.random.RandomState(seed)
        self.n_dims = n_dims
        self.dataset_size = dataset_size
        self.n_datapoints = n_datapoints
        self.X, self.y = self._construct_dataset()

    def __getitem__(self, index):
        return self.X[index], self.y[index]

    def __len__(self):
        return self.dataset_size

    def _construct_dataset(self):
        X = self.rng.binomial(1, p=0.5, size=(self.dataset_size, self.n_datapoints, self.n_dims))
        y = np.cumsum(X, axis=1).sum(axis=-1) % 2
        X = torch.tensor(X).float()
        y = torch.tensor(y).float()
        return X, y

    def to(self, device):
        self.X, self.y = self.X.to(device), self.y.to(device)


class LinearDataset(TaskDataset):
    def __init__(self, n_dims, seed, n_datapoints, dataset_size, feat_cov_coef, theta_cov_coef) -> None:
        assert 0.0 <= feat_cov_coef <= 1.0, "Feature Covariance must be between zero and one!"
        assert 0.0 <= theta_cov_coef <= 1.0, "Theta Covariance must be between zero and one!"
        self.feat_cov = (1 - feat_cov_coef) * np.eye(n_dims) + feat_cov_coef
        self.theta_cov = (1 - theta_cov_coef) * np.eye(n_dims) + theta_cov_coef
        super(LinearDataset, self).__init__(n_dims, seed, n_datapoints, dataset_size)

    def _construct_dataset(self):
        num_datapoints = self.n_datapoints
        X = self.rng.multivariate_normal(
            mean=np.zeros(self.n_dims), cov=self.feat_cov, size=(self.dataset_size, num_datapoints)
        )
        w = self.rng.multivariate_normal(
            mean=np.zeros(self.n_dims), cov=self.theta_cov, size=(self.dataset_size, 1)
        )
        y = (X * w).sum(axis=-1)
        X = torch.tensor(X).float()
        y = torch.tensor(y).float()
        return X, y


class SignLinearDataset(LinearDataset):
    def __init__(self, same_sign: bool, **kwargs) -> None:
        self.same_sign = same_sign
        super(SignLinearDataset, self).__init__(**kwargs)

    def _construct_dataset(self):
        num_datapoints = self.n_datapoints
        X_signs = 2 * self.rng.binomial(1, p=0.5, size=(self.dataset_size, 1, 1)) - 1
        X = np.abs(self.rng.multivariate_normal(
            mean=np.zeros(self.n_dims), cov=self.feat_cov, size=(self.dataset_size, num_datapoints)
        )) * X_signs
        w_signs = 1 if self.same_sign else -1
        w = np.abs(self.rng.multivariate_normal(
            mean=np.zeros(self.n_dims), cov=self.theta_cov, size=(self.dataset_size, 1)
        )) * w_signs
        y = (X * w).sum(axis=-1)
        X = torch.tensor(X).float()
        y = torch.tensor(y).float()
        return X, y


class WikiDataset(Dataset):
    VAL_SIZE = 10_000
    TEST_SIZE = 10_000

    def __init__(self, context_size, tokenizer_type, split='train', max_size=-1):
        super().__init__()
        self.context_size = context_size
        self.tokenizer_type = tokenizer_type
        download_dir = os.path.join(os.environ.get('IN_CONTEXT_DATA_PATH', '/bigdata/in-context'), 'datasets/wikipedia')

        if tokenizer_type == 'gpt2':
            self.tokenizer = transformers.AutoTokenizer.from_pretrained("gpt2", cache_dir=download_dir)
        elif tokenizer_type == 'byt5':
            self.tokenizer = transformers.AutoTokenizer.from_pretrained("google/byt5-base", cache_dir=download_dir)
        self.tokenizer.add_special_tokens({'pad_token': '[PAD]', 'cls_token': '[CLS]'})
        self.vocab_size = self.tokenizer.vocab_size

        all_tokenized_dataset = self._get_or_prepare_dataset(download_dir, context_size, max_size=max_size)
        self.tokenized_dataset = self._split_dataset(all_tokenized_dataset)[split]

    def __len__(self):
        return len(self.tokenized_dataset)

    def _get_or_prepare_dataset(self, download_dir, context_size, max_size=-1):
        if max_size == -1:
            max_size_str = 'size_all'
        else:
            max_size_str = f'size_{max_size}'
        prepared_dataset_path = os.path.join(
            os.environ.get('IN_CONTEXT_DATA_PATH', '/bigdata/in-context'),
            f'datasets/wikipedia_prepared_{self.tokenizer_type}_{context_size}_{max_size_str}'
        )
        if os.path.exists(prepared_dataset_path):
            logging.info(f'Loading prepared dataset from {prepared_dataset_path}')
            return datasets.load_from_disk(prepared_dataset_path)
        else:
            logging.info(f'Preparing dataset and saving to {prepared_dataset_path}')
            tokenized_dataset = self._prepare_dataset(download_dir, context_size)
            if max_size != -1:
                tokenized_dataset = tokenized_dataset.select(range(max_size))
            tokenized_dataset.save_to_disk(prepared_dataset_path)
            return tokenized_dataset

    def _prepare_dataset(self, download_dir, context_size):
        wiki = datasets.load_dataset(
            'wikipedia', '20220301.en',
            download_config=datasets.DownloadConfig(local_files_only=False, cache_dir=download_dir),
            split='train'
        )

        def tokenize_fn(element):
            outputs = self.tokenizer(
                element["text"],
                truncation=True,
                max_length=context_size,
                return_overflowing_tokens=True,
                return_length=True,
            )
            input_batch = []
            for length, input_ids in zip(outputs["length"], outputs["input_ids"]):
                if length == context_size:  # TODO: We may not want to discard these
                    input_batch.append(input_ids)
            return {"input_ids": input_batch}

        logging.info("Tokenizing dataset")
        tokenized_dataset = wiki.map(
            tokenize_fn,
            batched=True, remove_columns=wiki.column_names,
            num_proc=10
        )

        tokenized_dataset.set_format(type='torch', columns=['input_ids'])
        return tokenized_dataset

    def __getitem__(self, idx):
        x = self.tokenized_dataset[idx]['input_ids']
        y = x
        return x, y

    def _split_dataset(self, all_tokenized_dataset):
        def _split_dataset(dataset, second_split_size):
            train_test = dataset.train_test_split(test_size=second_split_size, shuffle=True, seed=42)
            return train_test['train'], train_test['test']

        val_size = self.VAL_SIZE
        test_size = self.TEST_SIZE
        train, testval = _split_dataset(all_tokenized_dataset, test_size + val_size)
        val, test = _split_dataset(testval, test_size)
        return {'train': train, 'val': val, 'test': test}


class RandomDataset(TaskDataset):
    E_BAG_SIZE = 500

    def __init__(self, n_dims, seed, n_classes, context_size, dataset_size, is_classification=True,
                 context_type='random'):
        self.dataset_size = dataset_size
        self.n_classes = n_classes
        self.context_size = context_size
        self.is_classification = is_classification
        self.context_type = context_type
        super().__init__(
            n_dims=n_dims, seed=seed, n_datapoints=context_size, dataset_size=dataset_size
        )

    def _construct_dataset(self):
        if self.context_type == 'fixed':
            E = self.rng.uniform(size=(1, self.context_size, self.n_dims))
            E = np.repeat(E, self.dataset_size, axis=0)
        elif self.context_type == 'random':
            E = self.rng.uniform(size=(self.dataset_size, self.context_size, self.n_dims))
        elif self.context_type == 'perm':
            E_bag = self.rng.uniform(size=(self.E_BAG_SIZE, self.n_dims))
            E = np.stack([
                E_bag[self.rng.permutation(self.E_BAG_SIZE)[:self.context_size]]
                for _ in range(self.dataset_size)
            ])
        else:
            raise ValueError(f'Unknown context type: {self.context_type}')
        e = self.rng.uniform(size=(self.dataset_size, self.n_dims))
        if self.is_classification:
            y = self.rng.multinomial(
                n=1, pvals=np.ones(self.n_classes) / self.n_classes, size=(self.dataset_size,)
            ).argmax(axis=-1)
            y = torch.tensor(y).long()
        else:
            y = self.rng.uniform(size=(self.dataset_size,))
            y = torch.tensor(y).float()
        X = np.concatenate([e[:, None, :], E], axis=1)
        X = torch.tensor(X).float()
        return X, y


class RandomWikiDataset(TaskDataset):
    MAX_CONTEXT_SIZE = 512
    MAX_DATASET_SIZE = 500_000

    def __init__(self, seed, n_classes, context_size, dataset_size, is_classification, context_type):
        self.dataset_size = dataset_size
        self.n_classes = n_classes
        self.context_size = context_size
        self.is_classification = is_classification
        self.context_type = context_type
        assert context_type in ['random', 'nxt_token']
        assert context_size <= 512, 'Context size must be <= 512'
        assert dataset_size <= 500_000, 'Dataset size must be <= 500_000'
        self.wiki_dataset = WikiDataset(
            context_size=512, tokenizer_type='byt5', max_size=self.MAX_DATASET_SIZE
        )  # Previously prepared
        super().__init__(
            n_dims=None, seed=seed, n_datapoints=context_size, dataset_size=dataset_size
        )

    def _construct_dataset(self):
        if self.context_type == 'random':
            return self._construct_random_dataset()
        elif self.context_type == 'nxt_token':
            return self._construct_nxt_token_dataset()
        else:
            raise ValueError(f'Unknown context type: {self.context_type}')

    def _construct_random_dataset(self):
        # Get a random permutation of wiki dataset
        idx = self.rng.permutation(len(self.wiki_dataset))
        X = []
        for i in range(self.dataset_size):
            context_ids = self.wiki_dataset[int(idx[i])][0][:self.context_size - 1]
            assert len(context_ids) == self.context_size - 1
            cls_token_id = self.wiki_dataset.tokenizer.cls_token_id
            X.append(np.array([cls_token_id] + context_ids.tolist()))
        X = np.stack(X)

        if self.is_classification:
            y = self.rng.multinomial(
                n=1, pvals=np.ones(self.n_classes) / self.n_classes, size=(self.dataset_size,)
            ).argmax(axis=-1)
            y = torch.tensor(y).long()
        else:
            y = self.rng.uniform(size=(self.dataset_size,))
            y = torch.tensor(y).float()
        X = torch.tensor(X)
        return X, y

    def _construct_nxt_token_dataset(self):
        idx = self.rng.permutation(len(self.wiki_dataset))
        y = []
        X = []
        for i in range(self.dataset_size):
            offset = self.rng.randint(10, 20)  # Random offset to avoid trivial next token prediction
            context_ids = self.wiki_dataset[int(idx[i])][0][-(self.context_size+1+offset):-offset]
            assert len(context_ids) == self.context_size + 1
            context = context_ids[:-1]
            nxt_token_id = context_ids[-1]
            X.append(np.array(context.tolist()))
            y.append(nxt_token_id)
        X = np.stack(X)
        y = np.stack(y)
        X = torch.tensor(X)
        y = torch.tensor(y).long()
        return X, y
