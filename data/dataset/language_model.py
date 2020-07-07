
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from transformers import PreTrainedTokenizer
import os
import logging
logger = logging.getLogger(__name__)


class LineByLineTextDataset(Dataset):
    def __init__(self, tokenizer, args, file_path, block_size=512):
        assert os.path.isfile(file_path)
        # Here, we do not cache the features, operating under the assumption
        # that we will soon use fast multithreaded tokenizers from the
        # `tokenizers` repo everywhere =)
        logger.info("Creating features from dataset file at %s", file_path)

        with open(file_path, encoding="utf-8") as f:
            lines = [line for line in f.read().splitlines() if (len(line) > 0 and not line.isspace())]

        self.examples = tokenizer.batch_encode_plus(lines, add_special_tokens=True, max_length=block_size)["input_ids"]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return torch.tensor(self.examples[i], dtype=torch.long)


class LineByLineTextDataset_shuffle_context(Dataset):
    def __init__(self, tokenizer, args, file_path, block_size=512):
        assert os.path.isfile(file_path)
        # Here, we do not cache the features, operating under the assumption
        # that we will soon use fast multithreaded tokenizers from the
        # `tokenizers` repo everywhere =)
        logger.info("Creating features from dataset file at %s", file_path)

        with open(file_path, encoding="utf-8") as f:
            lines = [line for line in f.read().splitlines() if (len(line) > 0 and not line.isspace())]
            tmp = [self.shuffle_context(line) for line in lines]
            lines = tmp

        self.examples = tokenizer.batch_encode_plus(lines, add_special_tokens=True, max_length=block_size)["input_ids"]

    def __len__(self):
        return len(self.examples)

    def shuffle_context(self, text):
        context, after_context = text.strip().split('<|context|>')[-1].split('<|endofcontext|>')
        tmp = context.split('<|user|>')
        # user_text = []
        # system_text = []
        all_text = []
        for txt in tmp:
            if txt == ' ':
                continue
            if '<|system|>' in txt:
                usr, system = txt.split('<|system|>')
                all_text.append('<|user|> {}'.format(usr))
                all_text.append('<|system|> {}'.format(system))
            else:
                usr = txt.strip()
                all_text.append('<|user|> {}'.format(usr))
        random.shuffle(all_text)
        shuffled_context = ' '.join(all_text)
        if after_context != '':
            new_text = '<|endoftext|> <|context|> {} <|endofcontext|> {}'.format(shuffled_context, after_context)
        else:
            new_text = '<|endoftext|> <|context|> {} <|endofcontext|>'.format(shuffled_context)

        return new_text


    def __getitem__(self, i):
        return torch.tensor(self.examples[i], dtype=torch.long)


class LineByLineTextDataset_shuffle_belief(Dataset):
    def __init__(self, tokenizer, args, file_path, block_size=512):
        assert os.path.isfile(file_path)
        # Here, we do not cache the features, operating under the assumption
        # that we will soon use fast multithreaded tokenizers from the
        # `tokenizers` repo everywhere =)
        logger.info("Creating features from dataset file at %s", file_path)

        with open(file_path, encoding="utf-8") as f:
            lines = [line for line in f.read().splitlines() if (len(line) > 0 and not line.isspace())]
            tmp = [self.shuffle_belief(line) for line in lines]
            lines = tmp

        self.examples = tokenizer.batch_encode_plus(lines, add_special_tokens=True, max_length=block_size)["input_ids"]

    def __len__(self):
        return len(self.examples)

    def shuffle_belief(self, text):
        before_belief = text.strip().split('<|belief|>')[0]
        belief, after_belief = text.strip().split('<|belief|>')[-1].split('<|endofbelief|>')
        tmp = belief.split(',')
        random.shuffle(tmp)
        new_belief = ','.join(tmp)
        new_text = '{} <|belief|> {} <|endofbelief|> {}'.format(before_belief, new_belief, after_belief)

        return new_text

    def shuffle_action(self, text):
        before_action = text.strip().split('<|action|>')[0]
        action, after_action = text.strip().split('<|action|>')[-1].split('<|endofaction|>')
        tmp = action.split(',')
        random.shuffle(tmp)
        new_action = ','.join(tmp)
        new_text = '{} <|action|> {} <|endofaction|> {}'.format(before_action, new_action, after_action)

        return new_text


    def __getitem__(self, i):
        return torch.tensor(self.examples[i], dtype=torch.long)


class LineByLineTextDataset_shuffle_belief_action(Dataset):
    def __init__(self, tokenizer, args, file_path, block_size=512):
        assert os.path.isfile(file_path)
        # Here, we do not cache the features, operating under the assumption
        # that we will soon use fast multithreaded tokenizers from the
        # `tokenizers` repo everywhere =)
        logger.info("Creating features from dataset file at %s", file_path)

        with open(file_path, encoding="utf-8") as f:
            lines = [line for line in f.read().splitlines() if (len(line) > 0 and not line.isspace())]
            tmp = [self.shuffle_belief(line) for line in lines]
            lines = tmp
            tmp = [self.shuffle_action(line) for line in lines]
            lines = tmp

        self.examples = tokenizer.batch_encode_plus(lines, add_special_tokens=True, max_length=block_size)["input_ids"]

    def __len__(self):
        return len(self.examples)

    def shuffle_belief(self, text):
        before_belief = text.strip().split('<|belief|>')[0]
        belief, after_belief = text.strip().split('<|belief|>')[-1].split('<|endofbelief|>')
        tmp = belief.split(',')
        random.shuffle(tmp)
        new_belief = ','.join(tmp)
        new_text = '{} <|belief|> {} <|endofbelief|> {}'.format(before_belief, new_belief, after_belief)

        return new_text

    def shuffle_action(self, text):
        before_action = text.strip().split('<|action|>')[0]
        action, after_action = text.strip().split('<|action|>')[-1].split('<|endofaction|>')
        tmp = action.split(',')
        random.shuffle(tmp)
        new_action = ','.join(tmp)
        new_text = '{} <|action|> {} <|endofaction|> {}'.format(before_action, new_action, after_action)

        return new_text


    def __getitem__(self, i):
        return torch.tensor(self.examples[i], dtype=torch.long)


def load_and_cache_examples(args, tokenizer, evaluate=False):
    file_path = args.eval_data_file if evaluate else args.train_data_file
    if not evaluate:
        if args.shuffle_context:
            return LineByLineTextDataset_shuffle_context(tokenizer, args, file_path=file_path, block_size=args.block_size)
        elif args.shuffle_belief_action:
            return LineByLineTextDataset_shuffle_belief_action(tokenizer, args, file_path=file_path,
                                                         block_size=args.block_size)
        elif args.shuffle_belief:
            return LineByLineTextDataset_shuffle_belief(tokenizer, args, file_path=file_path,
                                                         block_size=args.block_size)
        else:
            return LineByLineTextDataset(tokenizer, args, file_path=file_path, block_size=args.block_size)
    else:
        return LineByLineTextDataset(tokenizer, args, file_path=file_path, block_size=args.block_size)


def get_dataloader(dataset, tokenizer, args, split='train'):

    def collate(examples):
        if tokenizer._pad_token is None:
            return pad_sequence(examples, batch_first=True)
        return pad_sequence(examples, batch_first=True, padding_value=tokenizer.pad_token_id)

    if split == 'train':
        args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
        batch_size = args.train_batch_size
        sampler = RandomSampler(dataset) if args.local_rank == -1 else DistributedSampler(dataset)
    else:
        args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
        batch_size = args.eval_batch_size
        sampler = SequentialSampler(dataset)

    dataloader = DataLoader(dataset, sampler=sampler, batch_size=batch_size, collate_fn=collate)

    return dataloader, args
