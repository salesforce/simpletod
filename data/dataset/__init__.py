
from .multiwoz import MultiWozDataset


def get_dataset(args, split, **kwargs):
    if args.dataset == 'multiwoz':
        args.multiwoz_version = '2.1'
        dataset = MultiWozDataset(args, split=split, **kwargs)
        # if not args.no_history:
        #     input_output_word2index = dataset.input_word2index
        #     input_output_word2index.update(dataset.output_word2index)
        #     args.src_vocab_size = len(dataset.input_word2index)
        #     args.trg_vocab_size = len(dataset.output_word2index)
        # else:
        args.src_vocab_size = len(dataset.input_word2index)
        args.trg_vocab_size = len(dataset.output_word2index)
        args.src_pad_idx = dataset.input_word2index['_PAD']
        args.trg_pad_idx = dataset.output_word2index['_PAD']
        args.trg_bos_idx = dataset.output_word2index['_GO']
        args.trg_eos_idx = dataset.output_word2index['_EOS']

        args.knowledge_vocab_size = len(dataset.knowledge_word2index)
        args.action_vocab_size = len(dataset.action_word2index)

    return dataset, args