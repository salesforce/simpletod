
from .dataset.multiwoz import MultiWozDataset

# def get_dataset(args):
#     if args.dataset == 'multiwoz':
#         args.multiwoz_version = '2.1'
#         dataset = MultiWozDataset(args)
#         args.src_vocab_size = len(dataset.input_word2index)
#         args.trg_vocab_size = len(dataset.output_word2index)
#         args.src_pad_idx = dataset.input_word2index['_PAD']
#         args.trg_pad_idx = dataset.output_word2index['_PAD']
#     else:
#         raise TypeError('unknown dataset')
#
#     return dataset, args
