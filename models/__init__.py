
# from .transformer.Transformer import Transformer
#
#
# def get_model(args, dataset):
#
#     if args.model_type == 'transformer':
#         model = Transformer(
#             args,
#             args.src_vocab_size,
#             args.trg_vocab_size,
#             args.knowledge_vocab_size,
#             args.action_vocab_size,
#             src_pad_idx=args.src_pad_idx,
#             trg_pad_idx=args.trg_pad_idx,
#             trg_bos_idx=args.trg_bos_idx,
#             trg_eos_idx=args.trg_eos_idx,
#             input_index2word=dataset.input_index2word,
#             output_index2word=dataset.output_index2word,
#             knowledge_index2word=dataset.knowledge_index2word,
#             knowledge_index2index_output=dataset.knowledge_index2index_output,
#             action_index2index_output=dataset.action_index2index_output,
#             trg_emb_prj_weight_sharing=args.proj_share_weight,
#             emb_src_trg_weight_sharing=args.embs_share_weight,
#             d_k=args.d_k,
#             d_v=args.d_v,
#             d_model=args.d_model,
#             d_word_vec=args.d_word_vec,
#             d_inner=args.d_inner_hid,
#             n_layers=args.n_layers,
#             n_head=args.n_head,
#             dropout=args.dropout,
#             n_position=args.seq_len,
#             decoding=args.decoding,
#             beam_size=args.beam_size).to(args.device)
#         model.logger = getattr(args, 'logger')
#     else:
#         raise TypeError('should specify model type')
#
#     return model
__version__ = "2.5.1"
from .configuration_gpt2 import GPT2_PRETRAINED_CONFIG_ARCHIVE_MAP, GPT2Config, GPT2SmallConfig
from .configuration_utils import PretrainedConfig
from .file_utils import (
    CONFIG_NAME,
    MODEL_CARD_NAME,
    PYTORCH_PRETRAINED_BERT_CACHE,
    PYTORCH_TRANSFORMERS_CACHE,
    TF2_WEIGHTS_NAME,
    TF_WEIGHTS_NAME,
    TRANSFORMERS_CACHE,
    WEIGHTS_NAME,
    add_end_docstrings,
    add_start_docstrings,
    cached_path,
    is_tf_available,
    is_torch_available,
)

from .modeling_utils import PreTrainedModel, prune_layer, Conv1D, top_k_top_p_filtering
from .modeling_gpt2 import (
        GPT2PreTrainedModel,
        GPT2Model,
        GPT2LMHeadModel,
        GPT2DoubleHeadsModel,
        load_tf_weights_in_gpt2,
        GPT2_PRETRAINED_MODEL_ARCHIVE_MAP,
    )