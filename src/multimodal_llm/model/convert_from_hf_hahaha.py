"""
A reverse of `converse_weights_to_hf.py`, i.e., convert HF model to lit-llama format.
"""
import os
import argparse
import logging

import tqdm
import torch
import einops  # necessary for conversion  # noqa: F401
from transformers import LlamaForCausalLM


try:
    import accelerate  # necessary for conversion  # noqa: F401
except ImportError:
    raise ValueError("`accelerate` is necessary for `low_cpu_mem_usage` conversion.")

logger = logging.getLogger(__name__)


def _permute_hf_to_litllama(w, n_embd, n_head, dtype):
    dim = n_embd
    w = w.type(dtype)
    return w.view(n_head, 2, dim // n_head // 2, dim).transpose(1, 2).reshape(dim, dim)


def _main_wrapper(
    # model_size,
    hf_model_path,
    torch_ckpt_path,
    dtype="float32",
    # verify=False,
):
    main_convert(hf_model_path, torch_ckpt_path, dtype)


def main_convert(
    # model_size,
    hf_model_path,
    torch_ckpt_path,
    dtype="float32",
    # verify=False,
):
    """
    Convert a HF llama model to lit-llama torch model, and save it to model_path.

    Parameters
    ----------
    model_size : str

    hf_model_path : str
        Path to load the HF llama model from.
    torch_ckpt_path : str
        Path to save a torch llama checkpoint to.
    dtype : str
        Data type to use for the model when saving. Default: "float32".
    verify : bool
        Whether to verify the conversion by loading the model and running a forward pass.

    """
    if not os.path.exists(torch_ckpt_path):
        raise ValueError(f"output torch_ckpt_path {torch_ckpt_path} does not exist")

    dt = getattr(torch, dtype, None)
    if not isinstance(dt, torch.dtype):
        raise ValueError(f"{dtype} is not a valid dtype.")
    dtype = dt


    # model
    # if verify:
    #     hf_model = LlamaForCausalLM.from_pretrained(
    #         hf_model_path, torch_dtype=dtype, low_cpu_mem_usage=True, device_map="cpu"
    #     )
    # else:
    with torch.inference_mode():  # save memory if possible.
        hf_model = LlamaForCausalLM.from_pretrained(
            hf_model_path, torch_dtype=dtype, low_cpu_mem_usage=True, device_map="cpu"
        )

    # config = LitLLaMAConfig.from_name(model_size)
    # torch_model = LLaMA(config)

    # n_layers = config.n_layer
    n_layers = 22
    n_head = 32
    n_embd = 2048
    intermediate_size = 5632

    # convert layers

    loaded = hf_model.state_dict()  # from this,
    new_state_dict = {}  # to this.
    # convert all the transformer layers
    for lyr_i in tqdm.tqdm(range(n_layers)):
        _q = _permute_hf_to_litllama(
            loaded[f"model.layers.{lyr_i}.self_attn.q_proj.weight"],
            n_embd,
            n_head,
            dtype,
        )
        _k = _permute_hf_to_litllama(
            loaded[f"model.layers.{lyr_i}.self_attn.k_proj.weight"],
            n_embd,
            n_head,
            dtype,
        )
        _v = loaded[f"model.layers.{lyr_i}.self_attn.v_proj.weight"]

        # fmt: off
        new_state_dict.update(
            {
                f"transformer.h.{lyr_i}.attn.attn.weight": torch.cat([_q, _k, _v], dim=0),
                f"transformer.h.{lyr_i}.attn.proj.weight": loaded[f"model.layers.{lyr_i}.self_attn.o_proj.weight"],
                f"transformer.h.{lyr_i}.mlp.swiglu.w1.weight": loaded[f"model.layers.{lyr_i}.mlp.gate_proj.weight"],
                f"transformer.h.{lyr_i}.mlp.swiglu.w2.weight": loaded[f"model.layers.{lyr_i}.mlp.up_proj.weight"],
                f"transformer.h.{lyr_i}.mlp.swiglu.w3.weight": loaded[f"model.layers.{lyr_i}.mlp.down_proj.weight"],
                f"transformer.h.{lyr_i}.norm_1.weight": loaded[f"model.layers.{lyr_i}.input_layernorm.weight"],
                f"transformer.h.{lyr_i}.norm_2.weight": loaded[f"model.layers.{lyr_i}.post_attention_layernorm.weight"],
            }
        )
        # fmt: on
    # convert embeddings
    new_state_dict.update(
        {
            "transformer.wte.weight": loaded["model.embed_tokens.weight"],
            "transformer.ln_f.scale": loaded["model.norm.weight"],
            "lm_head.weight": loaded["lm_head.weight"],
        }
    )
    logger.info("Conversion: Done. Loading the new state_dict to the torch model..")
    # torch_model.load_state_dict(new_state_dict, strict=True)
    #
    # if verify:
    #     logger.info("Verification: Comparing output tokens of litllama and huggingface.")
    #     torch.use_deterministic_algorithms(True)
    #     # generate a sample for testing
    #     torch.manual_seed(132)
    #     token_sample = torch.randint(
    #         0, config.vocab_size, size=(1, config.block_size), dtype=torch.int64
    #     )
    #     litllama_out = torch_model(token_sample)
    #
    #     # load a float32 model for testing on CPU.
    #     hf_out = hf_model(token_sample)["logits"]
    #     hf_tokens = torch.argmax(hf_out, dim=-1)
    #     litllama_tokens = torch.argmax(litllama_out, dim=-1)
    #     torch.testing.assert_close(
    #         hf_tokens, litllama_tokens
    #     ), "The outputs of the models are not the same."
    #     logger.info("Verification: Done. ")
    #
    # logger.info("Saving...")
    # torch.save(
    #     torch_model.state_dict(),
    #     os.path.join(torch_ckpt_path, f"{model_size}-torch.state_dict.ckpt"),
    # )
    logger.info(f"Saved at {torch_ckpt_path}. Mahalo!")


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    # args.add_argument("--model_size", type=str, default="7B")

    args.add_argument(
        "--hf_model_path", type=str, required=True, help="Path to the converted HF model, or model name."
    )
    args.add_argument(
        "--torch_ckpt_path",
        type=str,
        default=None,
        help="Path to torch model state_dict file (.pt). To be used by Prescient-LM users.",
    )
    # args.add_argument(
    #     "--verify", action="store_true", help="Verify the conversion by comparing the outputs."
    # )

    args = args.parse_args()

    # argument verification
    _main_wrapper(
        args.hf_model_path,
        args.torch_ckpt_path,
        dtype="float32",
        # verify=args.verify,
    )
