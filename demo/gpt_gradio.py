import torch
import hydra
import gradio as gr
from omegaconf import DictConfig
import tiktoken
import logging
from typing import Tuple, Dict
import boto3

log = logging.getLogger(__name__)


def demo(cfg: DictConfig) -> Tuple[dict, dict]:

    assert cfg.bucket_name
    assert cfg.object_name
    assert cfg.file_name

    s3 = boto3.client('s3')

    log.info("Running Demo")

    s3.download_file(cfg.bucket_name, cfg.object_name, cfg.file_name)

    log.info(f"Instantiating scripted model <{cfg.file_name}>")
    model = torch.jit.load(cfg.file_name)

    log.info(f"Loaded Model: {model}")

    cl100k_base = tiktoken.get_encoding("cl100k_base")

    encoder = tiktoken.Encoding(
            name="cl100k_im",
            pat_str=cl100k_base._pat_str,
            mergeable_ranks=cl100k_base._mergeable_ranks,
            special_tokens={
                **cl100k_base._special_tokens,
                "<|im_start|>": 100264,
                "<|im_end|>": 100265,
            },
        )

    def generate(prompt: str, token_len: int=256) -> str:
        if prompt is None:
            return None

        prompt = torch.tensor(encoder.encode(prompt)).unsqueeze(0).long()
        with torch.no_grad():
            generated = model.model.generate(prompt, max_new_tokens=token_len)
        generated = encoder.decode(generated[0].cpu().numpy().tolist())
        return generated

    demo = gr.Interface(
        fn=generate,
        inputs=[gr.inputs.Textbox(lines=5, label="Prompt"),gr.Slider(2, 512, value=256, label="Output Token Length", info="Range: 2 and 512")],
        outputs=gr.outputs.Textbox(label="Generated Text"),
        title="GPT Gradio Demo",
        description="Generate text from a prompt using a custom trained GPT model.",
    )

    demo.launch(server_name= "0.0.0.0", server_port=80)

@hydra.main(version_base="1.3", config_path="../configs", config_name="demo.yaml")
def main(cfg: DictConfig) -> None:
    demo(cfg)

if __name__ == "__main__":
    main()