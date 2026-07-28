# ABOUTME: Driver: load DiscrimEval, select contrasts, generate calibrated paraphrases
# ABOUTME: Reference tokenizer = Llama-3.1-8B (token edit distance basis)

import argparse, asyncio, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from dotenv import load_dotenv
from transformers import AutoTokenizer
from openai import AsyncOpenAI
from discrimeval_data import load_discrimeval_explicit, select_samples
from discrimeval_paraphrase import generate_all_paraphrases


def main():
    load_dotenv()  # picks up OPENAI_API_KEY / HF auth from .env if present
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", default="results/discrimeval/paraphrases.json")
    ap.add_argument("--tokenizer", default="meta-llama/Llama-3.1-8B-Instruct")
    ap.add_argument("--sample_size", type=int, default=None)
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    tok = AutoTokenizer.from_pretrained(args.tokenizer)  # ambient HF auth
    samples = select_samples(load_discrimeval_explicit())
    if args.sample_size:
        samples = samples[:args.sample_size]
    client = AsyncOpenAI()  # reads OPENAI_API_KEY from env
    asyncio.run(generate_all_paraphrases(samples, args.output, tok, client))


if __name__ == "__main__":
    main()
