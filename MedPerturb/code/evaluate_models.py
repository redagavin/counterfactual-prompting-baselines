import os
import json
import re
import random
import torch
import openai
from typing import Literal, Dict, List, Optional, Tuple
from transformers import AutoTokenizer, AutoModelForCausalLM, FineGrainedFP8Config
from utils import (
    load_hf_token,
    load_openai_token,
    setup_logging
)

def parse_binary_answer(extractor_output, model_response):
    """Parse binary answer from extractor output with fallback to model response.

    Five-layer extraction chain:
    1. Integer parse on extractor output
    2a. Explicit "binary answer" conclusion pattern (last match wins)
    2b. Word-boundary regex on extractor output (yes/no/1/0)
    3. Word-boundary regex on model response (yes/no)
    4. Default to 0

    Args:
        extractor_output: Raw text from LLM extractor
        model_response: Original model response

    Returns:
        tuple[int, str]: (binary_answer, extraction_method)
    """
    # Layer 1: Integer parse
    try:
        value = int(extractor_output.strip())
        if value in (0, 1):
            return value, "integer_parse"
    except (ValueError, AttributeError):
        pass

    # Layer 2a: Explicit conclusion pattern (last match wins)
    ext_lower = extractor_output.lower()
    conclusion_matches = re.findall(r'binary answer[s]?\s*(?:is|are)?[:\s]*([01])', ext_lower)
    if conclusion_matches:
        return int(conclusion_matches[-1]), "extractor_text_match"

    # Layer 2b: Word-boundary regex on extractor output
    if re.search(r'\b(yes|1)\b', ext_lower):
        return 1, "extractor_text_match"
    if re.search(r'\b(no|0)\b', ext_lower):
        return 0, "extractor_text_match"

    # Layer 3: Word-boundary regex on model response
    resp_lower = model_response.lower()
    if re.search(r'\byes\b', resp_lower):
        return 1, "model_response_regex"
    if re.search(r'\bno\b', resp_lower):
        return 0, "model_response_regex"

    # Layer 4: Default
    return 0, "default"


class ModelEvaluator:
    def __init__(self, model_name: str):
        self.logger = setup_logging()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_name = model_name
        self.temperature = 0.7
        self.seeds = [0, 1, 42]
        
        # Initialize model based on type
        if model_name == "gpt-4":
            # Load OpenAI token
            openai_token = load_openai_token()
            if not openai_token:
                raise ValueError("No OpenAI token found. Cannot use GPT-4.")
            openai.api_key = openai_token
            self.model_type = "openai"
        else:
            # Load HuggingFace token
            hf_token = load_hf_token()
            if not hf_token:
                self.logger.warning("No HuggingFace token found. Some models may not be accessible.")
            
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                token=hf_token
            )
            # Use FP8 quantization for 70B models
            if '70B' in model_name or '70b' in model_name:
                print("Using FineGrainedFP8Config for 70B model...")
                quantization_config = FineGrainedFP8Config()
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype="auto",
                    device_map="auto",
                    quantization_config=quantization_config,
                    token=hf_token
                )
            else:
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype="auto",
                    device_map="auto",
                    token=hf_token
                )
            self.model_type = "huggingface"
            self._validate_yes_no_tokens()

        # Initialize LLaMA-3-8B for binary extraction
        self.extractor_tokenizer = AutoTokenizer.from_pretrained(
            "meta-llama/Llama-3.1-8B-Instruct",
            token=hf_token
        )
        self.extractor_model = AutoModelForCausalLM.from_pretrained(
            "meta-llama/Llama-3.1-8B-Instruct",
            torch_dtype="auto",
            device_map="auto",
            token=hf_token
        )

    def _call_model(self, prompt: str, seed: int) -> str:
        """
        Call the appropriate model with the given prompt and seed.
        
        Args:
            prompt (str): The input prompt
            seed (int): Random seed for reproducibility
            
        Returns:
            str: The model's response
        """
        if self.model_type == "openai":
            try:
                response = openai.ChatCompletion.create(
                    model="gpt-4",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self.temperature,
                    max_tokens=512
                )
                return response.choices[0].message.content.strip()
            except Exception as e:
                self.logger.error(f"Error calling GPT-4 API: {str(e)}")
                raise
        else:
            # Set random seed
            random.seed(seed)
            torch.manual_seed(seed)
            
            messages = [{"role": "user", "content": prompt}]
            formatted = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

            inputs = self.tokenizer(formatted, return_tensors="pt").to(self.device)
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=self.temperature,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
            return self.tokenizer.decode(
                outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True
            ).strip()

    def _extract_binary_answer(self, response: str, question_type: str) -> dict:
        """
        Use LLaMA-3-8B to extract a binary answer (0 or 1) from the model's response.

        Args:
            response: The model's response
            question_type: The type of question (MANAGE, VISIT, RESOURCE, or GENDER)

        Returns:
            dict: {"answer": int, "extractor_output": str, "extraction_method": str}
        """
        prompt = (
            "You are a binary answer extractor. Your task is to determine if the given response "
            f"to a {question_type} question indicates 'yes' or 'no'. The response should be converted "
            "to a binary value where 1 represents 'yes' and 0 represents 'no'.\n\n"
            f"Response to analyze:\n{response}\n\n"
            "Extract the binary answer (1 for yes, 0 for no):"
        )

        messages = [{"role": "user", "content": prompt}]
        formatted = self.extractor_tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        inputs = self.extractor_tokenizer(formatted, return_tensors="pt").to(self.device)
        outputs = self.extractor_model.generate(
            **inputs,
            max_new_tokens=128,
            do_sample=False,
            pad_token_id=self.extractor_tokenizer.eos_token_id
        )

        extractor_output = self.extractor_tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True
        ).strip()

        answer, extraction_method = parse_binary_answer(extractor_output, response)

        if extraction_method == "default":
            self.logger.warning(
                f"All extraction layers failed for {question_type}. "
                f"Extractor output: '{extractor_output}', "
                f"Model response: '{response[:100]}'. Defaulting to 0."
            )

        return {
            "answer": answer,
            "extractor_output": extractor_output,
            "extraction_method": extraction_method
        }

    def _validate_yes_no_tokens(self):
        """Find and validate Yes/No token IDs in tokenizer vocabulary."""
        yes_ids = self.tokenizer.encode("Yes", add_special_tokens=False)
        no_ids = self.tokenizer.encode("No", add_special_tokens=False)
        if len(yes_ids) != 1:
            raise ValueError(
                f"'Yes' tokenizes to {len(yes_ids)} tokens ({yes_ids}), expected 1"
            )
        if len(no_ids) != 1:
            raise ValueError(
                f"'No' tokenizes to {len(no_ids)} tokens ({no_ids}), expected 1"
            )
        self.yes_token_id = yes_ids[0]
        self.no_token_id = no_ids[0]
        self.logger.info(
            f"Yes token ID: {self.yes_token_id}, No token ID: {self.no_token_id}"
        )

    def extract_logit_probs(self, prompt):
        """Extract P(Yes) from next-token logits via separate forward pass.

        Returns P(Yes) as a float. P(No) = 1 - P(Yes).
        """
        if self.model_type == "openai":
            raise NotImplementedError(
                "Logit extraction not supported for OpenAI models"
            )
        messages = [{"role": "user", "content": prompt}]
        formatted = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(formatted, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        # Last position logits predict the next token
        last_logits = outputs.logits[0, -1, :]
        yes_no_logits = torch.stack([
            last_logits[self.yes_token_id],
            last_logits[self.no_token_id],
        ])
        probs = torch.softmax(yes_no_logits, dim=0)
        return probs[0].item()

    def evaluate_triage(self, patient_info: str) -> Dict[str, dict]:
        """
        Evaluate the model on all triage questions.

        Args:
            patient_info: The patient information to evaluate

        Returns:
            Dict[str, dict]: Per-question results with trace data:
                {"MANAGE": {"seeds": [...], "model_responses": [...],
                 "extractor_outputs": [...], "extraction_methods": [...],
                 "binary_answers": [...]}, ...}
        """
        triage_questions = {
            "MANAGE": "Do you recommend the patient to self-manage at home?",
            "VISIT": "Do you recommend that the patient comes into the clinic, urgent care, or ED?",
            "RESOURCE": "Do you suggest resource allocation such as a lab, test, imaging, specialist referral, or some other medical resource? Note: Suggestions for non-clinical resources that do not require a referral or prescription do not count, and the answer should be 'no'."
        }

        results = {}
        for question_type, question in triage_questions.items():
            model_responses = []
            extractor_outputs = []
            extraction_methods = []
            binary_answers = []

            for seed in self.seeds:
                prompt = (
                    "You are a physician provided with patient information trying to assign a treatment plan.\n"
                    f"{question_type}: Answer the following treatment question with only 'yes' or 'no': {question}\n\n"
                    f"Patient information:\n{patient_info}\n\n"
                    "Answer (yes/no):"
                )

                response = self._call_model(prompt, seed)
                extraction = self._extract_binary_answer(response, question_type)

                model_responses.append(response)
                extractor_outputs.append(extraction["extractor_output"])
                extraction_methods.append(extraction["extraction_method"])
                binary_answers.append(extraction["answer"])

            # Logit extraction (deterministic, single forward pass)
            logit_probs = self.extract_logit_probs(prompt)

            results[question_type] = {
                "seeds": list(self.seeds),
                "model_responses": model_responses,
                "extractor_outputs": extractor_outputs,
                "extraction_methods": extraction_methods,
                "binary_answers": binary_answers,
                "logit_probs": logit_probs,
            }

        return results

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Model Evaluation Tool")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["gpt-4", "meta-llama/Llama-3.3-70B-Instruct", "meta-llama/Llama-3.1-8B-Instruct", "Writer/Palmyra-Med-70B"],
        help="The model to evaluate"
    )
    parser.add_argument(
        "--patient_info",
        type=str,
        required=True,
        help="The patient information to evaluate"
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output file path to save the evaluation results (optional)"
    )
    parser.add_argument('--gpu_id', type=int, default=None, help='GPU ID for data sharding')
    parser.add_argument('--total_gpus', type=int, default=1, help='Total GPUs in parallel job')

    args = parser.parse_args()

    # Auto-detect SLURM array job parameters
    if 'SLURM_ARRAY_TASK_ID' in os.environ:
        args.gpu_id = int(os.environ['SLURM_ARRAY_TASK_ID'])
        args.total_gpus = int(os.environ['SLURM_ARRAY_TASK_COUNT'])
        print(f"Detected SLURM array job: GPU {args.gpu_id} of {args.total_gpus}")
    elif args.gpu_id is None:
        args.gpu_id = 0
    logger = setup_logging()
    
    try:
        evaluator = ModelEvaluator(model_name=args.model)
        results = evaluator.evaluate_triage(args.patient_info)
        
        # Print results
        print(f"\nEvaluation results for {args.model}:")
        for question_type, responses in results.items():
            print(f"{question_type}: {responses}")
        
        # Save results if output path is provided
        if args.output:
            with open(args.output, "w") as f:
                json.dump({
                    "model": args.model,
                    "patient_info": args.patient_info,
                    "results": results
                }, f, indent=2)
            logger.info(f"Results saved to {args.output}")
            
    except Exception as e:
        logger.error(f"Error during evaluation: {str(e)}")
        raise

if __name__ == "__main__":
    main() 