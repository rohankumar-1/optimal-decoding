from decoders.base import BaseDecoder
from configs import GenerationConfig
import torch
import torch.nn.functional as F


class BaselineDecoder(BaseDecoder):
    """Baseline decoder."""
    
    def __init__(self, model_name, device):
        super().__init__(model_name, device)
    

    def generate(self, input_text: str, gen_config: GenerationConfig) -> str:
        input_ids = self.tokenizer(input_text, return_tensors="pt").input_ids.to(self.device)
        outputs = self._baseline_generate(input_ids, gen_config)
        return "".join([self.tokenizer.decode(output, skip_special_tokens=True) for output in outputs])


    def _baseline_generate(self, input_ids, gen_config: GenerationConfig):
        with torch.no_grad():
            result = []
            for step in range(gen_config.max_new_tokens):
                output = self.model(input_ids=input_ids)
                logits = output.logits[:, -1, :]

                logits = logits / gen_config.temperature
                if gen_config.top_p > 0.0:
                    logits = self._top_p_filtering(logits, gen_config.top_p)
                if gen_config.top_k > 0:
                    logits = self._top_k_filtering(logits, gen_config.top_k)

                sm_logits = F.softmax(logits, dim=-1)
                if gen_config.deterministic:
                    next_token = torch.argmax(sm_logits, dim=-1).view(1,1)
                else:
                    next_token = torch.multinomial(sm_logits, num_samples=1).view(1,1)

                if self.stopping_criteria and next_token.item() in self.stopping_criteria:
                    print(f"Stopping criteria met at step {step}")
                    break

                input_ids = torch.cat([input_ids, next_token], dim=1)
                result.append(next_token.item())
            return result


    def lm_score(self, context: str, target: str, gen_config: GenerationConfig) -> float:
        with torch.no_grad():
            input_text = context + target
            input_ids = self.tokenizer(input_text, return_tensors="pt").input_ids.to(self.device)
            context_ids = self.tokenizer(context, return_tensors="pt").input_ids.to(self.device)
            target_ids = input_ids[0, context_ids.shape[-1]:] 

            # run model, grab logits for the target tokens
            outputs = self.model(input_ids=input_ids).logits[0, context_ids.shape[-1] - 1: -1, :]
            sm_outputs = self._adjust_logits(
                outputs, 
                deterministic=gen_config.deterministic,
                temperature=gen_config.temperature,
                top_p=gen_config.top_p, 
                top_k=gen_config.top_k
            )
            logsm_outputs = sm_outputs.log()

            # grab logprobs for the target tokens
            logprob = logsm_outputs[range(logsm_outputs.shape[0]), target_ids].sum().item()
        return logprob