import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.generation.stopping_criteria import StoppingCriteriaList
from dataclasses import dataclass, field
from typing import List
import numpy as np

@dataclass
class DoLaConfig:
    mode: str = 'dola'
    candidate_premature_layers: List[int] = field(default_factory=lambda: [])
    mature_layer: int = -1
    top_p: float = 0.95
    top_k: int = 0
    temperature: float = 0.8
    post_softmax: bool = True
    max_new_tokens: int = 256
    relative_top: float = 0.1
    deterministic: bool = False


class DoLa:
    def __init__(self, model_name, device):
        self.model_name = model_name
        self.device = device
        self.model, self.tokenizer = self.load_model(model_name)
        self.stopping_criteria = None
        
    def load_model(self, model_name):
        if self.device == "cuda":
            kwargs = {'dtype': torch.float16, 'device_map': 'auto'}
        else:
            kwargs = {}
        model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        return model, tokenizer

    def set_stop_words(self, stop_words):
        self.stopping_criteria = StoppingCriteriaList()
        for stop_word in stop_words:
            stop_word_ids = self.tokenizer.encode('\n' + stop_word)[3:]
            self.stopping_criteria.append(stop_word_ids)

    def generate(self, input_text, config: DoLaConfig):
        input_ids = self.tokenizer(input_text, return_tensors="pt").input_ids.to(self.device)
        if config.mode == 'dola':
            outputs = self._dola_generate(input_ids, config)
        elif config.mode == 'adaptive-dola':
            outputs = self._adaptive_dola_generate(input_ids, config)
        elif config.mode == 'baseline':
            outputs = self._baseline_generate(input_ids, config)
        elif config.mode == 'beam-dola':
            outputs = self._beam_dola_generate(input_ids, config)
        elif config.mode == 'beam-baseline':
            outputs = self._beam_baseline_generate(input_ids, config)
        else:
            raise ValueError(f"Invalid mode: {config.mode}")
        return "".join([self.tokenizer.decode(output, skip_special_tokens=True) for output in outputs])
                
    def _baseline_generate(self, input_ids, config: DoLaConfig):
        """ this should mimic model.generate() behavior """
        with torch.no_grad():
            result = []
            for step in range(config.max_new_tokens):
                output = self.model(input_ids=input_ids)
                logits = output.logits[:, -1, :]
                next_token = self._sample_from_logits(logits, config.deterministic, config.temperature, config.top_p, config.top_k)
                result.append(next_token.item())
                input_ids = torch.cat([input_ids, next_token], dim=1)
                if self.stopping_criteria and next_token.item() in self.stopping_criteria:
                    print(f"Stopping criteria met at step {step}")
                    break
            return result

    def _beam_dola_generate(self, input_ids, config: DoLaConfig):
        raise NotImplementedError("Beam DoLa generation is not implemented")

    def _beam_baseline_generate(self, input_ids, config: DoLaConfig):
        raise NotImplementedError("Beam baseline generation is not implemented")

    def _dola_generate(self, input_ids, config: DoLaConfig):
        """ CURRENTLY ONLY SUPPORT BATCH SIZE 1 """
        with torch.no_grad():
            result = []
            for step in range(config.max_new_tokens):
                hidden_states = self.model(
                    input_ids=input_ids,
                    return_dict=True,
                    output_attentions=False,
                    output_hidden_states=True,
                ).get('hidden_states', None)

                mature_hidden_state = hidden_states[config.mature_layer][0, -1, :]
                mature_logits = self.model.lm_head(mature_hidden_state.unsqueeze(0)).squeeze(0)
                
                premature_hidden_states = torch.stack(
                    [hidden_states[layer][0, -1, :] for layer in config.candidate_premature_layers],
                    dim=0
                )
                premature_logits = self.model.lm_head(premature_hidden_states)

                sm_mature_logits = F.softmax(mature_logits, dim=-1).unsqueeze(0)
                sm_premature_logits = F.softmax(premature_logits, dim=-1)
                M = 0.5 * (sm_mature_logits + sm_premature_logits)

                logsm_mature_logits = F.log_softmax(mature_logits, dim=-1).unsqueeze(0)
                logsm_premature_logits = F.log_softmax(premature_logits, dim=-1)

                kl1 = F.kl_div(logsm_mature_logits, M, reduction='none').mean(-1)
                kl2 = F.kl_div(logsm_premature_logits, M, reduction='none').mean(-1)

                js_divs = 0.5 * (kl1 + kl2)
                premature_layer_idx = int(js_divs.argmax().cpu().item())
            
                base_logits = premature_logits[premature_layer_idx]
                final_logits = mature_logits
                if config.relative_top > 0.0:
                    final_logits = self._relative_top_filter(final_logits, config.relative_top)
                    base_logits = base_logits.log_softmax(dim=-1)
                    mask = final_logits < -1e3
                    base_logits[mask] = -1e3
                logits = final_logits - base_logits
                next_token_logits = logits
                
                next_token = self._sample_from_logits(next_token_logits, config.temperature, config.top_p, config.top_k)

                # check if the next token is a stopping criteria
                if self.stopping_criteria and next_token.item() in self.stopping_criteria:
                    print(f"Stopping criteria met at step {step}")
                    break

                input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=1)
                result.append(next_token.item())

        return result


    def _adaptive_dola_generate(self, input_ids, config: DoLaConfig):
        raise NotImplementedError("Adaptive DoLa generation is not implemented")

    

    def _relative_top_filter(self, scores, relative_top=0.1, filter_value=-float("Inf"), min_tokens_to_keep=1):
        scores_normalized = scores.log_softmax(dim=-1) 
        sorted_logits, sorted_indices = torch.sort(scores_normalized, descending=True)
        min_thresh = sorted_logits[..., min_tokens_to_keep-1] 
        probs_max = torch.max(scores_normalized, dim=-1).values
        probs_thresh = probs_max + np.log(relative_top)
        probs_thresh = torch.min(min_thresh, probs_thresh)
        probs_thresh = probs_thresh.unsqueeze(-1)
        scores_normalized[scores_normalized < probs_thresh] = filter_value
        return scores_normalized




if __name__ == "__main__":
    model_name = "Qwen/Qwen3-0.6B"
    device = "cpu"

    config = DoLaConfig(
        mode="baseline",
        candidate_premature_layers=[0, 8, 16, 24],
        mature_layer=-1,
        max_new_tokens=50,
        top_p=0.0,
        temperature=0.9
    )

    dola = DoLa(model_name, device)
    result = dola.generate("What is the highest peak in the world?", config)
    print(result)