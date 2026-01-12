import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.generation.stopping_criteria import StoppingCriteriaList
from dataclasses import dataclass, field
from typing import List

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

class DoLa:
    def __init__(self, model_name, device):
        self.model_name = model_name
        self.device = device
        self.model, self.tokenizer = self.load_model(model_name)
        
    def load_model(self, model_name):
        if self.device == "cuda":
            kwargs = {'dtype': torch.float16, 'device_map': 'auto'}
        else:
            kwargs = {}
        model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        return model, tokenizer

    def set_stop_words(self, stop_words):
        self.stop_words = stop_words
        self.stopping_criteria = StoppingCriteriaList()
        for stop_word in self.stop_words:
            stop_word_ids = self.tokenizer.encode('\n' + stop_word)[3:]
            self.stopping_criteria.append(stop_word_ids)

    def generate(self, input_text, config: DoLaConfig):
        with torch.no_grad():
            input_ids = self.tokenizer(input_text, return_tensors="pt").input_ids.to(self.device)
            if config.mode == 'dola':
                outputs = self._dola_generate(input_ids, config)
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
        raise NotImplementedError("Baseline generation is not implemented")

    def _beam_dola_generate(self, input_ids, config: DoLaConfig):
        raise NotImplementedError("Beam DoLa generation is not implemented")

    def _beam_baseline_generate(self, input_ids, config: DoLaConfig):
        raise NotImplementedError("Beam baseline generation is not implemented")

    def _dola_generate(self, input_ids, config: DoLaConfig):
        """ CURRENTLY ONLY SUPPORT BATCH SIZE 1 """
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
            
            diff_logits = logsm_mature_logits.squeeze(0) - logsm_premature_logits[premature_layer_idx]
            
            next_token = self._sample_from_logits(diff_logits, config.temperature, config.top_p, config.top_k)
            input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=1)
            result.append(next_token.item())

        return result

    def _sample_from_logits(self, logits, temperature=1.0, top_p=0.0, top_k=0):
        logits = logits / temperature
        if top_p > 0.0:
            logits = self._top_p_filtering(logits, top_p)
        if top_k > 0:
            logits = self._top_k_filtering(logits, top_k)
        probs = F.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1)

    def _top_p_filtering(self, logits, top_p):
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        sorted_probs = F.softmax(sorted_logits, dim=-1)
        cumsum_probs = torch.cumsum(sorted_probs, dim=-1)

        sorted_indices_to_remove = cumsum_probs > top_p
        sorted_indices_to_remove[..., 0] = False

        indices_to_remove = torch.zeros_like(logits, dtype=torch.bool)
        indices_to_remove.scatter_(-1, sorted_indices, sorted_indices_to_remove)

        logits[indices_to_remove] = -float('inf')

        return logits

    def _top_k_filtering(self, logits, top_k):
        top_k = min(top_k, logits.size(-1))
        topk_logits, topk_indices = torch.topk(logits, top_k, dim=-1)
        
        indices_to_remove = torch.ones_like(logits, dtype=torch.bool)
        indices_to_remove.scatter_(-1, topk_indices, False)
        
        logits[indices_to_remove] = -float('inf')
        
        return logits

if __name__ == "__main__":
    model_name = "Qwen/Qwen3-0.6B"
    device = "cpu"

    config = DoLaConfig(
        mode="dola",
        candidate_premature_layers=[0, 8, 16, 24],
        mature_layer=-1,
        max_new_tokens=10,
        top_p=0.0,
        temperature=0.8
    )

    dola = DoLa(model_name, device)
    result = dola.generate("What is the highest peak in the world?", config)
    print(result)