from decoders.base import BaseDecoder
import torch
import torch.nn.functional as F
from configs import DoLaConfig, GenerationConfig
import numpy as np

class DoLaDecoder(BaseDecoder):
    """DoLa decoding with dynamic layer selection."""
    
    
    def __init__(self, model_name, device, config: DoLaConfig):
        super().__init__(model_name, device)
        self.config = config
    

    def generate(self, input_text: str, gen_config: GenerationConfig) -> str:
        input_ids = self.tokenizer(input_text, return_tensors="pt").input_ids.to(self.device)
        outputs = self._dola_generate(input_ids, gen_config)
        return "".join([self.tokenizer.decode(output, skip_special_tokens=True) for output in outputs])
    

    def lm_score(self, context:str, target:str, gen_config: GenerationConfig) -> float:
        with torch.no_grad():
            input_text = context + target
            input_ids = self.tokenizer(input_text, return_tensors="pt").input_ids.to(self.device)
            context_ids = self.tokenizer(context, return_tensors="pt").input_ids.to(self.device)
            target_ids = input_ids[0, context_ids.shape[-1]:] 

            hidden_states = self.model(
                input_ids=input_ids,
                return_dict=True,
                output_attentions=False,
                output_hidden_states=True
            ).get('hidden_states', None)

            mature_hidden_state = hidden_states[self.config.mature_layer][0, context_ids.shape[-1] - 1: -1, :] # get mature layers for all tokens
            mature_logits = self.model.lm_head(mature_hidden_state.unsqueeze(0)).squeeze(0)

            premature_hidden_states = torch.stack([hidden_states[self.config.candidate_premature_layers][0, context_ids.shape[-1] - 1: -1, :]])
            premature_logits = self.model.lm_head(premature_hidden_states)

            sm_mature_logits = F.softmax(mature_logits, dim=-1).unsqueeze(0)
            sm_premature_logits = F.softmax(premature_logits, dim=-1)
            M = 0.5 * (sm_mature_logits + sm_premature_logits)

            logsm_mature_logits = F.log_softmax(mature_logits, dim=-1).unsqueeze(0)
            logsm_premature_logits = F.log_softmax(premature_logits, dim=-1)

            outputs = outputs[context_ids.shape[-1] - 1: -1, :]

            # grab logprobs for the target tokens
            logprob = outputs[range(outputs.shape[0]), target_ids].sum().item()

        return logprob
    

    def _dola_generate(self, input_ids, gen_config: GenerationConfig):
        with torch.no_grad():
            result = []
            for step in range(gen_config.max_new_tokens):
                hidden_states = self.model(
                    input_ids=input_ids,
                    return_dict=True,
                    output_attentions=False,
                    output_hidden_states=True,
                ).get('hidden_states', None)

                # hidden states has structure: (batch, sequence idx, actual values)

                mature_hidden_state = hidden_states[self.config.mature_layer][0, -1, :] # assume batch_sz=1, get last sequence idx (most recent token)
                mature_logits = self.model.lm_head(mature_hidden_state.unsqueeze(0)).squeeze(0) # unsqueeze to add batch back in, then squeeze after lm_head
                
                premature_hidden_states = torch.stack([hidden_states[layer][0, -1, :] for layer in self.config.candidate_premature_layers], dim=0) # ends up being (batch = # premature layers, hidden)
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
                if gen_config.relative_top > 0.0:
                    final_logits = self._relative_top_filter(final_logits, gen_config.relative_top)
                    base_logits = base_logits.log_softmax(dim=-1)
                    mask = final_logits < -1e3
                    base_logits[mask] = -1e3
                logits = final_logits - base_logits
                next_token_logits = logits

                next_token = self._sample_from_logits(next_token_logits, gen_config.temperature, gen_config.top_p, gen_config.top_k)

                if self.stopping_criteria and next_token.item() in self.stopping_criteria:
                    print(f"Stopping criteria met at step {step}")
                    break

                input_ids = torch.cat([input_ids, next_token], dim=1)
                result.append(next_token.item())

        return result


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