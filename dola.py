import torch
import heapq
from transformers import AutoModelForCausalLM, AutoTokenizer

class DoLaBeamSearchDecoder:
    def __init__(self, model, tokenizer, beam_width=4, mature_layer=31, 
                 early_layer=None, contrast_alpha=0.1):
        self.model = model
        self.tokenizer = tokenizer
        self.beam_width = beam_width
        self.mature_layer = mature_layer
        self.early_layer = early_layer
        self.contrast_alpha = contrast_alpha
    
    def get_dola_logits(self, input_ids):
        """
        Compute logits using DoLa contrast.
        Returns: logits of shape [batch_size, vocab_size]
        """
        with torch.no_grad():
            outputs = self.model(
                input_ids,
                output_hidden_states=True,
                return_dict=True
            )
        
        final_logits = outputs.logits[:, -1, :]  # [batch, vocab]
        
        # Get early layer logits (you could do dynamic selection here too)
        if self.early_layer is None:
            # Dynamic selection: find layer with max divergence
            early_logits = self._select_early_layer_dynamic(outputs, final_logits)
        else:
            # Static selection
            early_hidden = outputs.hidden_states[self.early_layer][:, -1, :]
            if self.model.lm_head.bias is None:
                early_logits = early_hidden @ self.model.embed_tokens.weight.T
            else:
                early_logits = early_hidden @ self.model.lm_head.weight.T + self.model.lm_head.bias
        
        # DoLa: contrast the logits
        dola_logits = final_logits - self.contrast_alpha * early_logits
        return dola_logits
    
    def _select_early_layer_dynamic(self, outputs, final_logits):
        """Select early layer with maximum Jensen-Shannon divergence."""
        # Simplified: just use a fixed set of candidate layers
        candidate_layers = [0, 8, 16, 24, self.mature_layer - 4]
        
        p_final = torch.softmax(final_logits, dim=-1)
        max_divergence = -float('inf')
        best_early_logits = None
        
        for layer_idx in candidate_layers:
            if layer_idx >= len(outputs.hidden_states):
                continue
            
            early_hidden = outputs.hidden_states[layer_idx][:, -1, :]
            if self.model.lm_head.bias is None:
                early_logits = early_hidden @ self.model.embed_tokens.weight.T
            else:
                early_logits = early_hidden @ self.model.lm_head.weight.T + self.model.lm_head.bias
            p_early = torch.softmax(early_logits, dim=-1)
            
            # Jensen-Shannon divergence (simplified)
            jsd = self._jsd(p_final, p_early)
            
            if jsd > max_divergence:
                max_divergence = jsd
                best_early_logits = early_logits
        
        return best_early_logits if best_early_logits is not None else early_logits
    
    def _jsd(self, p, q):
        """Jensen-Shannon divergence between two distributions."""
        p_avg = 0.5 * (p + q)
        kl_p = torch.sum(p * (torch.log(p + 1e-10) - torch.log(p_avg + 1e-10)), dim=-1)
        kl_q = torch.sum(q * (torch.log(q + 1e-10) - torch.log(p_avg + 1e-10)), dim=-1)
        return 0.5 * (kl_p + kl_q)
    
    def decode(self, prompt, max_length=50, temperature=1.0, top_k=50):
        """
        Beam search decoding with DoLa guidance.
        
        Returns the best sequence according to beam score.
        """
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt")
        batch_size = input_ids.shape[0]
        
        # Heap: (negative_score, sequence_id, input_ids, length)
        # We use negative score so heapq gives us max-heap behavior
        beams = []
        for i in range(self.beam_width):
            heapq.heappush(beams, (0.0, i, input_ids.clone(), 0))
        
        eos_token_id = self.tokenizer.eos_token_id
        pad_token_id = self.tokenizer.pad_token_id
        
        for step in range(max_length):
            # Collect all candidate beams
            candidates = []  # (score, seq_id, sequence, length)
            
            while beams:
                neg_score, seq_id, sequence, length = heapq.heappop(beams)
                score = -neg_score
                
                # Get DoLa logits for this sequence
                dola_logits = self.get_dola_logits(sequence)
                
                # Get log probabilities
                log_probs = torch.log_softmax(dola_logits, dim=-1)  # [1, vocab]
                
                # Get top-k candidates
                topk_log_probs, topk_indices = torch.topk(log_probs, k=min(top_k, log_probs.shape[-1]), dim=-1)
                
                for k in range(topk_log_probs.shape[1]):
                    token_id = topk_indices[0, k].item()
                    token_log_prob = topk_log_probs[0, k].item()
                    
                    # New sequence: append this token
                    new_sequence = torch.cat([sequence, torch.tensor([[token_id]])], dim=1)
                    new_score = score + token_log_prob
                    
                    # Normalize score by length to avoid favoring short sequences
                    normalized_score = new_score / (length + 1)
                    
                    candidates.append((
                        -normalized_score,  # negative for max-heap
                        seq_id * 1000 + k,  # unique id
                        new_sequence,
                        length + 1
                    ))
            
            # Keep only top beam_width candidates
            candidates.sort()
            for i in range(min(self.beam_width, len(candidates))):
                neg_score, seq_id, sequence, length = candidates[i]
                heapq.heappush(beams, (neg_score, seq_id, sequence, length))
            
            # Check if all beams have ended
            all_ended = all(sequence[0, -1].item() == eos_token_id for _, _, sequence, _ in beams)
            if all_ended:
                break
        
        # Return best sequence
        best_beam = min(beams, key=lambda x: x[0])
        best_sequence = best_beam[2]
        
        return self.tokenizer.decode(best_sequence[0], skip_special_tokens=True)


# Usage
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")

decoder = DoLaBeamSearchDecoder(
    model=model,
    tokenizer=tokenizer,
    beam_width=4,
    mature_layer=31,
    early_layer=16,  # or None for dynamic selection
    contrast_alpha=0.1
)

result = decoder.decode(
    prompt="The capital of France is",
    max_length=20
)
print(result)