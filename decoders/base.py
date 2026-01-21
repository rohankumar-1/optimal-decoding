from abc import ABC, abstractmethod
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.generation.stopping_criteria import StoppingCriteriaList
from configs import GenerationConfig

class BaseDecoder(ABC):
    """Abstract base class for all decoding methods."""
    
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

    @abstractmethod
    def generate(self, input_text: str, gen_config: GenerationConfig) -> str:
        """Generate sequence of token ids."""
        pass
    
    @abstractmethod
    def lm_score(self, context: str, target: str, gen_config: GenerationConfig) -> float:
        """Score likelihood of target given context."""
        pass
    
    def _sample_from_logits(self, logits, deterministic=False, temperature=1.0, top_p=0.0, top_k=0):
        if deterministic:
        logits = logits / temperature
        if top_p > 0.0:
            logits = self._top_p_filtering(logits, top_p)
        if top_k > 0:
            logits = self._top_k_filtering(logits, top_k)
        return F.softmax(logits, dim=-1)

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