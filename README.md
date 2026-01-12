

# Beam + DoLa:

Goal: how do we guide decoding to factuality? 

Dola tries to use layer differences, but does not have a good way to guide the decoding process. Can we use beam search with DoLa and a repetition penalty?

Also, instead of normal DoLa, can we dynamically adjust alpha, via adaptive-DoLa?


# TODO (priority order)
- [ ] DoLa generation
    - [ ] add relative top filter from paper
    - [ ] add repitition penalty
- add stopping criteria properly
- [ ] implement beam search
    - add each beam to heap, evaluate each beam, keep top k beams
- [ ] Support batch size > 1



Do we need to train lm_heads for each layer? Seems bad to only use last layer's lm_head on all hidden states.

I think the DoLa implementation is still wrong: do we log_softmax before or after the difference?

