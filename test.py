from decoders.baseline import BaselineDecoder
from decoders.dola import DoLaDecoder
from configs import GenerationConfig, DoLaConfig

def test_baseline_decoder(model_name, device, gen_config, question):
    baseline_decoder = BaselineDecoder(model_name, device)
    output = baseline_decoder.generate(question, gen_config)
    return output

def test_baseline_lmscore(model_name, device, gen_config, question, choices):
    baseline_decoder = BaselineDecoder(model_name, device)
    responses = {}
    for i in choices:
        responses[i] = baseline_decoder.lm_score(question, i, gen_config)
    return responses

def test_dola_decoder(model_name, device, gen_config, question):
    dola_config = DoLaConfig(
        candidate_premature_layers=[0, 8, 16, 24],
        mature_layer=-1,
    )
    dola_decoder = DoLaDecoder(model_name, device, config=dola_config)
    output = dola_decoder.generate(question, gen_config)
    return output

def test_dola_lmscore(model_name, device, gen_config, question, choices):
    dola_config = DoLaConfig(
        candidate_premature_layers=[0, 8, 16, 24],
        mature_layer=-1,
    )
    baseline_decoder = DoLaDecoder(model_name, device, dola_config)
    responses = {}
    for i in choices:
        responses[i] = baseline_decoder.lm_score(question, i, gen_config)
    return responses


if __name__ == "__main__":
    model_name = "Qwen/Qwen3-0.6B"
    device = "cpu"
    gen_config = GenerationConfig(
        max_new_tokens=10,
        temperature=0.9,
        top_p=0.0,
        top_k=0,
        deterministic=True,
    )

    ### LM head test
    input_text = "What is the capital of France?"
    responses = ["Paris", "Riga", "Bologna", "Teardrop"]

    print(test_baseline_lmscore(model_name, device, gen_config, question=input_text, choices=responses))


    ### Generation test
    
    # input_text = "What is the capital of France? "
    # print("Testing baseline decoder...")
    # print("Q: What is the capital of France?")
    # print("A: ", test_baseline_decoder(model_name, device, gen_config, input_text))
    # print("")
    # print("Testing DoLa decoder...")
    # print("Q: What is the capital of France?")
    # print("A: ", test_dola_decoder(model_name, device, gen_config, input_text))
    # print("")