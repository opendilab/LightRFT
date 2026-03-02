from vllm import LLM, SamplingParams
# from vllm_ascend import LLM, SamplingParams

prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]

def main():
    # Create a sampling params object.
    sampling_params = SamplingParams(temperature=0.8, top_p=0.95)
    # Create an LLM.
    #llm = LLM(model="Qwen/Qwen2.5-0.5B-Instruct")
    llm = LLM(model="/data/puyuan/LightRFT/model/Qwen2.5-0.5B-Instruct/7ae557604adf67be50417f59c2c2f167def9a775/")


    # Generate texts from the prompts.
    outputs = llm.generate(prompts, sampling_params)
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")


if __name__ == "__main__":
    main()
