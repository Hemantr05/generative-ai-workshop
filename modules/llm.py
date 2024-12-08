from langchain_community.llms import Ollama

class OllamaModel():
    def __init__(self,
                model_name: str='phi',
                prompt_file_path: str="prompts/general_prompt.txt",
                temperature: float=0.7,
                top_k: int=1,
                top_p: float=0.9,
                max_tokens: int=512,
                repeat_penalty: float=0.0,
                stream: bool=False
            ):
        
        self.model_name = model_name

        # The temperature of the model. Increasing the temperature will make the model answer more creatively. (Default: 0.8)
        self.temperature = temperature

        # Reduces the probability of generating nonsense. A higher value (e.g. 100) will give more diverse answers, while a lower value (e.g. 10) will be more conservative. (Default: 40)
        self.top_k = top_k

        # Works together with top-k. A higher value (e.g., 0.95) will lead to more diverse text, while a lower value (e.g., 0.5) will generate more focused and conservative text. (Default: 0.9)
        self.top_p = top_p

        # Maximum number of tokens to predict when generating text. (Default: 128, -1 = infinite generation, -2 = fill context)
        self.max_tokens = max_tokens

        # Sets how strongly to penalize repetitions. A higher value (e.g., 1.5) will penalize repetitions more strongly, while a lower value (e.g., 0.9) will be more lenient. (Default: 1.1)
        self.repeat_penalty = repeat_penalty

        self.stream = stream

        self.system_prompt = open(prompt_file_path, 'r').read()

        self.llm = None


    def init_model(self):
        self.llm = Ollama(
            model=self.model_name, 
            temperature=self.temperature, 
            top_k=self.top_k, 
            top_p=self.top_p, 
            num_predict=self.max_tokens)

    def __call__(self, query: str):
        self.init_model()
        messages = [
            ("system", self.system_prompt), 
            ("human", query)
        ]

        if self.stream:
            return self.llm.stream(messages)        
        return self.llm.invoke(messages)


if __name__ == "__main__":

    # query = "Write hello world code in python"
    query = "Write python program for quick sort algorithm"
    model = OllamaModel()
    print(model(query=query))

    

