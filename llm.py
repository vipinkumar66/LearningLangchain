from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain.chains import LLMChain
from dotenv import load_dotenv
import os

load_dotenv()


class LLMModel:

    def __init__(self):
        pass

    def azure_llm_model(self, temperature=None, deployment_name=None):

        if temperature is None:
            temperature = 0.5

        if deployment_name:
            azure_deployment_name = deployment_name
        else:
            azure_deployment_name = 'DEPLOYMENT_NAME_GPT4o'

        azure_cred_llm = AzureChatOpenAI(
            openai_api_base=os.environ.get('API_BASE') + os.environ.get(azure_deployment_name),
            openai_api_version=os.environ.get('API_VERSION'),
            openai_api_key=os.environ.get('API_KEY'),
            openai_api_type=os.environ.get('API_TYPE'),
            # streaming=True,
            # model="gpt-4", #need to specify when i was using the trimmer,
            temperature=temperature
        )
        return azure_cred_llm

    def create_embeddings(self):
        azure_deployment_name = 'EMBEDDING_DEPLOYMENT_NAME'

        azure_embeddings = AzureOpenAIEmbeddings(
            openai_api_base=os.environ.get('API_BASE') + os.environ.get(azure_deployment_name),
            openai_api_version=os.environ.get('API_VERSION'),
            openai_api_key=os.environ.get('API_KEY'),
            openai_api_type=os.environ.get('API_TYPE'),
        )
        return azure_embeddings

    def llm_chain(self, prompt, **kwargs):

        azure_llm_model_kwargs = {}
        if 'deployment_name' in kwargs:
            azure_llm_model_kwargs['deployment_name'] = kwargs['deployment_name']
            del kwargs['deployment_name']

        llm_chain = LLMChain(llm=self.azure_llm_model(**azure_llm_model_kwargs), prompt=prompt)
        paraphrase_text = llm_chain.predict(**kwargs)
        return paraphrase_text
