import os
openai_api_key = os.getenv("OPENAI_API_KEY")

from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.chains.sequential import SequentialChain
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings

import sys  
sys.stdout.reconfigure(encoding='utf-8')  

llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    api_key = openai_api_key,
    base_url= "https://api.chatanywhere.tech/v1"
)

class Experimental:

    def __init__(self,llm):
        self.llm = llm
        self.experimental_steps_chain()
        # self.embedding = OpenAIEmbeddings(
        #     openai_api_key=openai_api_key,
        #     openai_api_base="https://api.chatanywhere.tech/v1"
        # )

    def __repr__(self):
        return f"Experimental({self.experimental})"
    
    def __call__(self,experimental):
        # 综合判断进行串联  
        self.combine_chain()
        # 根据query获取experimental
        # vectorstore = Chroma(persist_directory="langchain/task_data", embedding_function=self.embedding, collection_name="openai_embed2")
        # resultn = vectorstore.similarity_search(experimental ,k = 3)
        # experimental += "\n".join([x.page_content for x in resultn])
        return self.sequential_chain.run(experimental=experimental)
        
    # 定义可能的链
    def experimental_steps_chain(self):

        # 1. 定义第一个链：总结实验步骤，，并将每个步骤分解为具体的元操作 RAG
        # experimental_template = """
        #     上下文信息如下：
        #     ----------
        #     {experimental}
        #     ----------
        #     请你基于上下文信息而不是自己的知
        #     识，回答以下问题，可以分点作答，如
        #     果上下文信息没有相关知识，可以回答
        #     不确定，不要复述上下文信息：
        #     {query_str},总结实验步骤，并将每个步骤分解为具体的元操作。
        #     回答：
        # """
        # experimental_prompt = PromptTemplate(input_variables=["experimental","query_str"], template=experimental_template)      
        # self.experimental_chain = LLMChain(llm=llm, prompt=experimental_prompt, output_key="experimental_steps")  

        #1. 定义第一个链：总结实验步骤，，并将每个步骤分解为具体的元操作
        experimental_template = """
            上下文信息如下：
            ----------
            ({experimental})
            ----------
            请你基于上下文信息而不是自己的知
            识，回答以下问题，请一步步思考：
            (总结实验步骤，并将每个步骤分解为具体的元操作。)
            回答：
        """
        experimental_prompt = PromptTemplate(input_variables=["experimental"], template=experimental_template)      
        self.experimental_chain = LLMChain(llm=llm, prompt=experimental_prompt, output_key="experimental_steps")  


        # 定义第二个链：提取上下文信息中元操作的实验器材和操作 
        steps_template = """
            上下文信息如下：
            ----------
            ({experimental_steps})
            ----------
            请你基于上下文信息而不是自己的知
            识，回答以下问题，请一步步思考：
            (
                提取上下文信息中元操作的实验器材和操作,
                一个步骤只包含一个元操作,输出格式为:
                步骤1，（实验器材，操作，实验器材）；
                步骤2，（实验器材，操作，实验器材）；
                ...
            )
            回答：
        """ 
        steps_prompt = PromptTemplate(input_variables=["experimental_steps"], template=steps_template)  
        self.steps_chain = LLMChain(llm=llm, prompt=steps_prompt, output_key="steps")  


    def combine_chain(self):
        chains = [self.experimental_chain,self.steps_chain]
         # 串联链  
        self.sequential_chain = SequentialChain(  
            chains=chains,  
            input_variables=["experimental"],  # 起始输入  
            output_variables=["steps"]  # 最终输出  
        )  


if __name__ == "__main__":  
    # if len(sys.argv) > 1:  
        # print("收到的参数:", sys.argv[1:]) 
        text = """操作1：检查装置气密性
        用火柴1点燃酒精灯，观察导管口是否会有连续、均匀气泡冒出 。

        操作2：熄灭酒精灯进行正式制取实验过程
        将酒精灯帽盖在酒精灯上。

        操作3：取药品
        打开装有高猛酸钾的塞，用药匙取高猛酸钾

        操作4：装药品
        将高锰酸钾药品平铺在试管底部，管口放一小团棉花，用带导管的单孔橡皮塞塞紧试管口。并固定在铁架台上

        操作5：点燃酒精灯加热
        用火柴2点燃酒精灯

        操作6：观察导管口是否会有连续、均匀气泡冒出，利用集气瓶收集气体
        操作7：实验结束
        """
        experimental = Experimental(llm)
        steps = experimental(text)
        print(steps)