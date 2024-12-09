import os
openai_api_key = os.getenv("OPENAI_API_KEY")

# 永久禁止显示所有警告  
import warnings  

# 仅屏蔽 LangChain 的弃用警告  
warnings.filterwarnings("ignore", category=UserWarning, message=".*LangChainDeprecationWarning.*")

from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.chains.sequential import SequentialChain

import sys  
sys.stdout.reconfigure(encoding='utf-8')  

llm = ChatOpenAI(
    model="gpt-4",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    api_key = openai_api_key,
    base_url= "https://api.chatanywhere.tech/v1"
    # organization="...",
    # other params...
)

class Experimental:

    def __init__(self,llm):
        self.llm = llm
        self.experimental_steps_chain()

    def __repr__(self):
        return f"Experimental({self.experimental})"
    
    def __call__(self,experimental):
        # 综合判断进行串联  
        self.combine_chain()
        return self.sequential_chain.run(experimental)
        
    # 定义可能的链
    def experimental_steps_chain(self):
        # 定义一个上下文链（已有的初高中相关实验）/存数据库/存向量数据库 ，输出直接添加到experimental
        # contex_template = "已知实验相关信息： {experimental}。请给出这个实验所需的步骤。"  
        # contex_prompt = PromptTemplate(input_variables=["experimental"], template=experimental_template)      
        # self.experimental_chain = LLMChain(llm=llm, prompt=experimental_prompt, output_key="experimental_steps",verbose=True) 

        # 1. 定义第一个链：给出实验的步骤
        experimental_template = "已知实验相关信息： {experimental}。请给出这个实验所需的步骤。实验结束，每个步骤包含一个实验器材及其对应操作，如将高锰酸钾药品平铺在试管底部，管口放一小团棉花，分成两个步骤：1.高锰酸钾药品平铺在试管底部；2.管口放一小团棉花，需要注意：实验结束并不包含实验物体不应该输出为步骤"  
        experimental_prompt = PromptTemplate(input_variables=["experimental"], template=experimental_template)      
        self.experimental_chain = LLMChain(llm=llm, prompt=experimental_prompt, output_key="experimental_steps")  

        # 定义第二个链：基于上述实验步骤提取每个步骤中的物体和操作 
        steps_template = "根据实验步骤：{experimental_steps}，输出，步骤，（实验器材，操作，实验器材），如火柴1点燃酒精灯输出为（火柴1，点燃，酒精灯）"  
        steps_prompt = PromptTemplate(input_variables=["experimental_steps"], template=steps_template)  
        self.steps_chain = LLMChain(llm=llm, prompt=steps_prompt, output_key="steps")  

        # 判断选取特殊物体容量，如容器

        # 判断结果准确率是否达到预期
        # check_template = "比较实验步骤：{experimental_steps}，比较输出（步骤，实验器材，操作，被操作实验器材）：{steps}，判断是否达到80%的准确率，回答yes/no"  
        # check_prompt = PromptTemplate(input_variables=["experimental_steps"], template=check_template)  
        # self.check_chain = LLMChain(llm=llm, prompt=check_prompt, output_key="check")  

    def combine_chain(self):
        chains = [self.experimental_chain,self.steps_chain]
         # 串联链  
        self.sequential_chain = SequentialChain(  
            chains=chains,  
            input_variables=["experimental"],  # 起始输入  
            output_variables=["steps"]  # 最终输出  
        )  


if __name__ == "__main__":  
    if len(sys.argv) > 1:  
        print("收到的参数:", sys.argv[1:]) 
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