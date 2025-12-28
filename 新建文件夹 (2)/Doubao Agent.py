import os
from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from dotenv import load_dotenv
import gradio as gr
from openai import OpenAI


class SecureConfig: 
    def __init__(self, api_key: str):
        self._api_key = api_key
    @property 
    def api_key(self):
        """安全显示API密钥（只显示部分）"""
        if self._api_key:
            return f'{self._api_key[:8]}...{self._api_key[-4:]}'
        return None
    def get_real_key(self):
        """获取真实API密钥"""
        return self._api_key

class DOUBAO_SEED:
    def __init__(self):
        self._setup_environment()
        self._setup_tools()
        self._setup_model()
        self._create_agent()
        self._setup_gradio_interface()
    
    def _setup_environment(self):
        load_dotenv(".env")
        # 使用ARK API密钥而不是DEEPSEEK API密钥
        ark_api_key = os.getenv('ARK_API_KEY')
        if not ark_api_key:
            raise ValueError("请在.env文件中设置ARK_API_KEY环境变量")
        self.api_key = SecureConfig(ark_api_key)
        print(f"ARK API Key:{self.api_key.api_key}")
    

    def _setup_tools(self):
        @tool("根据描述生成图片")
        def generate_image_tool(description: str) -> str:
            """使用Doubao Seed模型根据描述生成图片"""
            client=OpenAI( 
                # The base URL for model invocation
                base_url="https://ark.cn-beijing.volces.com/api/v3", 
                # Get API Key：https://console.volcengine.com/ark/region:ark+cn-beijing/apikey
                api_key=self.api_key.get_real_key(), 
                timeout=60.0,  # 设置60秒超时
            ) 
            response = client.images.generate(
            model="ep-20251228095312-nkrcj",  # 使用正确的模型ID
            prompt=description,
            size="1024x1024",
            response_format="url",
            n=1,
            extra_body={
                "watermark": False,
            },
        )
            return response.data[0].url
        
        self.tools = [generate_image_tool]
        print(f"工具配置完成:{[tool.name for tool in self.tools]}")
    def _setup_model(self):
        self.system_prompt = """你是一个专业的多媒体生成助手，可以根据用户的描述生成图片。
请仔细理解用户的描述，然后选择合适的工具生成符合描述的内容。
如果用户要求生成图片，请使用图片生成工具。
亲切友好、耐心细致地为用户提供服务。"""
        
        self.llm = ChatOpenAI(
            openai_api_key=os.getenv("ARK_API_KEY"),
            openai_api_base="https://ark.cn-beijing.volces.com/api/v3",
            model="ep-20251228095159-vrkrx",
            temperature=0.7,
            max_tokens=500,
            timeout=60.0,  # 设置60秒超时
        )
        
        print("Doubao模型配置完成")
    
    def _create_agent(self):
        self.agent = create_agent(
            model=self.llm,
            tools=self.tools,
            system_prompt=self.system_prompt,
            checkpointer=InMemorySaver() 
        )
        self.config = {"configurable": {"thread_id":"default-session"}}
    def response(self, message, history):
        """处理用户消息并返回助手回复"""
        config = self.config
        response = self.agent.invoke(
            {"messages": [{"role": "user", "content": message}]},
            config=config
            )
        return response["messages"][-1].content

    def _setup_gradio_interface(self):
        """设置Gradio界面"""
        self.demo = gr.ChatInterface(
            fn=self.response,
            chatbot=gr.Chatbot(
                label="图片生成助手",
                height=600  
            ),
            textbox=gr.Textbox(
                placeholder="我是多功能助手，可以生成图片",
                container=False,
                scale=7,
                autofocus=True
            ),
            title="多功能助手",
            description="这是一个支持图片和视频生成的AI助手",
            examples=[
                ["生成一个美丽的风景图片，有山有水"],
                ["画一只可爱的小猫"],
            ]
        )

if __name__ == "__main__":
    bot = DOUBAO_SEED()
    bot.demo.launch(share=True, server_port=8080)

