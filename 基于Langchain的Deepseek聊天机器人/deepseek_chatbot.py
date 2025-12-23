import os
import sys
import datetime
from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_deepseek import ChatDeepSeek
from langgraph.checkpoint.memory import InMemorySaver
import requests
from dotenv import load_dotenv
import gradio as gr

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

class DeepSeekChatbot:
    def __init__(self):
        self._setup_environment()
        self._setup_tools()
        self._setup_model()
        self._create_agent()
        self._setup_gradio_interface()
    
    def _setup_environment(self):
        load_dotenv(".env")
        self.api_key = SecureConfig(os.getenv('DEEPSEEK_API_KEY'))
        print(f"API Key:{self.api_key.api_key}")
    def _setup_tools(self):
        @tool
        def get_weather(city: str) -> str:
            """获取指定城市的天气信息"""
            try:
                city_codes = {
            "北京":"101010100",
            "上海":"101020100",
            "广州":"101280101",
            "深圳":"101280601",
            "杭州":"101210101",
            "成都":"101270101",
            "武汉":"101200101",
            "南京":"101190101",
            "西安":"101110101",
            "重庆":"101040100"
        }
                city_code =city_codes.get(city)
                if not city_code:
                    return f"暂不支持城市'{city}'，支持的城市:{', '.join(city_codes.keys())}"

                url =f"http://t.weather.sojson.com/api/weather/city/{city_code}"
                response = requests.get(url, timeout=10) 
                data = response.json()     

                if data.get("status") != 200: 
                    return f"获取{city}天气失败"
        
                weather_data = data["data"] 
                forecast_today = weather_data["forecast"][0]
                return f'{city}今日天气情况{forecast_today}'
            except Exception as e:
                return f"获取{city}天气失败:{str(e)}"
        
        @tool
        def calculator(expression: str) -> str:
            """计算数学表达式，支持加减乘除运算"""
            result = eval(expression, {"__builtins__": {}}, {})
            return f"表达式 {expression} 的结果是:{result}"
        @tool
        def get_current_time() -> str:
            """获取当前日期和时间"""
            now = datetime.datetime.now() 
            return f"当前时间:{now.strftime('%Y年%m月%d日%H:%M:%S')}"
        self.tools = [get_weather, calculator, get_current_time]
        print(f"工具配置完成:{[tool.name for tool in self.tools]}")
    def _setup_model(self):
        self.system_prompt = """你是一个智能聊天助手，具有以下能力：
1. **对话能力**: 能够进行自然、友好的对话
2. **工具调用**: 可以调用实时天气查询、计算器、时间查询工具
3. **上下文记忆**: 能够记住对话历史
4. **多轮交互**: 支持复杂的多轮对话
请根据用户的问题，选择合适的工具或直接回答问题。
回答时请用中文，除非用户说要用英文，否则请用中文回答。
亲切友好、耐心细致、务实可靠。"""
        
        self.llm = ChatDeepSeek(
            model="deepseek-chat",
            api_key=self.api_key.get_real_key(),   
            temperature=0.7,      
            top_p=0.9,           
            max_tokens=500,       
            timeout=30,           
            max_retries=2,        
        )
        
        print("DeepSeek模型配置完成")
    
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
                label="DeepSeek Chatbot",
                height=300  
            ),
            textbox=gr.Textbox(
                placeholder="我是DeepSeek Chatbot，可以询问我任何问题...",
                container=False,
                scale=7,
                autofocus=True
            ),
            title="DeepSeek Chatbot",
            description="这是一个基于DeepSeek的智能聊天机器人，支持天气查询、数学计算等功能",
            examples=[
                ["今天上海的天气怎么样？"],
                ["帮我计算一下 25 * 4 + 30 / 2"],
                ["现在是什么时间？"],
                ["介绍一下你自己"]
            ]
        )

if __name__ == "__main__":
    bot = DeepSeekChatbot()
    bot.demo.launch(share=True, server_port=8080)

