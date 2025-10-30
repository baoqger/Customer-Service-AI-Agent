import os 
import asyncio
from typing import Annotated, Tuple
from dotenv import load_dotenv
from contextlib import AsyncExitStack
from semantic_kernel.agents import ChatCompletionAgent, ChatHistoryAgentThread
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion
from semantic_kernel.connectors.mcp import MCPSsePlugin, MCPStdioPlugin, MCPStreamableHttpPlugin
from semantic_kernel.connectors.ai import FunctionChoiceBehavior, PromptExecutionSettings
from semantic_kernel.functions import kernel_function, KernelArguments, KernelPlugin
from ..models import ChatMessage, Role
from semantic_kernel.contents import FunctionCallContent, FunctionResultContent, StreamingTextContent, ChatMessageContent


async def init_agent(exit_stack: AsyncExitStack) -> Tuple[ChatCompletionAgent, list[MCPStdioPlugin]]: 
    api_key = os.environ.get("API_KEY")
    deployment_name = os.environ.get("MODEL_DEPLOYMENT_NAME")
    endpoint = os.environ.get("PROJECT_ENDPOINT")

    # stdio communication case
    # plugins:list[MCPStdioPlugin] = []
    # plugin = MCPStdioPlugin(
    #     name="CustomerService",
    #     description="Customer Service Plugin",
    #     command="python",
    #     args=[".\\src\\mcp\\server.py"],
    #     env={},
    # )
    # await plugin.connect()

    # http communication case
    plugins:list[MCPStreamableHttpPlugin] = []
    plugin = MCPStreamableHttpPlugin(
        name="CustomerServiceHTTP",
        description="Customer Service HTTP Plugin",
        url="http://localhost:8000/mcp", # Replace this if you're not running it locally
        headers={"Content-Type": "application/json"},
        timeout=30,
    )
    await plugin.connect()

    # sse communication case
    # plugins:list[MCPSsePlugin] = []
    # plugin = MCPSsePlugin(
    #     name="ContosoMCP",
    #     description="Contoso MCP Plugin",
    #     url="http://localhost:8000/sse", # Replace this if you're not running it locally
    #     headers={"Content-Type": "application/json"},
    #     timeout=30,
    # )
    # await plugin.connect()


    plugins.append(plugin)

    # Configure the function choice behavior to auto invoke kernel functions
    settings =  PromptExecutionSettings()
    settings.function_choice_behavior = FunctionChoiceBehavior.Auto()

    service_id = "agent"
    instrutions = """ 
            You are a helpful assistant. You can use multiple tools to find information
            and answer questions. Review the tools available under the MCPTools plugin 
            and use them as needed. You can also ask clarifying questions if the user is not clear.
            """
    
    # Now create our agent and plug in the MCP plugin
    agent = ChatCompletionAgent(
        service=AzureChatCompletion(
            service_id=service_id, 
            api_key= api_key,
            deployment_name= deployment_name,
            endpoint=endpoint
        ),
        name="sk-agent",
        instructions=instrutions,
        plugins=[plugin],
        arguments=KernelArguments(settings=settings),
    )
    print("Agent initialized")
    return agent, plugins

async def handle_intermediate_steps(message: ChatMessageContent) -> None:
    for item in message.items or []:
        if isinstance(item, FunctionCallContent):
            print(f"Function Call:> {item.name} with arguments: {item.arguments}")
        elif isinstance(item, FunctionResultContent):
            print(f"Function Result:> {item.result} for function: {item.name}")
        else:
            print(f"{message.role}: {message.content}")

class SemanticKernelAgent: 
    """
    Semantic Kernel Agent to process user messages.
    """
    def __init__(self, agent: ChatCompletionAgent, plugins: list[MCPStdioPlugin]):
        print("sk agent init")
        self.agent: ChatCompletionAgent = agent
        self.plugins = plugins
        self.thread: ChatHistoryAgentThread = None



    async def process_message(self, message: str) -> ChatMessage:
        """
        Process a user message and return the assistant's response.
        
        Args:
            message: The user's message
            
        Returns:
            ChatMessage object containing the assistant's response
        """ 

        if not self.agent:
            return ChatMessage(
                role=Role.ASSISTANT,
                content="Semantic Kernel Agent is not properly configured. Please check your settings."
            )
        full_response: list[str] = []
        async for response in self.agent.invoke_stream(
            messages=message,
            thread=self.thread,
            on_intermediate_message=handle_intermediate_steps,
        ):
            self.thread = response.thread
            content_items = list(response.items)
            for item in content_items:
                if isinstance(item, StreamingTextContent) and item.text:
                    full_response.append(item.text)   
        print(f"agent: {''.join(full_response)}")         
        content  = "".join(full_response)
        return ChatMessage(
            role=Role.ASSISTANT,
            content=content if content else "I received your message but couldn't generate a response."
        )        

    async def cleanup(self):
        """Cleanup method for session management."""
        for plugin in self.plugins:
            await plugin.close()

    @classmethod
    async def create(cls, exit_stack: AsyncExitStack):
        agent, plugins = await init_agent(exit_stack)
        return cls(agent, plugins)