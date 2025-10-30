import os, time
import json
from contextlib import AsyncExitStack
from typing import Optional
from mcp import ClientSession, StdioServerParameters
from mcp.types import Tool
from mcp.client.stdio import stdio_client
from azure.identity import DefaultAzureCredential
from azure.ai.projects import AIProjectClient
from azure.ai.agents.models import FunctionTool
from ..models import ChatMessage, Role

async def connect_to_server(exit_stack: AsyncExitStack):
    
    # This is a MCP server written in .NET
    # server_params = StdioServerParameters(
    #     command="C:\\develop\\open-source\\azure-ai-agents-dotnet\\MiniMCPServer\\bin\\Release\\net8.0\\MiniMCPServer.exe",
    #     args=[],
    #     env=None
    # )

    # This is a MCP server written in Python
    server_params = StdioServerParameters(
        command="python",
        args=[".\\src\\local-mcp\\server.py"],
        env=None
    )  

    # Start the MCP server
    stdio_transport = await exit_stack.enter_async_context(stdio_client(server_params))
    stdio, write = stdio_transport

    # Create an MCP client session
    session = await exit_stack.enter_async_context(ClientSession(stdio, write))
    await session.initialize()

    # List available tools
    response = await session.list_tools()
    tools = response.tools  

    # Build a function for each tool
    def make_tool_func(tool_name):
        async def tool_func(**kwargs):
            result = await session.call_tool(tool_name, kwargs)
            return result
        
        tool_func.__name__ = tool_name
        return tool_func  

    functions_dict = {tool.name: make_tool_func(tool.name) for tool in tools}
    # mcp_function_tool = FunctionTool(functions=list(functions_dict.values())) 

    print("\nConnected to server with tools:", [tool.name for tool in tools]) 
    return functions_dict



class FoundryTaskAgent:
    """
    Agent that interfaces with Azure AI Foundry to process user messages.
    
    This agent:
    - Initializes connection to Azure AI Foundry using environment variables
    - Manages agent session and conversation thread
    - Sends user messages to agent and retrieves responses
    - Handles errors and configuration issues gracefully
    
    Environment variables required:
    - AZURE_AI_FOUNDRY_PROJECT_ENDPOINT: The endpoint URL for the Azure AI Foundry project
    - AZURE_AI_FOUNDRY_AGENT_ID: The identifier of the agent to use
    """
    
    def __init__(self, functions_dict):
        mcpTools = FunctionTool(functions=list(functions_dict.values()))
        self.tools = mcpTools
        self.functions_dict = functions_dict
        self.project_client = None
        self.thread_id = None
        self.agent_id = None

        # Initialize the agent
        endpoint = os.getenv("AZURE_AI_FOUNDRY_PROJECT_ENDPOINT")
        model_deployment = os.getenv("MODEL_DEPLOYMENT_NAME")

        if not endpoint or not model_deployment:
            print("Azure AI Foundry configuration missing. Set AZURE_AI_FOUNDRY_PROJECT_ENDPOINT and AZURE_AI_FOUNDRY_AGENT_ID")
            return  
        
        try:
            # Create the project client using Azure credentials
            self.project_client = AIProjectClient(
                endpoint=endpoint,
                credential=DefaultAzureCredential()
            )

            # Create the agent
            agent = self.project_client.agents.create_agent(
                model=model_deployment,
                name="inventory-agent",
                instructions="""
                You are an inventory assistant. Here are some general guidelines:
                - Recommend restock if item inventory < 10  and weekly sales > 15
                - Recommend clearance if item inventory > 20 and weekly sales < 5
                """,
                tools=mcpTools.definitions
            )  
            self.agent_id = agent.id

            print(f"Created agent: {self.agent_id}")

            # Enable auto function calling
            self.project_client.agents.enable_auto_function_calls(tools=mcpTools)

            # Create a thread for this session
            thread = self.project_client.agents.threads.create()
            self.thread_id = thread.id
            print(f"Created thread: {self.thread_id}")
            print("Azure AI Foundry Task Agent initialized successfully")
            
        except ImportError as e:
            print(f"Azure AI Projects SDK not available. Install azure-ai-projects package: {e}")
        except Exception as e:
            print(f"Failed to initialize Azure AI Foundry agent: {e}")        
    
    async def process_message(self, message: str) -> ChatMessage:
        """
        Process a user message and return the assistant's response.
        
        Args:
            message: The user's message
            
        Returns:
            ChatMessage object containing the assistant's response
        """
        if not self.project_client or not self.thread_id:
            return ChatMessage(
                role=Role.ASSISTANT,
                content="Azure AI Foundry agent is not properly configured. Please check your settings."
            )
        
        try:
            # Create the message in the thread
            message_obj = self.project_client.agents.messages.create(
                thread_id=self.thread_id,
                role="user",
                content=message
            )
            print(f"Created message, ID: {message_obj.id}")
            
            # Create and process the run
            run = self.project_client.agents.runs.create(
                thread_id=self.thread_id,
                agent_id=self.agent_id
            )

            while run.status in ["queued", "in_progress", "requires_action"]:
                time.sleep(1)
                run = self.project_client.agents.runs.get(thread_id=self.thread_id, run_id=run.id)
                tool_outputs = []

                if run.status == "requires_action":

                    print("status:  requires_action")

                    tool_calls = run.required_action.submit_tool_outputs.tool_calls

                    for tool_call in tool_calls:

                        # Retrieve the matching function tool
                        function_name = tool_call.function.name
                        args_json = tool_call.function.arguments
                        kwargs = json.loads(args_json)
                        required_function = self.functions_dict.get(function_name)
                        
                        print("function call name:", function_name)
                        print("function call args:", args_json, kwargs)

                        # Invoke the function
                        output = await required_function(**kwargs)

                        print("function output:", output)

                        # Append the output text
                        tool_outputs.append({
                            "tool_call_id": tool_call.id,
                            "output": output.content[0].text,
                        })
                    
                    # Submit the tool call output
                    self.project_client.agents.runs.submit_tool_outputs(thread_id=self.thread_id, run_id=run.id, tool_outputs=tool_outputs)

            print(f"Run finished with status:  {run.status}")
            
            if run.status == "failed":
                print(f"Run failed: {run.last_error}")
                return ChatMessage(
                    role=Role.ASSISTANT,
                    content="I encountered an error processing your request. Please try again."
                )
            
            if run.status == "completed":
                # Fetch the latest messages from the thread
                messages = self.project_client.agents.messages.list(thread_id=self.thread_id)
                
                # Find the latest assistant message
                for msg in messages:
                    if msg.role == "assistant":
                        # Extract text content from the message
                        content = ""
                        if hasattr(msg, 'content') and msg.content:
                            for content_item in msg.content:
                                if hasattr(content_item, 'text') and hasattr(content_item.text, 'value'):
                                    print("Found text content:", content_item.text.value)
                                    content += content_item.text.value
                                elif hasattr(content_item, 'value'):
                                    print("Found generic content:", content_item.value)
                                    content += str(content_item.value)
                        
                        return ChatMessage(
                            role=Role.ASSISTANT,
                            content=content if content else "I received your message but couldn't generate a response."
                        )
                
                return ChatMessage(
                    role=Role.ASSISTANT,
                    content="I processed your request but couldn't find a response."
                )
            else:
                return ChatMessage(
                    role=Role.ASSISTANT,
                    content=f"I encountered an issue processing your request. Status: {run.status}"
                )
                
        except Exception as e:
            print(f"Error processing message with Azure AI Foundry: {e}")
            import traceback
            traceback.print_exc()
            return ChatMessage(
                role=Role.ASSISTANT,
                content="I apologize, but I encountered an error processing your request."
            )
    
    async def cleanup(self):
        """Cleanup method for session management (no-op for Azure AI Foundry)."""
        # Azure AI Foundry handles cleanup automatically
        self.project_client.agents.delete_agent(self.agent_id)

    @classmethod
    async def create(cls, exit_stack: AsyncExitStack):
        functionDics = await connect_to_server(exit_stack)
        return cls(functionDics)
