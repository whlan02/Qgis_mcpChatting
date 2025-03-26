import asyncio
import json
import logging
import os
import shutil
import sys
from typing import Dict, List, Optional, Any

import requests
from dotenv import load_dotenv
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class Configuration:
    """Manages configuration and environment variables for the MCP client."""

    def __init__(self) -> None:
        """Initialize configuration with environment variables."""
        self.load_env()
        self.api_key = os.getenv("LLM_API_KEY")
        # self.api_key = os.getenv("GITHUB_API_KEY")

    @staticmethod
    def load_env() -> None:
        """Load environment variables from .env file."""
        load_dotenv()

    @staticmethod
    def load_config(file_path: str) -> Dict[str, Any]:
        """Load server configuration from JSON file.
        
        Args:
            file_path: Path to the JSON configuration file.
            
        Returns:
            Dict containing server configuration.
            
        Raises:
            FileNotFoundError: If configuration file doesn't exist.
            JSONDecodeError: If configuration file is invalid JSON.
        """
        with open(file_path, 'r') as f:
            return json.load(f)

    @property
    def llm_api_key(self) -> str:
        """Get the LLM API key.
        
        Returns:
            The API key as a string.
            
        Raises:
            ValueError: If the API key is not found in environment variables.
        """
        if not self.api_key:
            raise ValueError("LLM_API_KEY not found in environment variables")
        return self.api_key


class Server:
    """Manages MCP server connections and tool execution."""

    def __init__(self, name: str, config: Dict[str, Any]) -> None:
        self.name: str = name
        self.config: Dict[str, Any] = config
        self.stdio_context: Optional[Any] = None
        self.session: Optional[ClientSession] = None
        self._cleanup_lock: asyncio.Lock = asyncio.Lock()
        self.capabilities: Optional[Dict[str, Any]] = None

    async def initialize(self) -> None:
        """Initialize the server connection."""
        server_params = StdioServerParameters(
            command=shutil.which("npx") if self.config['command'] == "npx" else self.config['command'],
            args=self.config['args'],
            env={**os.environ, **self.config['env']} if self.config.get('env') else None
        )
        try:
            self.stdio_context = stdio_client(server_params)
            read, write = await self.stdio_context.__aenter__()
            self.session = ClientSession(read, write)
            await self.session.__aenter__()
            self.capabilities = await self.session.initialize()
        except Exception as e:
            logging.error(f"Error initializing server {self.name}: {e}")
            await self.cleanup()
            raise

    async def list_tools(self) -> List[Any]:
        """List available tools from the server.
        
        Returns:
            A list of available tools.
            
        Raises:
            RuntimeError: If the server is not initialized.
        """
        if not self.session:
            raise RuntimeError(f"Server {self.name} not initialized")
        
        tools_response = await self.session.list_tools()
        tools = []
        
        supports_progress = (
            self.capabilities 
            and 'progress' in self.capabilities
        )
        
        if supports_progress:
            logging.info(f"Server {self.name} supports progress tracking")
        
        for item in tools_response:
            if isinstance(item, tuple) and item[0] == 'tools':
                for tool in item[1]:
                    tools.append(Tool(tool.name, tool.description, tool.inputSchema))
                    if supports_progress:
                        logging.info(f"Tool '{tool.name}' will support progress tracking")
        
        return tools

    async def execute_tool(
        self, 
        tool_name: str, 
        arguments: Dict[str, Any], 
        retries: int = 2, 
        delay: float = 1.0
    ) -> Any:
        """Execute a tool with retry mechanism.
        
        Args:
            tool_name: Name of the tool to execute.
            arguments: Tool arguments.
            retries: Number of retry attempts.
            delay: Delay between retries in seconds.
            
        Returns:
            Tool execution result.
            
        Raises:
            RuntimeError: If server is not initialized.
            Exception: If tool execution fails after all retries.
        """
        if not self.session:
            raise RuntimeError(f"Server {self.name} not initialized")

        attempt = 0
        while attempt < retries:
            try:
                supports_progress = (
                    self.capabilities 
                    and 'progress' in self.capabilities
                )

                if supports_progress:
                    logging.info(f"Executing {tool_name} with progress tracking...")
                    result = await self.session.call_tool(
                        tool_name, 
                        arguments,
                        progress_token=f"{tool_name}_execution"
                    )
                else:
                    logging.info(f"Executing {tool_name}...")
                    result = await self.session.call_tool(tool_name, arguments)

                return result

            except Exception as e:
                attempt += 1
                logging.warning(f"Error executing tool: {e}. Attempt {attempt} of {retries}.")
                if attempt < retries:
                    logging.info(f"Retrying in {delay} seconds...")
                    await asyncio.sleep(delay)
                else:
                    logging.error("Max retries reached. Failing.")
                    raise

    async def cleanup(self) -> None:
        """Clean up server resources."""
        async with self._cleanup_lock:
            try:
                if self.session:
                    try:
                        await self.session.__aexit__(None, None, None)
                    except Exception as e:
                        logging.warning(f"Warning during session cleanup for {self.name}: {e}")
                    finally:
                        self.session = None

                if self.stdio_context:
                    try:
                        await self.stdio_context.__aexit__(None, None, None)
                    except (RuntimeError, asyncio.CancelledError) as e:
                        logging.info(f"Note: Normal shutdown message for {self.name}: {e}")
                    except Exception as e:
                        logging.warning(f"Warning during stdio cleanup for {self.name}: {e}")
                    finally:
                        self.stdio_context = None
            except Exception as e:
                logging.error(f"Error during cleanup of server {self.name}: {e}")


class Tool:
    """Represents a tool with its properties and formatting."""

    def __init__(self, name: str, description: str, input_schema: Dict[str, Any]) -> None:
        self.name: str = name
        self.description: str = description
        self.input_schema: Dict[str, Any] = input_schema

    def format_for_llm(self) -> str:
        """Format tool information for LLM.
        
        Returns:
            A formatted string describing the tool.
        """
        args_desc = []
        if 'properties' in self.input_schema:
            for param_name, param_info in self.input_schema['properties'].items():
                arg_desc = f"- {param_name}: {param_info.get('description', 'No description')}"
                if param_name in self.input_schema.get('required', []):
                    arg_desc += " (required)"
                args_desc.append(arg_desc)
        
        return f"""
Tool: {self.name}
Description: {self.description}
Arguments:
{chr(10).join(args_desc)}
"""


class LLMClient:
    """Manages communication with the LLM provider."""

    def __init__(self, api_key: str) -> None:
        self.api_key: str = api_key

    def get_response(self, messages: List[Dict[str, str]]) -> str:
        """Get a response from the LLM.
        
        Args:
            messages: A list of message dictionaries.
            
        Returns:
            The LLM's response as a string.
            
        Raises:
            RequestException: If the request to the LLM fails.
        """
        url = "https://api.openai.com/v1/chat/completions"

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        payload = {
            "messages": messages,
            "model": "gpt-4o-mini",
            "temperature": 0.7,
            "max_tokens": 4096,
            "top_p": 1,
            "stream": False
        }
        
        try:
            response = requests.post(url, headers=headers, json=payload)
            response.raise_for_status()
            data = response.json()
            return data['choices'][0]['message']['content']
            
        except requests.exceptions.RequestException as e:
            error_message = f"Error getting LLM response: {str(e)}"
            logging.error(error_message)
            
            if e.response is not None:
                status_code = e.response.status_code
                logging.error(f"Status code: {status_code}")
                logging.error(f"Response details: {e.response.text}")
                
            return f"I encountered an error: {error_message}. Please try again or rephrase your request."


class ChatSession:
    """Orchestrates the interaction between user, LLM, and tools."""

    def __init__(self, servers: List[Server], llm_client: LLMClient) -> None:
        self.servers: List[Server] = servers
        self.llm_client: LLMClient = llm_client
        self.messages = []
        self.all_tools = []

    async def cleanup_servers(self) -> None:
        """Clean up all servers properly."""
        for server in self.servers:
            try:
                await server.cleanup()
            except Exception as e:
                logging.warning(f"Warning during cleanup of server: {e}")

    async def process_llm_response(self, llm_response: str) -> str:
        """Process the LLM response and execute tools if needed."""
        try:
            # Try to find all JSON tool calls in the response
            tool_calls = []
            current_pos = 0
            while current_pos < len(llm_response):
                try:
                    # Find opening brace
                    start = llm_response.find('{', current_pos)
                    if start == -1:
                        break
                        
                    # Find closing brace
                    count = 1
                    pos = start + 1
                    while count > 0 and pos < len(llm_response):
                        if llm_response[pos] == '{':
                            count += 1
                        elif llm_response[pos] == '}':
                            count -= 1
                        pos += 1
                    
                    if count == 0:
                        # Extract and parse JSON
                        json_str = llm_response[start:pos]
                        tool_call = json.loads(json_str)
                        if isinstance(tool_call, dict) and "tool" in tool_call and "arguments" in tool_call:
                            tool_calls.append(tool_call)
                        current_pos = pos
                    else:
                        break
                except json.JSONDecodeError:
                    current_pos += 1
                    continue
            
            if tool_calls:
                results = []
                for tool_call in tool_calls:
                    result = await self._execute_tool(tool_call)
                    results.append(result)
                return "Tool execution results: " + json.dumps(results)
            
            return llm_response
        except Exception as e:
            logging.error(f"Error processing LLM response: {e}")
            return llm_response

    async def _execute_tool(self, tool_call: dict) -> str:
        """Execute a single tool call."""
        logging.info(f"Executing tool: {tool_call['tool']}")
        logging.info(f"With arguments: {tool_call['arguments']}")
        
        for server in self.servers:
            tools = await server.list_tools()
            if any(tool.name == tool_call["tool"] for tool in tools):
                try:
                    result = await server.execute_tool(tool_call["tool"], tool_call["arguments"])
                    
                    if isinstance(result, dict) and 'progress' in result:
                        progress = result['progress']
                        total = result['total']
                        logging.info(f"Progress: {progress}/{total} ({(progress/total)*100:.1f}%)")
                        
                    return f"Tool execution result: {result}"
                except Exception as e:
                    error_msg = f"Error executing tool: {str(e)}"
                    logging.error(error_msg)
                    return error_msg
        
        return f"No server found with tool: {tool_call['tool']}"

    async def initialize_servers(self) -> bool:
        """Initialize all servers and collect tools."""
        for server in self.servers:
            try:
                await server.initialize()
            except Exception as e:
                logging.error(f"Failed to initialize server: {e}")
                await self.cleanup_servers()
                return False
        
        for server in self.servers:
            tools = await server.list_tools()
            self.all_tools.extend(tools)
        
        return True

    def setup_system_message(self) -> None:
        """Set up the system message with tools information."""
        tools_description = "\n".join([tool.format_for_llm() for tool in self.all_tools])
        
        system_message = f"""You are a helpful assistant with access to these tools: 

{tools_description}
Choose the appropriate tool(s) based on the user's question. If no tool is needed, reply directly.

IMPORTANT: When you need to use multiple tools, you MUST combine them into a single JSON array like this:
[
    {{
        "tool": "get_layers",
        "arguments": {{}}
    }},
    {{
        "tool": "remove_layer",
        "arguments": {{
            "layer_id": "<id_from_get_layers_result>"
        }}
    }}
]

DO NOT send multiple separate tool calls in the same message. Always use the array format for multiple tools.

For a single tool, use this format:
{{
    "tool": "tool-name",
    "arguments": {{
        "argument-name": "value"
    }}
}}

After receiving tool responses:
1. Transform the raw data into a natural, conversational response
2. Keep responses concise but informative
3. Focus on the most relevant information
4. Use appropriate context from the user's question
5. Avoid simply repeating the raw data

Please use only the tools that are explicitly defined above."""

        self.messages = [
            {
                "role": "system",
                "content": system_message
            }
        ]

    async def start_terminal(self) -> None:
        """Start the chat session in terminal mode."""
        logging.info("Starting chat in terminal mode")
        try:
            # Initialize servers
            if not await self.initialize_servers():
                return
            
            # Set up system message
            self.setup_system_message()
            
            print("\nWelcome to the AI Assistant. Type 'exit' or 'quit' to end the session.")
            
            while True:
                try:
                    user_input = input("\nYou: ").strip()
                    if user_input.lower() in ['quit', 'exit']:
                        logging.info("\nExiting...")
                        break

                    self.messages.append({"role": "user", "content": user_input})
                    
                    # Get LLM response
                    llm_response = self.llm_client.get_response(self.messages)
                    print(f"\nAssistant: {llm_response}")

                    # Process potential tool calls
                    result = await self.process_llm_response(llm_response)
                    
                    if result != llm_response:
                        self.messages.append({"role": "assistant", "content": llm_response})
                        self.messages.append({"role": "system", "content": result})
                        
                        # Get final response after tool execution
                        final_response = self.llm_client.get_response(self.messages)
                        print(f"\nFinal response: {final_response}")
                        self.messages.append({"role": "assistant", "content": final_response})
                    else:
                        self.messages.append({"role": "assistant", "content": llm_response})

                except KeyboardInterrupt:
                    logging.info("\nExiting...")
                    break
        
        finally:
            await self.cleanup_servers()


# GUI Components - only imported if GUI mode is used
try:
    from PySide6.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, 
                                QTextEdit, QLineEdit, QPushButton, QWidget, QLabel)
    from PySide6.QtCore import QObject, Signal, Slot, Qt, QTimer
    
    class SignalEmitter(QObject):
        update_chat = Signal(str, str)  # role, content
        tool_progress = Signal(int, int)  # progress, total
    
    class ChatSessionGUI(ChatSession):
        """GUI version of ChatSession."""
    
        def __init__(self, servers: List[Server], llm_client: LLMClient) -> None:
            super().__init__(servers, llm_client)
            self.signals = SignalEmitter()
    
        async def process_llm_response(self, llm_response: str) -> str:
            """Process the LLM response and execute tools if needed."""
            try:
                # Try to find all JSON tool calls in the response
                tool_calls = []
                current_pos = 0
                while current_pos < len(llm_response):
                    try:
                        # Find opening brace
                        start = llm_response.find('{', current_pos)
                        if start == -1:
                            break
                            
                        # Find closing brace
                        count = 1
                        pos = start + 1
                        while count > 0 and pos < len(llm_response):
                            if llm_response[pos] == '{':
                                count += 1
                            elif llm_response[pos] == '}':
                                count -= 1
                            pos += 1
                        
                        if count == 0:
                            # Extract and parse JSON
                            json_str = llm_response[start:pos]
                            tool_call = json.loads(json_str)
                            if isinstance(tool_call, dict) and "tool" in tool_call and "arguments" in tool_call:
                                tool_calls.append(tool_call)
                            current_pos = pos
                        else:
                            break
                    except json.JSONDecodeError:
                        current_pos += 1
                        continue
                
                if tool_calls:
                    results = []
                    for tool_call in tool_calls:
                        result = await self._execute_tool(tool_call)
                        results.append(result)
                    return "Tool execution results: " + json.dumps(results)
                
                return llm_response
            except Exception as e:
                logging.error(f"Error processing LLM response: {e}")
                return llm_response
    
        async def _execute_tool(self, tool_call: dict) -> str:
            """Execute a single tool call."""
            logging.info(f"Executing tool: {tool_call['tool']}")
            logging.info(f"With arguments: {tool_call['arguments']}")
            
            for server in self.servers:
                tools = await server.list_tools()
                if any(tool.name == tool_call["tool"] for tool in tools):
                    try:
                        result = await server.execute_tool(tool_call["tool"], tool_call["arguments"])
                        
                        if isinstance(result, dict) and 'progress' in result:
                            progress = result['progress']
                            total = result['total']
                            logging.info(f"Progress: {progress}/{total} ({(progress/total)*100:.1f}%)")
                            
                        return f"Tool execution result: {result}"
                    except Exception as e:
                        error_msg = f"Error executing tool: {str(e)}"
                        logging.error(error_msg)
                        return error_msg
            
            return f"No server found with tool: {tool_call['tool']}"
    
        async def send_message(self, user_input: str) -> None:
            """Process a user message and get response."""
            # Add user message to history
            self.messages.append({"role": "user", "content": user_input})
            self.signals.update_chat.emit("user", user_input)
            
            # Get LLM response
            llm_response = self.llm_client.get_response(self.messages)
            self.signals.update_chat.emit("assistant", llm_response)
            logging.info("\nAssistant: %s", llm_response)
    
            # Process potential tool calls
            result = await self.process_llm_response(llm_response)
            
            if result != llm_response:
                self.messages.append({"role": "assistant", "content": llm_response})
                self.messages.append({"role": "system", "content": result})
                
                # Get final response after tool execution
                final_response = self.llm_client.get_response(self.messages)
                self.signals.update_chat.emit("assistant", final_response)
                logging.info("\nFinal response: %s", final_response)
                self.messages.append({"role": "assistant", "content": final_response})
            else:
                self.messages.append({"role": "assistant", "content": llm_response})
    
    
    async def run_gui_mode():
        """Run the application in GUI mode."""
        app = QApplication(sys.argv)
        
        # Initialize configuration and servers
        config = Configuration()
        server_config = config.load_config('servers_config.json')
        servers = [Server(name, srv_config) for name, srv_config in server_config['mcpServers'].items()]
        llm_client = LLMClient(config.llm_api_key)
        
        # Create chat session
        chat_session = ChatSessionGUI(servers, llm_client)
        
        # Initialize servers
        servers_initialized = await chat_session.initialize_servers()
        if not servers_initialized:
            logging.error("Failed to initialize application. Exiting.")
            return
        
        # Set up system message with tools
        chat_session.setup_system_message()
        
        # Import the new MainWindow from main_gui
        from main_gui import MainWindow
        
        # Create and show the main window
        window = MainWindow(chat_session)
        window.show()
        
        # Set up cleanup on exit
        app.aboutToQuit.connect(lambda: asyncio.ensure_future(chat_session.cleanup_servers()))
        
        # Keep reference to prevent garbage collection
        app.chat_session = chat_session
        
        # Run the event loop manually to work with asyncio
        while True:
            app.processEvents()
            await asyncio.sleep(0.01)
            if not window.isVisible():
                break
        
        # Cleanup
        await chat_session.cleanup_servers()

except ImportError:
    # PySide6 not available
    pass


async def main() -> None:
    """Initialize and run the chat session in the appropriate mode."""
    import argparse
    parser = argparse.ArgumentParser(description='AI Chat Application')
    parser.add_argument('--gui', action='store_true', help='Run in GUI mode (requires PySide6)')
    args = parser.parse_args()
    
    if args.gui:
        try:
            # Check if PySide6 is available
            from PySide6.QtWidgets import QApplication
            logging.info("Starting GUI mode")
            await run_gui_mode()
        except ImportError:
            logging.error("PySide6 is not installed. Please install it with 'pip install pyside6' to use GUI mode.")
            logging.info("Falling back to terminal mode")
            # Fall back to terminal mode
            config = Configuration()
            server_config = config.load_config('servers_config.json')
            servers = [Server(name, srv_config) for name, srv_config in server_config['mcpServers'].items()]
            llm_client = LLMClient(config.llm_api_key)
            chat_session = ChatSession(servers, llm_client)
            await chat_session.start_terminal()
    else:
        # Terminal mode
        config = Configuration()
        server_config = config.load_config('servers_config.json')
        servers = [Server(name, srv_config) for name, srv_config in server_config['mcpServers'].items()]
        llm_client = LLMClient(config.llm_api_key)
        chat_session = ChatSession(servers, llm_client)
        await chat_session.start_terminal()

if __name__ == "__main__":
    asyncio.run(main())