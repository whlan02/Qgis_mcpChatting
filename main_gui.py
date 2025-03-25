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
from PySide6.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, 
                              QTextEdit, QLineEdit, QPushButton, QWidget, QLabel)
from PySide6.QtCore import QObject, Signal, Slot, Qt, QTimer

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Import your existing classes
from main import Configuration, Server, Tool, LLMClient

class SignalEmitter(QObject):
    update_chat = Signal(str, str)  # role, content
    tool_progress = Signal(int, int)  # progress, total

class ChatSessionGUI(QObject):
    """Orchestrates the interaction between user, LLM, and tools with GUI integration."""

    def __init__(self, servers: List[Server], llm_client: LLMClient, signal_emitter: SignalEmitter) -> None:
        super().__init__()
        self.servers: List[Server] = servers
        self.llm_client: LLMClient = llm_client
        self.signals = signal_emitter
        self.messages = []
        self.all_tools = []

    async def cleanup_servers(self) -> None:
        """Clean up all servers properly."""
        cleanup_tasks = []
        for server in self.servers:
            cleanup_tasks.append(asyncio.create_task(server.cleanup()))
        
        if cleanup_tasks:
            try:
                await asyncio.gather(*cleanup_tasks, return_exceptions=True)
            except Exception as e:
                logging.warning(f"Warning during final cleanup: {e}")

    async def process_llm_response(self, llm_response: str) -> str:
        """Process the LLM response and execute tools if needed."""
        import json
        try:
            tool_call = json.loads(llm_response)
            if "tool" in tool_call and "arguments" in tool_call:
                logging.info(f"Executing tool: {tool_call['tool']}")
                logging.info(f"With arguments: {tool_call['arguments']}")
                
                # Display the tool execution in the chat
                self.signals.update_chat.emit("assistant", f"Executing tool: {tool_call['tool']}")
                
                for server in self.servers:
                    tools = await server.list_tools()
                    if any(tool.name == tool_call["tool"] for tool in tools):
                        try:
                            result = await server.execute_tool(tool_call["tool"], tool_call["arguments"])
                            
                            if isinstance(result, dict) and 'progress' in result:
                                progress = result['progress']
                                total = result['total']
                                self.signals.tool_progress.emit(progress, total)
                                logging.info(f"Progress: {progress}/{total} ({(progress/total)*100:.1f}%)")
                                
                            return f"Tool execution result: {result}"
                        except Exception as e:
                            error_msg = f"Error executing tool: {str(e)}"
                            logging.error(error_msg)
                            return error_msg
                
                return f"No server found with tool: {tool_call['tool']}"
            return llm_response
        except json.JSONDecodeError:
            return llm_response

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
Choose the appropriate tool based on the user's question. If no tool is needed, reply directly.

IMPORTANT: When you need to use a tool, you must ONLY respond with the exact JSON object format below, nothing else:
{{
    "tool": "tool-name",
    "arguments": {{
        "argument-name": "value"
    }}
}}

After receiving a tool's response:
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


class MainWindow(QMainWindow):
    def __init__(self, chat_session):
        super().__init__()
        self.chat_session = chat_session
        self.init_ui()
        
    def init_ui(self):
        self.setWindowTitle("AI Chat Interface")
        self.setMinimumSize(800, 600)
        
        # Main widget and layout
        main_widget = QWidget()
        main_layout = QVBoxLayout(main_widget)
        
        # Chat display area
        self.chat_display = QTextEdit()
        self.chat_display.setReadOnly(True)
        main_layout.addWidget(self.chat_display)
        
        # Progress bar (initially hidden)
        progress_layout = QHBoxLayout()
        self.progress_label = QLabel("Tool execution progress:")
        self.progress_label.setVisible(False)
        progress_layout.addWidget(self.progress_label)
        main_layout.addLayout(progress_layout)
        
        # Input area
        input_layout = QHBoxLayout()
        self.input_field = QLineEdit()
        self.input_field.setPlaceholderText("Type your message here...")
        self.input_field.returnPressed.connect(self.send_message)
        self.send_button = QPushButton("Send")
        self.send_button.clicked.connect(self.send_message)
        
        input_layout.addWidget(self.input_field)
        input_layout.addWidget(self.send_button)
        main_layout.addLayout(input_layout)
        
        self.setCentralWidget(main_widget)
        
        # Connect signals
        self.chat_session.signals.update_chat.connect(self.update_chat_display)
        self.chat_session.signals.tool_progress.connect(self.update_progress)
        
        # Welcome message
        self.update_chat_display("system", "Welcome to the AI Assistant. How can I help you today?")
    
    @Slot(str, str)
    def update_chat_display(self, role, content):
        """Update the chat display with new messages."""
        if role == "user":
            self.chat_display.append(f"<p style='color:#0000FF'><b>You:</b> {content}</p>")
        elif role == "assistant":
            self.chat_display.append(f"<p style='color:#008000'><b>Assistant:</b> {content}</p>")
        elif role == "system":
            self.chat_display.append(f"<p style='color:#800000'><b>System:</b> {content}</p>")
        
        # Scroll to bottom
        scrollbar = self.chat_display.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())
    
    @Slot(int, int)
    def update_progress(self, progress, total):
        """Update the progress bar."""
        percentage = int((progress / total) * 100)
        self.progress_label.setText(f"Tool execution progress: {progress}/{total} ({percentage}%)")
        self.progress_label.setVisible(True)
        
        # Hide progress after 5 seconds
        QTimer.singleShot(5000, lambda: self.progress_label.setVisible(False))
    
    def send_message(self):
        """Send the message from the input field."""
        user_input = self.input_field.text().strip()
        if not user_input:
            return
        
        self.input_field.clear()
        self.input_field.setEnabled(False)
        self.send_button.setEnabled(False)
        
        # Run async processing
        asyncio.ensure_future(self.process_message(user_input))
    
    async def process_message(self, user_input):
        """Process the message asynchronously."""
        await self.chat_session.send_message(user_input)
        
        # Re-enable input
        self.input_field.setEnabled(True)
        self.send_button.setEnabled(True)
        self.input_field.setFocus()


async def initialize_app():
    """Initialize the application and servers."""
    # Initialize your configuration and servers
    config = Configuration()
    server_config = config.load_config('servers_config.json')
    servers = [Server(name, srv_config) for name, srv_config in server_config['mcpServers'].items()]
    llm_client = LLMClient(config.llm_api_key)
    
    # Create signal emitter
    signal_emitter = SignalEmitter()
    
    # Create chat session
    chat_session = ChatSessionGUI(servers, llm_client, signal_emitter)
    
    # Initialize servers
    servers_initialized = await chat_session.initialize_servers()
    if not servers_initialized:
        return None
    
    # Set up system message with tools
    chat_session.setup_system_message()
    
    return chat_session


def main():
    """Main application entry point."""
    app = QApplication(sys.argv)
    
    # Create event loop
    loop = asyncio.get_event_loop()
    
    # Initialize chat session
    chat_session = loop.run_until_complete(initialize_app())
    if not chat_session:
        logging.error("Failed to initialize application. Exiting.")
        return
    
    # Create and show the main window
    window = MainWindow(chat_session)
    window.show()
    
    # Set up cleanup on exit
    app.aboutToQuit.connect(lambda: loop.run_until_complete(chat_session.cleanup_servers()))
    
    # Run the application
    sys.exit(app.exec())


if __name__ == "__main__":
    main()