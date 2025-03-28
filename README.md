# AI Chat Interface Project

This README guide will help you set up and run the AI Chat Interface project, which provides a GUI-based chat interface with tool integration capabilities.

## Prerequisites

- Python 3.8 or higher
- pip (Python package installer)
- Node.js and npm (for NPX commands)
- uv (Python package installer and environment manager) for the QGIS server

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/ai-chat-interface.git
cd ai-chat-interface
```

### 2. Set Up a Virtual Environment (Recommended)

```bash
python -m venv venv

# Activate the virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### 3. Install Required Python Packages

```bash
pip install requests python-dotenv mcp

# For GUI mode:
pip install PySide6
```

### 4. Set Up the Environment Variables

Create a `.env` file in the project root directory with your API key: 

```
LLM_API_KEY="your_openai_api_key_here"
```

### 5. Configure Servers

Update the `servers_config.json` file to point to the correct directories for your system:

```json
{
  "mcpServers": {
      "filesystem": {
          "command": "npx",
          "args": [
              "-y",
              "@modelcontextprotocol/server-filesystem",
              "/path/to/your/Desktop",
              "/path/to/your/Downloads"
          ]
      },
      "qgis": {
          "command": "uv",
          "args": [
              "--directory",
              "/path/to/qgis_mcp/src/qgis_mcp",
              "run",
              "qgis_mcp_server.py"
          ]
      }
  }
}
```

## Usage

### Run the Application

Run the application in GUI mode with:

```bash
python main.py
```

If you want to run the application in terminal mode, you can use:

```bash
python main.py --terminal
```

## Project Structure

- `main.py` - The main application file with GUI mode support
- `servers_config.json` - Configuration file for MCP servers
- `.env` - Environment variables (API keys)

## Features

- Chat with an AI assistant powered by OpenAI's models
- Execute tools through Model Context Protocol (MCP) servers
- File system operations via the filesystem MCP server
- QGIS integration via the QGIS MCP server
- Progress tracking for tool execution
- Intuitive GUI interface with PySide6 (Qt)

## Troubleshooting

### Common Issues

1. **ModuleNotFoundError**: Make sure all required packages are installed
   ```bash
   pip install requests python-dotenv mcp PySide6
   ```

2. **API Key Error**: Ensure your `.env` file contains the correct API key
   ```
   LLM_API_KEY="your_openai_api_key_here"
   ```

3. **Server Configuration Errors**: Check that the paths in `servers_config.json` are correct for your system

4. **GUI Mode Not Working**: Make sure PySide6 is installed
   ```bash
   pip install PySide6
   ```

## Additional Setup for QGIS Server

If you're using the QGIS MCP server:

1. Install `uv` if not already installed:
   ```bash
   pip install uv
   ```

2. Set up the QGIS MCP repository:
   ```bash
   git clone https://github.com/path/to/qgis_mcp.git
   cd qgis_mcp
   pip install -e .
   ```

3. Update the path in `servers_config.json` to point to your QGIS MCP installation.