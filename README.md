# mcpchatting

A short, two days Hackathon #ifgiHACK25 project. [(Presentation Video)](https://www.youtube.com/watch?v=JkR3NgwAmRc)

mcpchatting is based on the following projects:
- [QGIS MCP](https://github.com/jjsantos01/qgis_mcp)
- [MCP Chatbot](https://github.com/3choff/mcp-chatbot)


The mcpchatting system has been enhanced with expanded language model support and a built-in chat interface. This integrated solution removes dependency on external chat platforms, providing users with unrestricted access(using their own API keys) without the constraints typically associated with free-plan services like Claude.
Moreover, it now uses Retrieval-Augmented Generation. It combines large language models (LLMs) with external knowledge sources (in this case QGIS docs) so that the LLM has more knowledge about e.g. QGIS functions. So far it was only tested with ChatGPT.

## Prerequisites

- Python 3.10 or newer
- pip (Python package installer)
- Node.js and npm (for NPX commands)
- uv (Python package installer and environment manager) for the QGIS server

## Installation

### 1. Start Generation Here
If you're on Mac, please install uv as
```bash
brew install uv
```

On Windows Powershell:
```bash
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```
Otherwise installation instructions are on their website:[Install uv](https://docs.astral.sh/uv/getting-started/installation/)

### 2. Qgis MCP Plugin
You need to copy the folder [qgis_mcp_plugin](/qgis_mcp_plugin/) and its content on your QGIS profile plugins folder.

You can get your profile folder in QGIS going to menu `Settings` -> `User profiles` -> `Open active profile folder` Then, go to `Python/plugins` and paste the folder `qgis_mcp_plugin`.

> On a Windows machine the plugins folder is usually located at:`C:\Users\USER\AppData\Roaming\QGIS\QGIS3\profiles\default\python\plugins`

> and on MacOS:
`~/Library/Application\ Support/QGIS/QGIS3/profiles/default/python/plugins` 

 Then close QGIS and open it again. Go to the menu option `Plugins` -> `Installing and Managing Plugins`, select the `All` tab and search for "QGIS MCP", then mark the QGIS MCP checkbox.


### 3. Clone the Repository

```bash
git clone https://github.com/whlan02/Qgis_mcpChatting
cd Qgis_mcpChatting
```

### 4. Set Up a Virtual Environment (Recommended)

```bash
python -m venv venv

# Activate the virtual environment

# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### 5. Install Required Python Packages

```bash
pip install -r requirements.txt
```



### 6. Configure Servers

Update the `servers_config_Example.json` file to point to the correct directories for your system:
```json
{
  "mcpServers": {
      "filesystem": {
          "command": "npx",
          "args": [
              "-y",
              "@modelcontextprotocol/server-filesystem",
              "/ABSOLUTE/PATH/TO/YOUR/WORKING/DIRECTORY" 
          ]
      },
      "qgis": {
          "command": "uv",
          "args": [
              "--directory",
              "/ABSOLUTE/PATH/TO/PARENT/REPO/FOLDER/qgis_mcp",
              "run",
              "qgis_mcp_server.py"
          ]
      }
  }
}
```
After you have configured the `servers_config_Example.json` file, you should rename it to `servers_config.json`. Otherwise, the QGIS MCP Server will not start.


## Usage

### 1. Start the QGIS MCP Server
1. In QGIS, go to `plugins` -> `QGIS MCP`-> `QGIS MCP`
    ![plugins menu](/images/Screenshot1.png)
2. Click "Start Server"
    ![start server](/images/Screenshot2.png)


### 2. Start the QGIS MCP Client

In the root folder of the project, run:

```bash
python main.py
```
### 3. Choose LLM Model and set API Key

![Settings](/images/Screenshot3.png)


Only OpenAI and Deepseek are tested so far.
