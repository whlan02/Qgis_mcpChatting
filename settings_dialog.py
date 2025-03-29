from PySide6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QComboBox, 
                               QPushButton, QLineEdit, QFormLayout)


class ModelConfig:
    PROVIDERS = {
        "OpenAI": ["gpt-4o", "o3-mini", "o1-mini", "gpt-4o-mini"],
        "Anthropic": ["Claude 3.7 Sonnet", "Claude 3.5 Sonnet", "Claude 3.5 Haiku"],
        "Google": ["Gemini 2.0 Flash", "Gemini 2.0 Flash-Lite"],
        "Deepseek": ["deepseek-chat", "deepseek-coder"]
    }

class SettingsDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Settings")
        self.setModal(True)
        self.setFixedWidth(400)
        self.setup_ui()
        
    def setup_ui(self):
        layout = QVBoxLayout(self)
        
        # Create form layout for settings
        form_layout = QFormLayout()
        
        # Provider selection
        self.provider_combo = QComboBox()
        self.provider_combo.addItems(ModelConfig.PROVIDERS.keys())
        self.provider_combo.currentTextChanged.connect(self.update_models)
        form_layout.addRow("Provider:", self.provider_combo)
        
        # Model selection
        self.model_combo = QComboBox()
        self.update_models(self.provider_combo.currentText())
        form_layout.addRow("Model:", self.model_combo)
        
        # API Key input
        self.api_key_input = QLineEdit()
        self.api_key_input.setEchoMode(QLineEdit.Password)
        form_layout.addRow("API Key:", self.api_key_input)
        
        layout.addLayout(form_layout)
        
        # Buttons
        button_layout = QHBoxLayout()
        save_button = QPushButton("Save")
        cancel_button = QPushButton("Cancel")
        
        save_button.clicked.connect(self.accept)
        cancel_button.clicked.connect(self.reject)
        
        button_layout.addWidget(save_button)
        button_layout.addWidget(cancel_button)
        layout.addLayout(button_layout)
        
        self.setLayout(layout)
        
    def update_models(self, provider):
        self.model_combo.clear()
        self.model_combo.addItems(ModelConfig.PROVIDERS[provider])
        
    def get_settings(self):
        return {
            "provider": self.provider_combo.currentText(),
            "model": self.model_combo.currentText(),
            "api_key": self.api_key_input.text()
        } 