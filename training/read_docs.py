from docutils.core import publish_parts
from bs4 import BeautifulSoup
import os

# Paths to your source .rst directories
paths = [
    "/Users/moritzdenk/Geoinformatics/Uni/ifgi_hack/mcpchatting/training/docs/user_manual",
    "/Users/moritzdenk/Geoinformatics/Uni/ifgi_hack/mcpchatting/training/docs/training_manual"
]

# Destination path to store converted .txt files
output_base = "/Users/moritzdenk/Geoinformatics/Uni/ifgi_hack/mcpchatting/training/converted_txt"

def rst_to_text(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        rst_content = f.read()
    try:
        # Docutils will ignore unknown roles by default; we just avoid crashing
        overrides = {
            'report_level': 5,  # Suppress errors/warnings
            'halt_level': 6,    # Never halt
            'input_encoding': 'unicode',
            'output_encoding': 'unicode',
            'strip_comments': True
        }
        html_parts = publish_parts(source=rst_content, writer_name='html', settings_overrides=overrides)
        soup = BeautifulSoup(html_parts['html_body'], 'html.parser')
        return soup.get_text()
    except Exception as e:
        print(f"Failed to convert {file_path}: {e}")
        return None

def convert_all_rst_to_text(source_dirs, output_root):
    for source_dir in source_dirs:
        for root, dirs, files in os.walk(source_dir):
            for file in files:
                if file.endswith('.rst'):
                    full_path = os.path.join(root, file)
                    relative_path = os.path.relpath(full_path, source_dir)
                    
                    # Create equivalent output path
                    output_path = os.path.join(output_root, os.path.basename(source_dir), os.path.splitext(relative_path)[0] + '.txt')
                    os.makedirs(os.path.dirname(output_path), exist_ok=True)
                    
                    # Convert and save
                    print(f"Converting: {full_path}")
                    text = rst_to_text(full_path)
                    if text:
                        with open(output_path, 'w', encoding='utf-8') as out_file:
                            out_file.write(text)

# Run it
convert_all_rst_to_text(paths, output_base)