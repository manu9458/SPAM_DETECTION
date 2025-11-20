import os

def create_project_structure(root_dir="."): # Changed default to "." (current directory)
    """
    Creates the standardized MLOps project folder structure in the current directory.
    """
    
    # Use os.getcwd() to show the actual path where files are being created
    print(f"Creating project structure in: {os.getcwd()}")

    # Define the core directory structure
    dirs = [
        f"{root_dir}/data/raw",
        f"{root_dir}/data/processed",
        f"{root_dir}/config",
        f"{root_dir}/src",
        f"{root_dir}/models",
        f"{root_dir}/logs",
        f"{root_dir}/tests" # New directory for EDA and quick validation
    ]

    # Create all directories
    for dir_path in dirs:
        # os.path.join is safer, but since we use f-strings with explicit '/', 
        # os.makedirs handles the relative path correctly.
        os.makedirs(dir_path, exist_ok=True)
        print(f"Created directory: {dir_path}")

    # Define core files to create
    core_files = [
        f"{root_dir}/requirements.txt",
        f"{root_dir}/.gitignore",
        f"{root_dir}/main.py",
        f"{root_dir}/config/config.yaml",
        f"{root_dir}/src/__init__.py",
        f"{root_dir}/src/data_loader.py",
        f"{root_dir}/src/preprocessor.py",
        f"{root_dir}/src/model_trainer.py",
        f"{root_dir}/src/evaluator.py",
        f"{root_dir}/tests/eda_notebook_template.ipynb",
        f"{root_dir}/tests/quick_test.py",
        f"{root_dir}/tests/data_validation.ipynb",
    ]

    # Create placeholder files
    for file_path in core_files:
        # Remove the leading './' if present for clean output display, but keep it in the path
        display_path = file_path.lstrip('./')
        
        if not os.path.exists(file_path):
            with open(file_path, 'w') as f:
                # Add basic content to key files to start
                if 'config.yaml' in file_path:
                    f.write("# Configuration parameters for the Spam Classifier project\n")
                elif 'requirements.txt' in file_path:
                    f.write("pandas\nscikit-learn\npyyaml\ntqdm\n")
                elif '.gitignore' in file_path:
                    f.write("# Python\n*.pyc\n__pycache__/\n\n# Data\ndata/processed/\nmodels/\nlogs/\n\n# Virtual Environment\nvenv/\n*.venv\n")
                elif 'main.py' in file_path:
                    f.write("import os\n# MLOps pipeline runner\n")
                
            print(f"Created file: {display_path}")

    print("\nProject structure setup complete in the current directory. Ready for development.")
    print("Next step: Add your 'spam.csv' file to the 'data/raw' folder.")

if __name__ == "__main__":
    create_project_structure()