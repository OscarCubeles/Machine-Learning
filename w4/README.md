# Work3-IML

## Team
- Elena Blanco Lopez
- Clàudia Boixader Garcia
- Óscar Cubeles Ollé
- Alba Fernández Coronado

## Set Up Environment
These sections show how to create a virtual environment for 
our script and how to install dependencies.

```bash
# 1. Open folder in terminal 
cd <root_folder_of_project>/ 

# 2. Create virtual env 
py -m venv venv/ 

# 3. Open virtual env 
venv\Scripts\activate

# 4. Install required dependencies 
pip install -r requirements.txt 

# 5. You can check if dependencies were installed by running the next 
# command, which should print a list of installed dependencies:
pip list 
```

## Run the Code
```bash
# 1. With the environment activated, run the main script
py main.py 
```
## Execute the Script

When you run the main script, the first thing that appears is a menu that asks the user to choose between the two datasets available for analysis: `hepatitis`, `mx` or `cmc`. This choice determines the dataset that the program will process.

After selecting the dataset, the main menu will appear, prompting the user to input the corresponding number for the mode they wish to execute. The options available are as follows:

```plaintext
---------------------------------------------------------
        Work 4 - Introduction to Machine Learning
---------------------------------------------------------

Type the name dataset to be analysed ('hepatitis'or 'breast'): [user input]
```
