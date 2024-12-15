import os

cwd = os.getcwd()
SUBFOLDER_NAME = "resources"
TEXT_FILE_NAME = "Notes.txt"
PDF_FILE_NAME = "attention.pdf"
JSON_FILE_NAME = "posts.json"

TEXT_FILEPATH = os.path.join(cwd, SUBFOLDER_NAME, TEXT_FILE_NAME)
PDF_FILEPATH = os.path.join(cwd, SUBFOLDER_NAME, PDF_FILE_NAME)
JSON_FILEPATH = os.path.join(cwd, SUBFOLDER_NAME, JSON_FILE_NAME)
