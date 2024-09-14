import os, sys, re
from PySide6.QtCore import QFile, QIODevice

cur_path = os.path.abspath(os.path.dirname(__file__))
pathPattern = re.compile(r'^[a-zA-Z]:\\(?:[a-zA-Z0-9_]+\\)*[a-zA-Z0-9_]+\.[a-zA-Z]{1,}$|^/[^/]+(?:/[^/]+)*$')

def openUiFile(uiname : str) -> QFile:
    ui_file = QFile(os.path.join(cur_path, uiname))
    
    if not ui_file.open(QIODevice.ReadOnly):
        print(f"Cannot open {uiname}: {ui_file.errorString()}")
        sys.exit(-1)

    return ui_file
