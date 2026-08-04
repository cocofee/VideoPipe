import sys
import os
from PyQt5.QtWidgets import QApplication
from realtime.main_window import main as window_main

def main():
    # 转发给 main_window 的 main 函数，它处理了所有的初始化逻辑
    window_main()

if __name__ == "__main__":
    main()
