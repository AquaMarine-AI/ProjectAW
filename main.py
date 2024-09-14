import sys, jsCommons
from initWindow import InitWindow
from PySide6.QtWidgets import QApplication
from PySide6.QtGui import QIcon

if __name__ == "__main__" :
    #QApplication : 프로그램을 실행시켜주는 클래스
    app = QApplication(sys.argv)
    app.setWindowIcon(QIcon(jsCommons.cur_path + "/window_icon.png"))

    #initWindow MainWindow 인스턴스 생성, 띄우기
    initWindow = InitWindow()
    initWindow.show()


    #프로그램을 이벤트 루프로 진입시키기
    sys.exit(app.exec())