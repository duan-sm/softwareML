## 自定义类 QmyFigureCanvas，父类QWidget
## 创建了FigureCanvas和NavigationToolbar，组成一个整体
## 便于可视化设计

##import numpy as np

from PyQt5.QtWidgets import QWidget
import numpy
import matplotlib as mpl
from matplotlib.figure import Figure
from PyQt5 import QtCore
from matplotlib.backends.backend_qt5agg import (FigureCanvasQTAgg as FigureCanvas,
                                                NavigationToolbar2QT as NavigationToolbar)

from PyQt5.QtWidgets import QVBoxLayout


class QmyFigureCanvas(QWidget):
    mouseMove = QtCore.pyqtSignal(numpy.float64, mpl.lines.Line2D)  # Custom trigger signals for interaction with the UI
    def __init__(self, parent=None, toolbarVisible=True, showHint=False):
        super().__init__(parent)

        self.figure = Figure()  # Public figure properties
        figCanvas = FigureCanvas(self.figure)  # To create a FigureCanvas object, you must pass a Figure object
        self.naviBar = NavigationToolbar(figCanvas, self)  # Public property naviBar

        # self.__changeActionLanguage()  # Change to Chinese

        actList = self.naviBar.actions()  # List of associated actions
        count = len(actList)  # Number of Actions
        self.__lastActtionHint = actList[count - 1]  # The last Action, coordinate prompt label
        self.__showHint = showHint  # Whether to display coordinate hints on the toolbar
        self.__lastActtionHint.setVisible(self.__showHint)  # Hide its original coordinate prompt
        self.__showToolbar = toolbarVisible  # Whether to display the toolbar
        self.naviBar.setVisible(self.__showToolbar)

        layout = QVBoxLayout(self)
        layout.addWidget(self.naviBar)  # Add a toolbar
        layout.addWidget(figCanvas)  # Adding a FigureCanvas Object
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # 鼠标滚轮缩放
        self.__cid = figCanvas.mpl_connect("scroll_event", self.do_scrollZoom)  # Support mouse wheel zoom
        self.__cid1 = figCanvas.mpl_connect("pick_event", self.do_series_pick)  # Support curve capture
        # self.__cid2 = figCanvas.mpl_connect("button_press_event",self.do_pressMouse) # Support mouse click
        self.__cid3 = figCanvas.mpl_connect("button_release_event", self.do_releaseMouse)  # Support mouse release
        self.__cid4 = figCanvas.mpl_connect("motion_notify_event", self.do_moveMouse)  # Support mouse movement
        self.mouseIsPress = False
        self.pickStatus = False


    ##=====Public interface functions
    def setToolbarVisible(self, isVisible=True):  ## Whether to display the toolbar
        self.__showToolbar = isVisible
        self.naviBar.setVisible(isVisible)

    def setDataHintVisible(self, isVisible=True):  ## Whether to display the last coordinate prompt label of the toolbar
        self.__showHint = isVisible
        self.__lastActtionHint.setVisible(isVisible)

    def redraw(self):  ## Redraw the curve, quick call
        self.figure.canvas.draw()

    def __changeActionLanguage(self):  ## Chinese toolbar
        actList = self.naviBar.actions()  # List of associated actions
        actList[0].setText("复位")  # Home
        actList[0].setToolTip("复位到原始视图")  # Reset original view

        actList[1].setText("回退")  # Back
        actList[1].setToolTip("回退前一视图")  # Back to previous view

        actList[2].setText("前进")  # Forward
        actList[2].setToolTip("前进到下一视图")  # Forward to next view

        actList[4].setText("平动")  # Pan
        actList[4].setToolTip("左键平移坐标轴，右键缩放坐标轴")  # Pan axes with left mouse, zoom with right

        actList[5].setText("缩放")  # Zoom
        actList[5].setToolTip("框选矩形框缩放")  # Zoom to rectangle

        actList[6].setText("子图")  # Subplots
        actList[6].setToolTip("设置子图")  # Configure subplots

        actList[7].setText("定制")  # Customize
        actList[7].setToolTip("定制图表参数")  # Edit axis, curve and image parameters

        actList[9].setText("保存")  # Save
        actList[9].setToolTip("保存图表")  # Save the figure

    def do_scrollZoom(self, event):  # Zooming with the mouse wheel
        ax = event.inaxes  # Generate event axes object
        if ax == None:
            return

        self.naviBar.push_current()  # Push the current view limits and position onto the stack，This will restore
        xmin, xmax = ax.get_xbound()
        xlen = xmax - xmin
        ymin, ymax = ax.get_ybound()
        ylen = ymax - ymin

        xchg = event.step * xlen / 20  # step [scalar],positive = ’up’, negative ='down'
        xmin = xmin + xchg
        xmax = xmax - xchg
        ychg = event.step * ylen / 20
        ymin = ymin + ychg
        ymax = ymax - ychg

        ax.set_xbound(xmin, xmax)
        ax.set_ybound(ymin, ymax)
        event.canvas.draw()


    def do_series_pick(self, event):  # The picker event gets the captured curve
        self.series = event.artist
        # index = event.ind[0]
        # print("series",event.ind)
        if isinstance(self.series, mpl.lines.Line2D):
            self.pickStatus = True


    def do_releaseMouse(self, event):  # Mouse release, release grab curve
        if event.inaxes == None:
            return
        if self.pickStatus == True:
            self.series.set_color(color="black")
            self.figure.canvas.draw()
            self.pickStatus = False
        # self.mouseRelease.emit(event.xdata,event.ydata)


    def do_moveMouse(self, event):  # Move the mouse and redraw the grab curve
        if event.inaxes == None:
            return
        if self.pickStatus == True:
            self.series.set_xdata([event.xdata, event.xdata])
            self.series.set_color(color="red")
            self.figure.canvas.draw()
            self.mouseMove.emit(event.xdata, self.series)  # Custom trigger signals for interaction with the UI
