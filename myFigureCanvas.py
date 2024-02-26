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
    mouseMove = QtCore.pyqtSignal(numpy.float64, mpl.lines.Line2D)  # 自定义触发信号，用于与UI交互
    def __init__(self, parent=None, toolbarVisible=True, showHint=False):
        super().__init__(parent)

        self.figure = Figure()  # 公共的figure属性
        figCanvas = FigureCanvas(self.figure)  # 创建FigureCanvas对象，必须传递一个Figure对象
        self.naviBar = NavigationToolbar(figCanvas, self)  # 公共属性naviBar

        self.__changeActionLanguage()  # 改为汉语

        actList = self.naviBar.actions()  # 关联的Action列表
        count = len(actList)  # Action的个数
        self.__lastActtionHint = actList[count - 1]  # 最后一个Action,坐标提示标签
        self.__showHint = showHint  # 是否在工具栏上显示坐标提示
        self.__lastActtionHint.setVisible(self.__showHint)  # 隐藏其原有的坐标提示
        self.__showToolbar = toolbarVisible  # 是否显示工具栏
        self.naviBar.setVisible(self.__showToolbar)

        layout = QVBoxLayout(self)
        layout.addWidget(self.naviBar)  # 添加工具栏
        layout.addWidget(figCanvas)  # 添加FigureCanvas对象
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # 鼠标滚轮缩放
        self.__cid = figCanvas.mpl_connect("scroll_event", self.do_scrollZoom)  # 支持鼠标滚轮缩放
        self.__cid1 = figCanvas.mpl_connect("pick_event", self.do_series_pick)  # 支持曲线抓取
        # self.__cid2 = figCanvas.mpl_connect("button_press_event",self.do_pressMouse)#支持鼠标按下
        self.__cid3 = figCanvas.mpl_connect("button_release_event", self.do_releaseMouse)  # 支持鼠标释放
        self.__cid4 = figCanvas.mpl_connect("motion_notify_event", self.do_moveMouse)  # 支持鼠标移动
        self.mouseIsPress = False
        self.pickStatus = False


    ##=====公共接口函数
    def setToolbarVisible(self, isVisible=True):  ##是否显示工具栏
        self.__showToolbar = isVisible
        self.naviBar.setVisible(isVisible)

    def setDataHintVisible(self, isVisible=True):  ##是否显示工具栏最后的坐标提示标签
        self.__showHint = isVisible
        self.__lastActtionHint.setVisible(isVisible)

    def redraw(self):  ##重绘曲线,快捷调用
        self.figure.canvas.draw()

    def __changeActionLanguage(self):  ##汉化工具栏
        actList = self.naviBar.actions()  # 关联的Action列表
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

    def do_scrollZoom(self, event):  # 通过鼠标滚轮缩放
        ax = event.inaxes  # 产生事件axes对象
        if ax == None:
            return

        self.naviBar.push_current()  # Push the current view limits and position onto the stack，这样才可以还原
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


    def do_series_pick(self, event):  # picker事件获取抓取曲线
        self.series = event.artist
        # index = event.ind[0]
        # print("series",event.ind)
        if isinstance(self.series, mpl.lines.Line2D):
            self.pickStatus = True


    def do_releaseMouse(self, event):  # 鼠标释放，释放抓取曲线
        if event.inaxes == None:
            return
        if self.pickStatus == True:
            self.series.set_color(color="black")
            self.figure.canvas.draw()
            self.pickStatus = False
        # self.mouseRelease.emit(event.xdata,event.ydata)


    def do_moveMouse(self, event):  # 鼠标移动，重绘抓取曲线
        if event.inaxes == None:
            return
        if self.pickStatus == True:
            self.series.set_xdata([event.xdata, event.xdata])
            self.series.set_color(color="red")
            self.figure.canvas.draw()
            self.mouseMove.emit(event.xdata, self.series)  # 自定义触发信号，用于与UI交互
