import os as _os
import socket

IS_LAPTOP = False
if socket.gethostname() == 'HAOYU-PAH':
    IS_LAPTOP = True

FIGURE_PATH = r'D:\OneDrive - 同济大学\4. 非专业课\5. 偏微分方程数值解\试验作业2\插图'
if IS_LAPTOP:
    FIGURE_PATH = r'C:\Users\wolf\OneDrive - 同济大学\4. 非专业课\5. 偏微分方程数值解\试验作业2\插图'
not _os.path.exists(FIGURE_PATH) and _os.makedirs(FIGURE_PATH)
