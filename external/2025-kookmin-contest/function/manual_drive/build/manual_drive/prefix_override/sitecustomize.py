import sys
if sys.prefix == '/usr':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/home/xytron/xycar_ws/src/study/manual_drive/install/manual_drive'
