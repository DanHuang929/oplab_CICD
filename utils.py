import sys
import traceback


def getexception(e):
    error_class = e.__class__.__name__  # 取得錯誤類型
    detail = e.args[0]  # 取得詳細內容
    cl, exc, tb = sys.exc_info()  # 取得Call Stack
    lastCallStack = traceback.extract_tb(tb)[-1]  # 取得Call Stack的最後一筆資料
    fileName = lastCallStack[0]  # 取得發生的檔案名稱
    lineNum = lastCallStack[1]  # 取得發生的行號
    funcName = lastCallStack[2]  # 取得發生的函數名稱

    errMsg = "File \"{}\", line {}, in {}: [{}] {}".format(fileName, lineNum, funcName, error_class, detail)
    print(errMsg)


def raise_warning(result, method):
    avg = sum(result) / len(result)
    if method == 'avg':
        return avg > 0.5
    elif method == 'max':
        return max(result) > 0.5
    elif method == 'min':
        return min(result) > 0.5
    elif method == 'best':
        return avg > 0.5
    return None