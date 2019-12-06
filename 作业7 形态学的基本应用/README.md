# 形态学的基本应用

## 连通分量提取

原图

![](img/7-1.png)

不同连通分量用不同的灰度值表示

![](img/7-2.png)

```python
def LabelConnRgn(src):  
    se = np.array(  
         [[1,1,1],  
          [1,1,1],  
          [1,1,1]], dtype = int)  
    nConnRgn = 255  
    dst = np.zeros(src.shape, src.dtype)  
    for i in range(src.shape[0]):  
        for j in range(src.shape[1]):  
            if src[i][j] == 255:  
                dst[i][j] = 255  
                while True:  
                    temp = dst.copy()  
                    dilate(dst, se, dst)  
                    dst = dst & src  
                    if (dst == temp).all():  
                        break  
                for k in range(src.shape[0]):  
                    for l in range(src.shape[1]):  
                        if dst[k][l] == 255:  
                            src[k][l] = nConnRgn  
                nConnRgn -= 40  
    return src  
```

## 凸壳

原图

![](img/7-1.png)

凸壳

![](img/7-3.png)

```python
def thresh_callback(_src):  
    src = _src.copy()  
    _dst = cv2.threshold(src, 100, 255, cv2.THRESH_BINARY)  
    dst = cv2.findContours(_dst[1], cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  
    hull = []  
    for i in range(len(dst[1])):  
        hull.append(cv2.convexHull(dst[1][i]))  
    drawing = np.zeros(_dst[1].shape)  
    for i in range(len(dst[1])):  
        color = 255  
        cv2.drawContours(drawing, dst[1], i, color)  
        cv2.drawContours(drawing, hull, i, color)  
    return drawing  
```

## 骨架提取和细化（表示不清楚两者区别在哪）

原图

![](img/7-1.png)

细化

![](img/7-4.png)

```python
def dilation(src):  
    se = np.array(  
         [[1,1,1,1,1],  
          [1,1,1,1,1],  
          [1,1,1,1,1],  
          [1,1,1,1,1],  
          [1,1,1,1,1]], dtype = int)  
    return dilate(src, se)
```

## 粗化

原图

![](img/7-1.png)

粗化

![](img/7-5.png)

```python
def erosion(src):  
    se = np.array(  
         [[1,1,1,1,1],  
          [1,1,1,1,1],  
          [1,1,1,1,1],  
          [1,1,1,1,1],  
          [1,1,1,1,1]], dty  
# 代码丢了
```

## 其他

一些形态学上的基本操作, 包括膨胀, 腐蚀, 还有开闭

```python
def dilate(_src, se):  
    src = _src.copy()  
    dst = _src.copy()  
    for i in range(se.shape[0] // 2, src.shape[0] - se.shape[0] // 2):  
        for j in range(se.shape[1] // 2, src.shape[1] - se.shape[1] // 2):  
            if src[i][j] == 255:  
                for p in range(se.shape[0]):  
                    for q in range(se.shape[1]):  
                        if se[p][q] == 1:  
                            dst[i - se.shape[0] // 2 + p][j - se.shape[1] // 2 + q] = 255                         
    return dst  
def erode(_src, se):  
    src = _src.copy()  
    dst = _src.copy()  
    ntemp = 0  
    for i in range(3):  
        for j in range(3):  
            if se[i][j] == 1:  
                ntemp += 1  
        continue  
    for i in range(src.shape[0] - 1):  
        for j in range(src.shape[1] - 1):  
            cnt = 0  
            for p in range(3):  
                for q in range(3):  
                    if se[p][q] == -1:  
                        continue  
                    elif se[p][q] == 1:  
                        if src[i-1+p][j-1+q] != 0:  
                            cnt += 1  
                    elif se[p][q] == 0:  
                        if src[i-1+p][j-1+q] != 255:  
                            cnt += 1  
                continue  
            if ntemp != cnt:  
                dst[i][j] = 0  
    return dst  
def open(src, se):  
    return dilate(erode(src, se), se)  
def close(src, se):  
    return erode(dilate(src, se), se)  
```