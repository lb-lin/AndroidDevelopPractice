# 构建CameraX应用

## 实验目的

1. 掌 握 Android CameraX 拍照功能的基本用法 .由于CameraX是开发智能应用的必要组件，本次实验十分必要。
2. 掌握Android CameraX 视频捕捉功能的基本用法
3. 进一步熟悉Kotlin语言的特性

## 过程

### 根据教程完成代码部分

请求权限

![Untitled](%E6%9E%84%E5%BB%BACameraX%E5%BA%94%E7%94%A8%20f048584c915b4ba9a810bb1f4cf20ef5/Untitled.png)

### 完成startCamera,takePhoto,captureVideo等方法

![Untitled](%E6%9E%84%E5%BB%BACameraX%E5%BA%94%E7%94%A8%20f048584c915b4ba9a810bb1f4cf20ef5/Untitled%201.png)

### 发现按照教程完成拍摄视频的功能后，拍照的功能就失效了，按下TAKEPHOTO直接闪退

将`cameraProvider.bindToLifecycle`  修改如下解决，此时可以同时拍照和录视频或者在录视频的过程中拍照

```kotlin
cameraProvider.bindToLifecycle(
                    this, cameraSelector, preview, imageCapture,videoCapture)
```

## 实验结果

### 软件主界面：

![Untitled](%E6%9E%84%E5%BB%BACameraX%E5%BA%94%E7%94%A8%20f048584c915b4ba9a810bb1f4cf20ef5/Untitled.jpeg)

### 点击TAKE PHOTO按钮，拍照并保存到相册中

![Untitled](%E6%9E%84%E5%BB%BACameraX%E5%BA%94%E7%94%A8%20f048584c915b4ba9a810bb1f4cf20ef5/Untitled%201.jpeg)

### 点击START CAPTURE,开始录制视频，START CAPTURE按钮变为STOP CAPTURE

![Untitled](%E6%9E%84%E5%BB%BACameraX%E5%BA%94%E7%94%A8%20f048584c915b4ba9a810bb1f4cf20ef5/Untitled%202.jpeg)

### 结束录制，将视频保存到相册

![Untitled](%E6%9E%84%E5%BB%BACameraX%E5%BA%94%E7%94%A8%20f048584c915b4ba9a810bb1f4cf20ef5/Untitled%203.jpeg)

### 相册中的CameraX-Image和CameraX-Video文件夹出现了刚刚拍摄的照片和录制的视频

![Untitled](%E6%9E%84%E5%BB%BACameraX%E5%BA%94%E7%94%A8%20f048584c915b4ba9a810bb1f4cf20ef5/Untitled%204.jpeg)

![Untitled](%E6%9E%84%E5%BB%BACameraX%E5%BA%94%E7%94%A8%20f048584c915b4ba9a810bb1f4cf20ef5/Untitled%205.jpeg)