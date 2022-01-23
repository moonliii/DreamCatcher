#### 相似度算法

- Hu矩形状匹配
- cosine similarity

#### 前后端base64传图片

- 小程序图片转base64

  `wx.getFileSystemManager().readFileSync(filePath, 'base64') `

- Base64与opencv图像互转

  - 将普通opencv图像转换成base64：

    ```python
    def image_to_base64(image_np):
     
        image = cv2.imencode('.jpg',image_np)[1]
        image_code = str(base64.b64encode(image))[2:-1]
     
        return image_code
    ```

  - 将Bse64转换成opencv图像：

    ```python
    def base64_to_image(base64_code):
        # base64解码
        img_data = base64.b64decode(base64_code)
        # 转换为np数组
        img_array = np.fromstring(img_data, np.uint8)
        # 转换成opencv可用格式
        img = cv2.imdecode(img_array, cv2.COLOR_RGB2BGR)
        return img
    ```

  