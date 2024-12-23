---
completed: true
version: 11.0.0
---
# Overview
The **Python Imaging Library** adds image processing capabilities to your Python interpreter.

This library provides extensive file format support, an efficient internal representation, and fairly powerful image processing capabilities.

The core image library is designed for fast access to data stored in a few basic pixel formats. It should provide a solid foundation for a general image processing tool.

Let’s look at a few possible uses of this library.

## Image Archives
The Python Imaging Library is ideal for image archival and batch processing applications. You can use the library to create thumbnails, convert between file formats, print images, etc.
>Python Imaging Library 非常适合图像归档和批处理应用
>你可以使用该库创建缩略图、在不同的文件格式之间转换、打印图像等

The current version identifies and reads a large number of formats. Write support is intentionally restricted to the most commonly used interchange and presentation formats.
>当前版本能够识别并读取大量格式
>出于有意的限制，写入支持仅限于最常用的交换和呈现格式

## Image Display
The current release includes Tk [`PhotoImage`](https://pillow.readthedocs.io/en/stable/reference/ImageTk.html#PIL.ImageTk.PhotoImage "PIL.ImageTk.PhotoImage") and [`BitmapImage`](https://pillow.readthedocs.io/en/stable/reference/ImageTk.html#PIL.ImageTk.BitmapImage "PIL.ImageTk.BitmapImage") interfaces, as well as a [`Windows DIB interface`](https://pillow.readthedocs.io/en/stable/reference/ImageWin.html#module-PIL.ImageWin "PIL.ImageWin") that can be used with PythonWin and other Windows-based toolkits. Many other GUI toolkits come with some kind of PIL support.
>PIL 当前版本的图像展示功能包括了 Tk 的 `PhotoImage` 和 `BitmapImage` 接口，以及一个可以与 PythonWin 和其他基于 Windows 的工具包一起使用的 `Windows DIB` 接口
>许多其他图形用户界面工具包都带有某种形式的 PIL 支持

For debugging, there’s also a [`show()`](https://pillow.readthedocs.io/en/stable/reference/Image.html#PIL.Image.Image.show "PIL.Image.Image.show") method which saves an image to disk, and calls an external display utility.
>此外，为了调试，还有一个 `show()` 方法，该方法会将图像保存到磁盘，并调用外部的显示实用程序

## Image Processing
The library contains basic image processing functionality, including point operations, filtering with a set of built-in convolution kernels, and colour space conversions.
>该库包含基本的图像处理功能，包括点操作、使用一组内置卷积核进行滤波以及颜色空间转换

The library also supports image resizing, rotation and arbitrary affine transforms.
>库还支持图像缩放、旋转和任意仿射变换

There’s a histogram method allowing you to pull some statistics out of an image. This can be used for automatic contrast enhancement, and for global statistical analysis.
>有一个直方图方法，可以让你从图像中提取一些统计信息，这可以用于自动对比度增强和全局统计分析

# Tutorial
## Using the Image class
The most important class in the Python Imaging Library is the [`Image`](https://pillow.readthedocs.io/en/stable/reference/Image.html#PIL.Image.Image "PIL.Image.Image") class, defined in the module with the same name. You can create instances of this class in several ways; either by loading images from files, processing other images, or creating images from scratch.
> `Image` 类定义于 `Image` 模块
> 支持通过：从文件中装载图片、处理其他图片、从零创建图片来构造 `Image` 对象

To load an image from a file, use the [`open()`](https://pillow.readthedocs.io/en/stable/reference/Image.html#PIL.Image.open "PIL.Image.open") function in the [`Image`](https://pillow.readthedocs.io/en/stable/reference/Image.html#module-PIL.Image "PIL.Image") module:

```python
>>> from PIL import Image
>>> im = Image.open("hopper.ppm")
```

If successful, this function returns an [`Image`](https://pillow.readthedocs.io/en/stable/reference/Image.html#PIL.Image.Image "PIL.Image.Image") object. 
> `Image` 模块的 `open()` 函数支持从文件中装载图片
> 返回 `Image` 对象

You can now use instance attributes to examine the file contents:

```python
>>> print(im.format, im.size, im.mode)
PPM (512, 512) RGB
```

The [`format`](https://pillow.readthedocs.io/en/stable/reference/Image.html#PIL.Image.Image.format "PIL.Image.Image.format") attribute identifies the source of an image. If the image was not read from a file, it is set to None. The `size` attribute is a 2-tuple containing width and height (in pixels). The [`mode`](https://pillow.readthedocs.io/en/stable/reference/Image.html#PIL.Image.Image.mode "PIL.Image.Image.mode") attribute defines the number and names of the bands in the image, and also the pixel type and depth. Common modes are “L” (luminance) for grayscale images, “RGB” for true color images, and “CMYK” for pre-press images.
> `Image` 对象的属性有：
> `format` - 图像的来源，如果图像不是从文件中读取，则为 None
> `size` - (width, height) (in pixels)
> `mode` - 定义了图像中的通道数量和名称，同时定义了像素类型和深度。常见的模式有 L (亮度) 用于灰度图像、RGB 用于真色彩图像、CMYK 用于预印刷图像

If the file cannot be opened, an [`OSError`](https://docs.python.org/3/library/exceptions.html#OSError "(in Python v3.13)") exception is raised.

Once you have an instance of the [`Image`](https://pillow.readthedocs.io/en/stable/reference/Image.html#PIL.Image.Image "PIL.Image.Image") class, you can use the methods defined by this class to process and manipulate the image. For example, let’s display the image we just loaded:

```python
>>> im.show()
```

![../_images/show_hopper.webp](https://pillow.readthedocs.io/en/stable/_images/show_hopper.webp)

Note
The standard version of [`show()`](https://pillow.readthedocs.io/en/stable/reference/Image.html#PIL.Image.Image.show "PIL.Image.Image.show") is not very efficient, since it saves the image to a temporary file and calls a utility to display the image. If you don’t have an appropriate utility installed, it won’t even work. When it does work though, it is very handy for debugging and tests.
> `Image` 对象的 `show()` 方法展示图像
> 其过程是将图像保存到临时文件，然后调用实用程序展示图像

The following sections provide an overview of the different functions provided in this library.

## Reading and writing images
The Python Imaging Library supports a wide variety of image file formats. To read files from disk, use the [`open()`](https://pillow.readthedocs.io/en/stable/reference/Image.html#PIL.Image.open "PIL.Image.open") function in the [`Image`](https://pillow.readthedocs.io/en/stable/reference/Image.html#module-PIL.Image "PIL.Image") module. You don’t have to know the file format to open a file. The library automatically determines the format based on the contents of the file.
> `Image` 模块的 `open()` 方法自动根据文件内容决定图片格式

To save a file, use the [`save()`](https://pillow.readthedocs.io/en/stable/reference/Image.html#PIL.Image.Image.save "PIL.Image.Image.save") method of the [`Image`](https://pillow.readthedocs.io/en/stable/reference/Image.html#PIL.Image.Image "PIL.Image.Image") class. When saving files, the name becomes important. Unless you specify the format, the library uses the filename extension to discover which file storage format to use.
> `Image` 类的 `save()` 方法将图片保存至文件，该方法通过文件名后缀来决定需要使用的文件存储格式

### Convert files to JPEG

```python
import os, sys
from PIL import Image

for infile in sys.argv[1:]:
    f, e = os.path.splitext(infile)
    outfile = f + ".jpg"
    if infile != outfile:
        try:
            with Image.open(infile) as im:
                im.save(outfile)
        except OSError:
            print("cannot convert", infile)
```

![../_images/hopper.jpg](https://pillow.readthedocs.io/en/stable/_images/hopper.jpg)

A second argument can be supplied to the [`save()`](https://pillow.readthedocs.io/en/stable/reference/Image.html#PIL.Image.Image.save "PIL.Image.Image.save") method which explicitly specifies a file format. If you use a non-standard extension, you must always specify the format this way:
> `save()` 也可以显式指定文件格式
> 如果文件名拓展不是标准拓展名，则需要显式指定文件格式

### Create JPEG thumbnails

```python
import os, sys
from PIL import Image

size = (128, 128)

for infile in sys.argv[1:]:
    outfile = os.path.splitext(infile)[0] + ".thumbnail"
    if infile != outfile:
        try:
            with Image.open(infile) as im:
                im.thumbnail(size)
                im.save(outfile, "JPEG")
        except OSError:
            print("cannot create thumbnail for", infile)
```

![../_images/thumbnail_hopper.jpg](https://pillow.readthedocs.io/en/stable/_images/thumbnail_hopper.jpg)

It is important to note that the library doesn’t decode or load the raster data unless it really has to. When you open a file, the file header is read to determine the file format and extract things like mode, size, and other properties required to decode the file, but the rest of the file is not processed until later.

This means that opening an image file is a fast operation, which is independent of the file size and compression type. 
>需要注意的是，除非确实需要，否则库不会解码或加载光栅数据
>当你打开一个文件时，PIL 会读取文件头以确定文件格式并提取诸如模式、大小和其他用于解码文件所需的属性，但文件的其余部分不会被立即处理
>因此打开图像文件是一个快速操作，它独立于文件大小和压缩类型

Here’s a simple script to quickly identify a set of image files:
### Identify Image Files

```python
import sys
from PIL import Image

for infile in sys.argv[1:]:
    try:
        with Image.open(infile) as im:
            print(infile, im.format, f"{im.size}x{im.mode}")
    except OSError:
        pass
```

## Cutting, pasting, and merging images
The [`Image`](https://pillow.readthedocs.io/en/stable/reference/Image.html#PIL.Image.Image "PIL.Image.Image") class contains methods allowing you to manipulate regions within an image. To extract a sub-rectangle from an image, use the [`crop()`](https://pillow.readthedocs.io/en/stable/reference/Image.html#PIL.Image.Image.crop "PIL.Image.Image.crop") method.

### Copying a subrectangle from an image

```python
box = (0, 0, 64, 64)
region = im.crop(box)
```

The region is defined by a 4-tuple, where coordinates are (left, upper, right, lower). The Python Imaging Library uses a coordinate system with (0, 0) in the upper left corner. Also note that coordinates refer to positions between the pixels, so the region in the above example is exactly 64x64 pixels.
> `crop()` 方法裁切一个长方形，需要给定左上角的像素坐标和右下角的像素坐标
> 坐标 (0, 0) 表示图片最左上角

The region could now be processed in a certain manner and pasted back.

![../_images/cropped_hopper.webp](https://pillow.readthedocs.io/en/stable/_images/cropped_hopper.webp)

### Processing a subrectangle, and pasting it back

```python
region = region.transpose(Image.Transpose.ROTATE_180)
im.paste(region, box)
```

When pasting regions back, the size of the region must match the given region exactly. In addition, the region cannot extend outside the image. However, the modes of the original image and the region do not need to match. If they don’t, the region is automatically converted before being pasted (see the section on [Color transforms](https://pillow.readthedocs.io/en/stable/handbook/tutorial.html#color-transforms) below for details).
> 裁切下来的对象可以通过 `paste` 方法被粘贴到指定的长方形位置
> 原图的模式和粘贴的区域图的模式并不需要一致，粘贴的区域图的模式会被自动转换以匹配原图

![../_images/pasted_hopper.webp](https://pillow.readthedocs.io/en/stable/_images/pasted_hopper.webp)

Here’s an additional example:

### Rolling an image

```python
def roll(im: Image.Image, delta: int) -> Image.Image:
    """Roll an image sideways."""
    xsize, ysize = im.size

    delta = delta % xsize
    if delta == 0:
        return im

    part1 = im.crop((0, 0, delta, ysize))
    part2 = im.crop((delta, 0, xsize, ysize))
    im.paste(part1, (xsize - delta, 0, xsize, ysize))
    im.paste(part2, (0, 0, xsize - delta, ysize))

    return im
```

![../_images/rolled_hopper.webp](https://pillow.readthedocs.io/en/stable/_images/rolled_hopper.webp)

Or if you would like to merge two images into a wider image:

### Merging images

```python
def merge(im1: Image.Image, im2: Image.Image) -> Image.Image:
    w = im1.size[0] + im2.size[0]
    h = max(im1.size[1], im2.size[1])
    im = Image.new("RGBA", (w, h))

    im.paste(im1)
    im.paste(im2, (im1.size[0], 0))

    return im
```

![../_images/merged_hopper.webp](https://pillow.readthedocs.io/en/stable/_images/merged_hopper.webp)

For more advanced tricks, the paste method can also take a transparency mask as an optional argument. In this mask, the value 255 indicates that the pasted image is opaque in that position (that is, the pasted image should be used as is). The value 0 means that the pasted image is completely transparent. Values in-between indicate different levels of transparency. For example, pasting an RGBA image and also using it as the mask would paste the opaque portion of the image but not its transparent background.
>`paste` 方法还可以接受一个透明度蒙版作为可选参数
>在这个蒙版中，值 255 表示在该位置粘贴的图像不透明（粘贴的图像保持原样），值 0 表示粘贴的图像是完全透明的，介于两者之间的值表示不同程度的透明度
>例如，粘贴一个 RGBA 图像并同时将其用作蒙版，将只粘贴图像的不透明部分而不是其透明背景

The Python Imaging Library also allows you to work with the individual bands of an multi-band image, such as an RGB image. The split method creates a set of new images, each containing one band from the original multi-band image. The merge function takes a mode and a tuple of images, and combines them into a new image. The following sample swaps the three bands of an RGB image:
>PIL 还允许你处理多通道图像（如 RGB 图像）的各个通道
> `split` 方法会创建一组新的图像，每张图像包含原始多通道图像的一个通道
> `merge` 函数接受一个模式和一个图像元组，并将它们合并成一张新图像
> 以下示例展示了如何交换 RGB 图像的三个通道：

```python
from PIL import Image

im = Image.open("example.jpg")
r, g, b = im.split()
rgba_image = Image.merge("RGBA", (g, b, r, im.getchannel('A')))
```

### Splitting and merging bands

```python
r, g, b = im.split()
im = Image.merge("RGB", (b, g, r))
```

Note that for a single-band image, [`split()`](https://pillow.readthedocs.io/en/stable/reference/Image.html#PIL.Image.Image.split "PIL.Image.Image.split") returns the image itself. To work with individual color bands, you may want to convert the image to “RGB” first.
> 对于单通道图像，`split()` 返回图像本身
> 如果要处理独立的颜色通道，我们需要先将图像转化为 RBG 模式

![../_images/rebanded_hopper.webp](https://pillow.readthedocs.io/en/stable/_images/rebanded_hopper.webp)

## Geometrical transforms
The [`PIL.Image.Image`](https://pillow.readthedocs.io/en/stable/reference/Image.html#PIL.Image.Image "PIL.Image.Image") class contains methods to [`resize()`](https://pillow.readthedocs.io/en/stable/reference/Image.html#PIL.Image.Image.resize "PIL.Image.Image.resize") and [`rotate()`](https://pillow.readthedocs.io/en/stable/reference/Image.html#PIL.Image.Image.rotate "PIL.Image.Image.rotate") an image. The former takes a tuple giving the new size, the latter the angle in degrees counter-clockwise.
> `resize()` 方法接受给定大小，对图片进行缩放
> `rotate()` 方法接受给定角度，对图片进行逆时针旋转

### Simple geometry transforms

```python
out = im.resize((128, 128))
out = im.rotate(45) # degrees counter-clockwise
```

![../_images/rotated_hopper_90.webp](https://pillow.readthedocs.io/en/stable/_images/rotated_hopper_90.webp)

To rotate the image in 90 degree steps, you can either use the [`rotate()`](https://pillow.readthedocs.io/en/stable/reference/Image.html#PIL.Image.Image.rotate "PIL.Image.Image.rotate") method or the [`transpose()`](https://pillow.readthedocs.io/en/stable/reference/Image.html#PIL.Image.Image.transpose "PIL.Image.Image.transpose") method. The latter can also be used to flip an image around its horizontal or vertical axis.
> 旋转90度可以用 `rotate()` 也可以用 `transpose()`
> `transpose()` 还可以用于沿着纵轴或者横轴翻转图像 (注意翻转和旋转略有不同)

### Transposing an image

```python
out = im.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
```

![../_images/flip_left_right_hopper.webp](https://pillow.readthedocs.io/en/stable/_images/flip_left_right_hopper.webp)

```python
out = im.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
```

![../_images/flip_top_bottom_hopper.webp](https://pillow.readthedocs.io/en/stable/_images/flip_top_bottom_hopper.webp)

```python
out = im.transpose(Image.Transpose.ROTATE_90)
```

![../_images/rotated_hopper_90.webp](https://pillow.readthedocs.io/en/stable/_images/rotated_hopper_90.webp)

```python
out = im.transpose(Image.Transpose.ROTATE_180)
```

![../_images/rotated_hopper_180.webp](https://pillow.readthedocs.io/en/stable/_images/rotated_hopper_180.webp)

```python
out = im.transpose(Image.Transpose.ROTATE_270)
```

![../_images/rotated_hopper_270.webp](https://pillow.readthedocs.io/en/stable/_images/rotated_hopper_270.webp)

`transpose(ROTATE)` operations can also be performed identically with [`rotate()`](https://pillow.readthedocs.io/en/stable/reference/Image.html#PIL.Image.Image.rotate "PIL.Image.Image.rotate") operations, provided the `expand` flag is true, to provide for the same changes to the image’s size.
>可以通过[`rotate()`](https://pillow.readthedocs.io/en/stable/reference/Image.html#PIL.Image.Image.rotate "PIL.Image.Image.rotate")操作来相同地执行 [`transpose(ROTATE)`](https://pillow.readthedocs.io/en/stable/) 操作，前提是设置了 `expand` 标志，以便对图像大小进行相同的更改

A more general form of image transformations can be carried out via the [`transform()`](https://pillow.readthedocs.io/en/stable/reference/Image.html#PIL.Image.Image.transform "PIL.Image.Image.transform") method.
>更通用的图像变换可以通过[`transform()`](https://pillow.readthedocs.io/en/stable/reference/Image.html#PIL.Image.Image.transform "PIL.Image.Image.transform")方法来进行

### Relative resizing
Instead of calculating the size of the new image when resizing, you can also choose to resize relative to a given size.
>在调整图像大小时，除了计算新图像的尺寸外，你还可以选择相对于给定的尺寸进行缩放
>
>- **ImageOps.contain()**: 该方法会调整图像大小，使其包含在指定的大小范围内，同时保持原始图像的宽高比不变。任何剩余的空间会被裁剪掉。
>- **ImageOps.cover()**: 该方法会调整图像的大小，使图像完全覆盖指定的大小范围，同时保持原始图像的宽高比。可能会出现裁剪的情况，因为图像的一部分可能会超出边界。
>- **ImageOps.fit()**: 该方法会调整图像大小，使其完全适应指定的大小范围，同时保持原始图像的宽高比。它不会裁剪图像，而是通过填充的方式来扩展图像，使得图像完全填满指定区域。
>- **ImageOps.pad()**: 该方法会调整图像大小，使其适应指定的大小范围，并且通过添加填充来保持原始图像的宽高比，从而确保图像完全填满指定区域，同时可以指定填充颜色。
>- **thumbnail()**: 该方法会生成一个缩小版的图像副本，并将图像尺寸调整为不超过指定的最大尺寸。它会直接修改原始图像对象，因此在保存时不需要再重新打开图像。注意，它不会改变原始图像的尺寸，而是生成一个较小的副本。

```python
from PIL import Image, ImageOps
size = (100, 150)
with Image.open("hopper.webp") as im:
    ImageOps.contain(im, size).save("imageops_contain.webp")
    ImageOps.cover(im, size).save("imageops_cover.webp")
    ImageOps.fit(im, size).save("imageops_fit.webp")
    ImageOps.pad(im, size, color="#f00").save("imageops_pad.webp")

    # thumbnail() can also be used,
    # but will modify the image object in place
    im.thumbnail(size)
    im.save("image_thumbnail.webp")
```

|                 | [`thumbnail()`](https://pillow.readthedocs.io/en/stable/reference/Image.html#PIL.Image.Image.thumbnail "PIL.Image.Image.thumbnail") | [`contain()`](https://pillow.readthedocs.io/en/stable/reference/ImageOps.html#PIL.ImageOps.contain "PIL.ImageOps.contain") | [`cover()`](https://pillow.readthedocs.io/en/stable/reference/ImageOps.html#PIL.ImageOps.cover "PIL.ImageOps.cover") | [`fit()`](https://pillow.readthedocs.io/en/stable/reference/ImageOps.html#PIL.ImageOps.fit "PIL.ImageOps.fit") | [`pad()`](https://pillow.readthedocs.io/en/stable/reference/ImageOps.html#PIL.ImageOps.pad "PIL.ImageOps.pad") |
| --------------- | ----------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------- |
| Given size      | `(100, 150)`                                                                                                                        | `(100, 150)`                                                                                                               | `(100, 150)`                                                                                                         | `(100, 150)`                                                                                                   | `(100, 150)`                                                                                                   |
| Resulting image | ![../_images/image_thumbnail.webp](https://pillow.readthedocs.io/en/stable/_images/image_thumbnail.webp)                            | ![../_images/imageops_contain.webp](https://pillow.readthedocs.io/en/stable/_images/imageops_contain.webp)                 | ![../_images/imageops_cover.webp](https://pillow.readthedocs.io/en/stable/_images/imageops_cover.webp)               | ![../_images/imageops_fit.webp](https://pillow.readthedocs.io/en/stable/_images/imageops_fit.webp)             | ![../_images/imageops_pad.webp](https://pillow.readthedocs.io/en/stable/_images/imageops_pad.webp)             |
| Resulting size  | `100×100`                                                                                                                           | `100×100`                                                                                                                  | `150×150`                                                                                                            | `100×150`                                                                                                      | `100×150`                                                                                                      |

## Color transforms
The Python Imaging Library allows you to convert images between different pixel representations using the [`convert()`](https://pillow.readthedocs.io/en/stable/reference/Image.html#PIL.Image.Image.convert "PIL.Image.Image.convert") method.
> `convert()` 方法用于将图片在不同的像素表示间转化

### Converting between modes

```python
from PIL import Image

with Image.open("hopper.ppm") as im:
    im = im.convert("L")
```

The library supports transformations between each supported mode and the “L” and “RGB” modes. To convert between other modes, you may have to use an intermediate image (typically an “RGB” image).
> PIL 支持将任意支持的模式转化为 L 或 RGB 模式，以及将 L 或 RGB 模式转化为任意支持的模式
> 故 L 或 RGB 模式可以作为两个不同模式的图片之间转化的中间模式

## Image enhancement
The Python Imaging Library provides a number of methods and modules that can be used to enhance images.

### Filters
The [`ImageFilter`](https://pillow.readthedocs.io/en/stable/reference/ImageFilter.html#module-PIL.ImageFilter "PIL.ImageFilter") module contains a number of pre-defined enhancement filters that can be used with the [`filter()`](https://pillow.readthedocs.io/en/stable/reference/Image.html#PIL.Image.Image.filter "PIL.Image.Image.filter") method.
> `filter()` 方法接受 `ImageFilter` 模块中预定义的增强滤波器作为参数，进行图像增强

#### Applying filters

```python
from PIL import ImageFilter
out = im.filter(ImageFilter.DETAIL)
```

![../_images/enhanced_hopper.webp](https://pillow.readthedocs.io/en/stable/_images/enhanced_hopper.webp)

### Point Operations
The [`point()`](https://pillow.readthedocs.io/en/stable/reference/Image.html#PIL.Image.Image.point "PIL.Image.Image.point") method can be used to translate the pixel values of an image (e.g. image contrast manipulation). In most cases, a function object expecting one argument can be passed to this method. Each pixel is processed according to that function:
>[`point()`](https://pillow.readthedocs.io/en/stable/reference/Image.html#PIL.Image.Image.point "PIL.Image.Image.point") 方法可用于转换图像的像素值（例如调整图像对比度）
>在大多数情况下，可以将接受一个参数的函数对象传递给此方法，每个像素根据该函数进行处理：

#### Applying point transforms

```python
# multiply each pixel by 20
out = im.point(lambda i: i * 20)
```

![../_images/transformed_hopper.webp](https://pillow.readthedocs.io/en/stable/_images/transformed_hopper.webp)

Using the above technique, you can quickly apply any simple expression to an image. You can also combine the [`point()`](https://pillow.readthedocs.io/en/stable/reference/Image.html#PIL.Image.Image.point "PIL.Image.Image.point") and [`paste()`](https://pillow.readthedocs.io/en/stable/reference/Image.html#PIL.Image.Image.paste "PIL.Image.Image.paste") methods to selectively modify an image:
> `point()` 和 `paste()` 可以结合以选择性地修改图像 (使用 `point` 选择出 mask，在 `paste` 中传入 mask)

#### Processing individual bands

```python
# split the image into individual bands
source = im.split()

R, G, B = 0, 1, 2

# select regions where red is less than 100
mask = source[R].point(lambda i: i < 100 and 255)

# process the green band
out = source[G].point(lambda i: i * 0.7)

# paste the processed band back, but only where red was < 100
source[G].paste(out, None, mask)

# build a new multiband image
im = Image.merge(im.mode, source)
```

Note the syntax used to create the mask:

```python
imout = im.point(lambda i: expression and 255)
```

![../_images/masked_hopper.webp](https://pillow.readthedocs.io/en/stable/_images/masked_hopper.webp)

Python only evaluates the portion of a logical expression as is necessary to determine the outcome, and returns the last value examined as the result of the expression. So if the expression above is false (0), Python does not look at the second operand, and thus returns 0. Otherwise, it returns 255.
>Python 仅评估逻辑表达式中必要的部分以确定结果，并将最后检查的值作为表达式的结果返回，因此，如果上面的 `expression`  为假（0），Python 不会查看第二个操作数，从而返回 0，否则，它返回 255
>( `and` 在第一个操作数为 `False` 时返回第一个操作数，即 `False/0`，在第一个操作数为 ` True ` 时返回第二个操作数 )
### Enhancement
For more advanced image enhancement, you can use the classes in the [`ImageEnhance`](https://pillow.readthedocs.io/en/stable/reference/ImageEnhance.html#module-PIL.ImageEnhance "PIL.ImageEnhance") module. Once created from an image, an enhancement object can be used to quickly try out different settings.
> 对于更高级的图像增强，可以使用 `ImageEnhance` 模块中的类
> 这些类的构造函数接受 `Image` 对象，构造出增强对象
> 从图像创建了增强对象之后，就可以快速尝试不同的设置

You can adjust contrast, brightness, color balance and sharpness in this way.
>可以用增强对象调整对比度、亮度、色彩平衡和锐度
#### Enhancing images

```python
from PIL import ImageEnhance

enh = ImageEnhance.Contrast(im)
enh.enhance(1.3).show("30% more contrast")
```

![../_images/contrasted_hopper.jpg](https://pillow.readthedocs.io/en/stable/_images/contrasted_hopper.jpg)

## Image sequences
The Python Imaging Library contains some basic support for image sequences (also called animation formats). Supported sequence formats include FLI/FLC, GIF, and a few experimental formats. TIFF files can also contain more than one frame.
>PIL 库包含对图像序列（也称为动画格式）的一些基本支持
>支持的序列格式包括 FLI/FLC、GIF 以及一些实验性格式，TIFF 文件也可以包含多个帧

When you open a sequence file, PIL automatically loads the first frame in the sequence. You can use the seek and tell methods to move between different frames:
>当打开一个序列文件时，PIL 会自动加载序列中的第一帧，可以使用 seek 和 tell 方法在不同的帧之间移动：

### Reading sequences

```python
from PIL import Image

with Image.open("animation.gif") as im:
    im.seek(1)  # skip to the second frame

    try:
        while 1:
            im.seek(im.tell() + 1)
            # do something to im
    except EOFError:
        pass  # end of sequence
```

As seen in this example, you’ll get an [`EOFError`](https://docs.python.org/3/library/exceptions.html#EOFError "(in Python v3.13)") exception when the sequence ends.

### Writing sequences
You can create animated GIFs with Pillow, e.g.
> 可以用多张图片创建 GIF

```python
from PIL import Image

# List of image filenames
image_filenames = [
    "hopper.jpg",
    "rotated_hopper_270.jpg",
    "rotated_hopper_180.jpg",
    "rotated_hopper_90.jpg",
]

# Open images and create a list
images = [Image.open(filename) for filename in image_filenames]

# Save the images as an animated GIF
images[0].save(
    "animated_hopper.gif",
    save_all=True,
    append_images=images[1:],
    duration=500,  # duration of each frame in milliseconds
    loop=0,  # loop forever
)
```

![../_images/animated_hopper.gif](https://pillow.readthedocs.io/en/stable/_images/animated_hopper.gif)

The following class lets you use the for-statement to loop over the sequence:
> `ImageSequence.Iterator` 类将图像序列包装为一个迭代器

### Using the [`Iterator`](https://pillow.readthedocs.io/en/stable/reference/ImageSequence.html#PIL.ImageSequence.Iterator "PIL.ImageSequence.Iterator") class

```python
from PIL import ImageSequence
for frame in ImageSequence.Iterator(im):
    # ...do something to frame...
```

## PostScript printing
The Python Imaging Library includes functions to print images, text and graphics on PostScript printers. Here’s a simple example:
>Python Imaging 库包含了可以在 PostScript 打印机上打印图像、文本和图形的功能，以下是一个简单的示例：

### Drawing PostScript

```python
from PIL import Image, PSDraw
import os

# Define the PostScript file
ps_file = open("hopper.ps", "wb")

# Create a PSDraw object
ps = PSDraw.PSDraw(ps_file)

# Start the document
ps.begin_document()

# Set the text to be drawn
text = "Hopper"

# Define the PostScript font
font_name = "Helvetica-Narrow-Bold"
font_size = 36

# Calculate text size (approximation as PSDraw doesn't provide direct method)
# Assuming average character width as 0.6 of the font size
text_width = len(text) * font_size * 0.6
text_height = font_size

# Set the position (top-center)
page_width, page_height = 595, 842  # A4 size in points
text_x = (page_width - text_width) // 2
text_y = page_height - text_height - 50  # Distance from the top of the page

# Load the image
image_path = "hopper.ppm"  # Update this with your image path
with Image.open(image_path) as im:
    # Resize the image if it's too large
    im.thumbnail((page_width - 100, page_height // 2))

    # Define the box where the image will be placed
    img_x = (page_width - im.width) // 2
    img_y = text_y + text_height - 200  # 200 points below the text

    # Draw the image (75 dpi)
    ps.image((img_x, img_y, img_x + im.width, img_y + im.height), im, 75)

# Draw the text
ps.setfont(font_name, font_size)
ps.text((text_x, text_y), text)

# End the document
ps.end_document()
ps_file.close()
```

![../_images/hopper_ps.webp](https://pillow.readthedocs.io/en/stable/_images/hopper_ps.webp)

Note
PostScript converted to PDF for display purposes

## More on reading images
As described earlier, the [`open()`](https://pillow.readthedocs.io/en/stable/reference/Image.html#PIL.Image.open "PIL.Image.open") function of the [`Image`](https://pillow.readthedocs.io/en/stable/reference/Image.html#module-PIL.Image "PIL.Image") module is used to open an image file. In most cases, you simply pass it the filename as an argument. `Image.open()` can be used as a context manager:

```python
from PIL import Image
with Image.open("hopper.ppm") as im:
    ...
```

If everything goes well, the result is an [`PIL.Image.Image`](https://pillow.readthedocs.io/en/stable/reference/Image.html#PIL.Image.Image "PIL.Image.Image") object. Otherwise, an [`OSError`](https://docs.python.org/3/library/exceptions.html#OSError "(in Python v3.13)") exception is raised.

You can use a file-like object instead of the filename. The object must implement `file.read`, `file.seek` and `file.tell` methods, and be opened in binary mode.
> `Image` 模块的 `open()` 除了接受文件名以外，也接受类文件的对象
> 该对象需要实现 `file.read/seek/tell` 方法，并且可以以二进制格式打开

### Reading from an open file

```python
from PIL import Image

with open("hopper.ppm", "rb") as fp:
    im = Image.open(fp)
```

> Python `open()` 函数返回的文件对象可以传递给 `Image` 模块的 `open()` 

To read an image from binary data, use the [`BytesIO`](https://docs.python.org/3/library/io.html#io.BytesIO "(in Python v3.13)") class:
>要从二进制数据中读取图像，可以使用[`BytesIO`](https://docs.python.org/3/library/io.html#io.BytesIO "(in Python v3.13)") 类：

### Reading from binary data

```python
from PIL import Image
import io

im = Image.open(io.BytesIO(buffer))
```

Note that the library rewinds the file (using `seek(0)`) before reading the image header. In addition, seek will also be used when the image data is read (by the load method).
>请注意，库会在读取图像头部之前重置文件位置（使用 `seek(0)`）
>此外，当通过 `load` 方法读取图像数据时也会使用 `seek`

If the image file is embedded in a larger file, such as a tar file, you can use the [`ContainerIO`](https://pillow. readthedocs. io/en/stable/PIL. html#module-PIL. ContainerIO "PIL. ContainerIO") or [`TarIO`](https://pillow. readthedocs. io/en/stable/PIL. html#module-PIL. TarIO "PIL. TarIO") modules to access it.
>如果图像文件嵌入在一个更大的文件中，例如tar文件，你可以使用 [`ContainerIO`](https://pillow.readthedocs.io/en/stable/PIL.html#module-PIL.ContainerIO "PIL.ContainerIO") 或 [`TarIO`](https://pillow.readthedocs.io/en/stable/PIL.html#module-PIL.TarIO "PIL.TarIO") 模块来访问它。

### Reading from URL

```python
from PIL import Image
from urllib.request import urlopen
url = "https://python-pillow.org/assets/images/pillow-logo.png"
img = Image.open(urlopen(url))
```

### Reading from a tar archive

```python
from PIL import Image, TarIO

fp = TarIO.TarIO("hopper.tar", "hopper.jpg")
im = Image.open(fp)
```

### Batch processing
Operations can be applied to multiple image files. For example, all PNG images in the current directory can be saved as JPEGs at reduced quality.

```python
import glob
from PIL import Image

def compress_image(source_path: str, dest_path: str) -> None:
    with Image.open(source_path) as img:
        if img.mode != "RGB":
            img = img.convert("RGB")
        img.save(dest_path, "JPEG", optimize=True, quality=80)

paths = glob.glob("*.png")
for path in paths:
    compress_image(path, path[:-4] + ".jpg")
```

Since images can also be opened from a `Path` from the `pathlib` module, the example could be modified to use `pathlib` instead of the `glob` module.

```python
from pathlib import Path

paths = Path(".").glob("*.png")
for path in paths:
    compress_image(path, path.stem + ".jpg")
```

## Controlling the decoder
Some decoders allow you to manipulate the image while reading it from a file. This can often be used to speed up decoding when creating thumbnails (when speed is usually more important than quality) and printing to a monochrome laser printer (when only a grayscale version of the image is needed).
>一些解码器允许你在从文件中读取图像时对图像进行操作，这在创建缩略图时非常有用（此时速度比质量更重要）以及打印到黑白激光打印机时（此时只需要图像的灰度版本）

The [`draft()`](https://pillow.readthedocs.io/en/stable/reference/Image.html#PIL.Image.Image.draft "PIL.Image.Image.draft") method manipulates an opened but not yet loaded image so it as closely as possible matches the given mode and size. This is done by reconfiguring the image decoder.
>[`draft()`](https://pillow.readthedocs.io/en/stable/reference/Image.html#PIL.Image.Image.draft "PIL.Image.Image.draft") 方法用于操纵已经打开但尚未加载的图像，使其尽可能接近给定的模式和尺寸，这是通过重新配置图像解码器来实现的

### Reading in draft mode
This is only available for JPEG and MPO files.

```python
from PIL import Image

with Image.open(file) as im:
    print("original =", im.mode, im.size)

    im.draft("L", (100, 100))
    print("draft =", im.mode, im.size)
```

This prints something like:

```
original = RGB (512, 512)
draft = L (128, 128)
```

Note that the resulting image may not exactly match the requested mode and size. To make sure that the image is not larger than the given size, use the thumbnail method instead.

# Concepts
The Python Imaging Library handles _raster images_; that is, rectangles of pixel data.
>PIL 处理的是光栅图像，即排列为矩形区域的像素数据
>(光栅图像是由像素组成的数字图像，这些像素通常排列成规则的二维网格结构，每个像素都包含颜色信息，常见的光栅图像文件格式包括JPEG、PNG和BMP等)
## Bands
An image can consist of one or more bands of data. The Python Imaging Library allows you to store several bands in a single image, provided they all have the same dimensions and depth. For example, a PNG image might have ‘R’, ‘G’, ‘B’, and ‘A’ bands for the red, green, blue, and alpha transparency values. Many operations act on each band separately, e.g., histograms. It is often useful to think of each pixel as having one value per band.
>一张图像可以由一个或多个数据带组成，PIL 允许我们在单个图像中存储多个带，前提是它们具有相同的尺寸和深度
>例如，一个 PNG 图像可能有 'R'、'G'、'B' 和 'A' 带，分别对应红色、绿色、蓝色和 alpha 透明度值
>许多操作会对每个单独的带进行处理，例如直方图
>我们通常可以认为每个像素在每个带上各有一个值

To get the number and names of bands in an image, use the [`getbands()`](https://pillow.readthedocs.io/en/stable/reference/Image.html#PIL.Image.Image.getbands "PIL.Image.Image.getbands") method.
> `getbands()` 方法用于获取图像中的带的数量和名称

## Modes
The `mode` of an image is a string which defines the type and depth of a pixel in the image. Each pixel uses the full range of the bit depth. So a 1-bit pixel has a range of 0-1, an 8-bit pixel has a range of 0-255, a 32-signed integer pixel has the range of INT32 and a 32-bit floating point pixel has the range of FLOAT32. 
>图像的 `mode` 是一个字符串，它定义了图像中像素的类型和深度
>在图像中，位深度定义了每个像素取值的完整范围，例如，1位像素的范围是0-1，8位像素的范围是0-255，32位有符号整数像素的范围是 INT32，32位浮点像素的范围是 FLOAT32

The current release supports the following standard modes:
>PIL 当前版本支持以下标准模式 (mode)：

> - `1` (1-bit pixels, black and white, stored with one pixel per byte)
> - `L` (8-bit pixels, grayscale)
> - `P` (8-bit pixels, mapped to any other mode using a color palette)
> - `RGB` (3x8-bit pixels, true color)
> - `RGBA` (4x8-bit pixels, true color with transparency mask)
> - `CMYK` (4x8-bit pixels, color separation)
> - `YCbCr` (3x8-bit pixels, color video format)
>     - Note that this refers to the JPEG, and not the ITU-R BT.2020, standard
> - `LAB` (3x8-bit pixels, the L*a*b color space)
> - `HSV` (3x8-bit pixels, Hue, Saturation, Value color space)
>     - Hue’s range of 0-255 is a scaled version of 0 degrees <= Hue < 360 degrees
> - `I` (32-bit signed integer pixels)
> - `F` (32-bit floating point pixels)

Pillow also provides limited support for a few additional modes, including:
> PIL 对部分额外的 mode 提供受限的支持，包括：

> - `LA` (L with alpha)
> - `PA` (P with alpha)
> - `RGBX` (true color with padding)
> - `RGBa` (true color with premultiplied alpha)
> - `La` (L with premultiplied alpha)
> - `I;16` (16-bit unsigned integer pixels)
> - `I;16L` (16-bit little endian unsigned integer pixels)
> - `I;16B` (16-bit big endian unsigned integer pixels)
> - `I;16N` (16-bit native endian unsigned integer pixels)

Premultiplied alpha is where the values for each other channel have been multiplied by the alpha. For example, an RGBA pixel of `(10, 20, 30, 127)` would convert to an RGBa pixel of `(5, 10, 15, 127)`. The values of the R, G and B channels are halved as a result of the half transparency in the alpha channel.
> `RGBa` 中的预乘 alpha 是指其他每个通道的值都乘以 alpha 值
>例如，一个RGBA像素 `(10, 20, 30, 127)` 会转换为 RGBa 像素 `(5, 10, 15, 127)`，因为 alpha 通道中的半透明度会将 R、G 和 B 通道的值被减半

Apart from these additional modes, Pillow doesn’t yet support multichannel images with a depth of more than 8 bits per channel.
>除了这些额外模式外，Pillow 尚不支持每个通道深度超过8位的多通道图像

Pillow also doesn’t support user-defined modes; if you need to handle band combinations that are not listed above, use a sequence of Image objects.
>Pillow 也不支持用户自定义模式；如果我们需要处理没有在上面列出的通道组合，需要使用 Image 对象的序列

You can read the mode of an image through the [`mode`](https://pillow.readthedocs.io/en/stable/reference/Image.html#PIL.Image.Image.mode "PIL.Image.Image.mode") attribute. This is a string containing one of the above values.
>可以通过 [`mode`](https://pillow.readthedocs.io/en/stable/reference/Image.html#PIL.Image.Image.mode) 属性读取图像的模式，`mode` 其实就是一个包含上述值之一的字符串

## Size
You can read the image size through the [`size`](https://pillow.readthedocs.io/en/stable/reference/Image.html#PIL.Image.Image.size "PIL.Image.Image.size") attribute. This is a 2-tuple, containing the horizontal and vertical size in pixels.
> `size` 属性为 2-tuple，存储了图像在水平和垂直上的像素数量

## Coordinate System
The Python Imaging Library uses a Cartesian pixel coordinate system, with (0,0) in the upper left corner. Note that the coordinates refer to the implied pixel corners; the centre of a pixel addressed as (0, 0) actually lies at (0.5, 0.5).
>PIL 使用笛卡尔像素坐标系统，其中 (0,0) 位于左上角
>注意像素的坐标是指它的左上角的坐标位置，因此地址为 (0, 0) 的像素的中心实际上位于 (0.5, 0.5)

Coordinates are usually passed to the library as 2-tuples (x, y). Rectangles are represented as 4-tuples, (x1, y1, x2, y2), with the upper left corner given first.
>坐标通常作为 2-元组 (x, y) 传递给库，矩形表示为 4-元组，(x1, y1, x2, y2)，其中左上角首先给出

## Palette
The palette mode (`P`) uses a color palette to define the actual color for each pixel.
>调色板模式（`P`）使用颜色调色板定义每个像素的实际颜色。

## Info
You can attach auxiliary information to an image using the [`info`](https://pillow.readthedocs.io/en/stable/reference/Image.html#PIL.Image.Image.info "PIL.Image.Image.info") attribute. This is a dictionary object.
>可以使用 [`info`](https://pillow. readthedocs. io/en/stable/reference/Image. html#PIL. Image. Image. info  "PIL. Image. Image. info") 属性将辅助信息附加到图像上，`info`  是一个字典对象

How such information is handled when loading and saving image files is up to the file format handler (see the chapter on [Image file formats](https://pillow.readthedocs.io/en/stable/handbook/image-file-formats.html#image-file-formats)). Most handlers add properties to the [`info`](https://pillow.readthedocs.io/en/stable/reference/Image.html#PIL.Image.Image.info "PIL.Image.Image.info") attribute when loading an image, but ignore it when saving images.
>在加载和保存图像文件时如何处理附加信息取决于文件格式处理器（见[图像文件格式](https://pillow.readthedocs.io/en/stable/handbook/image-file-formats.html#image-file-formats)章节）
>大多数处理器在加载图像时会向 [`info`](https://pillow. readthedocs. io/en/stable/reference/Image. html#PIL. Image. Image. info  "PIL. Image. Image. info") 添加属性，但在保存图像时会忽略它

## Transparency
If an image does not have an alpha band, transparency may be specified in the [`info`](https://pillow.readthedocs.io/en/stable/reference/Image.html#PIL.Image.Image.info "PIL.Image.Image.info") attribute with a “transparency” key.
>如果图像没有 alpha 通道，透明度可以在 [`info`](https://pillow. readthedocs. io/en/stable/reference/Image. html#PIL. Image. Image. info  "PIL. Image. Image. info") 属性中使用 “transparency” 键指定

Most of the time, the “transparency” value is a single integer, describing which pixel value is transparent in a “1”, “L”, “I” or “P” mode image. However, PNG images may have three values, one for each channel in an “RGB” mode image, or can have a byte string for a “P” mode image, to specify the alpha value for each palette entry.
>大多数情况下，“transparency” 值是单个整数，它描述在 “1”，“L”，“I” 或 “P” 模式图像中哪个像素值是透明的
>PNG 图像可以有三个值，分别对应 “RGB” 模式图像的每个通道，或者对于 “P” 模式图像，可以使用一个字节字符串来指定每个调色板条目的 alpha 值

## Orientation
A common element of the [`info`](https://pillow.readthedocs.io/en/stable/reference/Image.html#PIL.Image.Image.info "PIL.Image.Image.info") attribute for JPG and TIFF images is the EXIF orientation tag. This is an instruction for how the image data should be oriented. For example, it may instruct an image to be rotated by 90 degrees, or to be mirrored. To apply this information to an image, [`exif_transpose()`](https://pillow.readthedocs.io/en/stable/reference/ImageOps.html#PIL.ImageOps.exif_transpose "PIL.ImageOps.exif_transpose") can be used.
>[`info`](https://pillow. readthedocs. io/en/stable/reference/Image. html#PIL. Image. Image. info  "PIL. Image. Image. info") 属性对于 JPG 和 TIFF 图像的一个常见元素是 EXIF 方向标签
>这是一个指示图像数据应该如何定向的指令，它可指示图像旋转 90 度，或者进行镜像等，要将此信息应用于图像，可以使用 [`exif_transpose ()`](https://pillow. readthedocs. io/en/stable/reference/ImageOps. html#PIL. ImageOps. exif_transpose  "PIL. ImageOps. exif_transpose")

## Filters
For geometry operations that may map multiple input pixels to a single output pixel, the Python Imaging Library provides different resampling _filters_.
>对于可能将多个输入像素映射到单个输出像素的几何操作，PIL 提供了不同的重采样滤波器

`Resampling.NEAREST`
Pick one nearest pixel from the input image. Ignore all other input pixels.
>从输入图像中选择一个最近的像素。忽略所有其他输入像素

`Resampling.BOX`
Each pixel of source image contributes to one pixel of the destination image with identical weights. For upscaling is equivalent of [`Resampling.NEAREST`](https://pillow.readthedocs.io/en/stable/reference/Image.html#PIL.Image.Resampling.NEAREST "PIL.Image.Resampling.NEAREST"). This filter can only be used with the [`resize()`](https://pillow.readthedocs.io/en/stable/reference/Image.html#PIL.Image.Image.resize "PIL.Image.Image.resize") and [`thumbnail()`](https://pillow.readthedocs.io/en/stable/reference/Image.html#PIL.Image.Image.thumbnail "PIL.Image.Image.thumbnail") methods.
>每个源图像像素以相同的权重贡献于目标图像的一个像素，对于放大，等同于 [`Resampling.NEAREST`](https://pillow. readthedocs. io/en/stable/reference/Image. html#PIL. Image. Resampling. NEAREST  "PIL. Image. Resampling. NEAREST")
>这个滤波器只能与 [`resize ()`](https://pillow. readthedocs. io/en/stable/reference/Image. html#PIL. Image. Image. resize  "PIL. Image. Image. resize") 和 [`thumbnail ()`](https://pillow. readthedocs. io/en/stable/reference/Image. html#PIL. Image. Image. thumbnail  "PIL. Image. Image. thumbnail") 方法一起使用

Added in version 3.4.0.

`Resampling.BILINEAR`
For resize calculate the output pixel value using linear interpolation on all pixels that may contribute to the output value. For other transformations linear interpolation over a 2x2 environment in the input image is used.
>对于 resize ，使用线性插值计算输出像素值，其中可能对输出值的做出贡献的所有像素都参与计算
>对于其他变换，使用输入图像中的 2x2 环境进行线性插值

`Resampling.HAMMING`
Produces a sharper image than [`Resampling.BILINEAR`](https://pillow.readthedocs.io/en/stable/reference/Image.html#PIL.Image.Resampling.BILINEAR "PIL.Image.Resampling.BILINEAR"), doesn’t have dislocations on local level like with [`Resampling.BOX`](https://pillow.readthedocs.io/en/stable/reference/Image.html#PIL.Image.Resampling.BOX "PIL.Image.Resampling.BOX"). This filter can only be used with the [`resize()`](https://pillow.readthedocs.io/en/stable/reference/Image.html#PIL.Image.Image.resize "PIL.Image.Image.resize") and [`thumbnail()`](https://pillow.readthedocs.io/en/stable/reference/Image.html#PIL.Image.Image.thumbnail "PIL.Image.Image.thumbnail") methods.
>产生的图像比 [`resampling.BILINEAR`](https://pillow. readthedocs. io/en/stable/reference/Image. html#PIL. Image. Resampling. BILINEAR  "PIL. Image. Resampling. BILINEAR") 更锐利，不会像 [`resampling.BOX`](https://pillow. readthedocs. io/en/stable/reference/Image. html#PIL. Image. Resampling. BOX  "PIL. Image. Resampling. BOX") 那样在局部级别上有错位
>这个滤波器只能与 [`resize ()`](https://pillow. readthedocs. io/en/stable/reference/Image. html#PIL. Image. Image. resize  "PIL. Image. Image. resize") 和 [`thumbnail ()`](https://pillow. readthedocs. io/en/stable/reference/Image. html#PIL. Image. Image. thumbnail  "PIL. Image. Image. thumbnail") 方法一起使用

Added in version 3.4.0.

`Resampling.BICUBIC`
For resize calculate the output pixel value using cubic interpolation on all pixels that may contribute to the output value. For other transformations cubic interpolation over a 4x4 environment in the input image is used.
>对于 resize，使用立方插值计算输出像素值，可能贡献输出值的所有像素都参与计算
>对于其他变换，使用输入图像中的 4x4 环境进行立方插值

`Resampling.LANCZOS`
Calculate the output pixel value using a high-quality Lanczos filter (a truncated sinc) on all pixels that may contribute to the output value. This filter can only be used with the [`resize()`](https://pillow.readthedocs.io/en/stable/reference/Image.html#PIL.Image.Image.resize "PIL.Image.Image.resize") and [`thumbnail()`](https://pillow.readthedocs.io/en/stable/reference/Image.html#PIL.Image.Image.thumbnail "PIL.Image.Image.thumbnail") methods.
>使用高质量的 Lanczos 过滤器（一个截断的 sinc）计算输出像素值，可能贡献输出值的所有像素都参与计算
>这个过滤器只能与 [`resize ()`](https://pillow. readthedocs. io/en/stable/reference/Image. html#PIL. Image. Image. resize  "PIL. Image. Image. resize") 和 [`thumbnail ()`](https://pillow. readthedocs. io/en/stable/reference/Image. html#PIL. Image. Image. thumbnail  "PIL. Image. Image. thumbnail") 方法一起使用

Added in version 1.1.3.

### Filters comparison table

|Filter|Downscaling quality|Upscaling quality|Performance|
|---|---|---|---|
|[`Resampling.NEAREST`](https://pillow.readthedocs.io/en/stable/reference/Image.html#PIL.Image.Resampling.NEAREST "PIL.Image.Resampling.NEAREST")|||⭐⭐⭐⭐⭐|
|[`Resampling.BOX`](https://pillow.readthedocs.io/en/stable/reference/Image.html#PIL.Image.Resampling.BOX "PIL.Image.Resampling.BOX")|⭐||⭐⭐⭐⭐|
|[`Resampling.BILINEAR`](https://pillow.readthedocs.io/en/stable/reference/Image.html#PIL.Image.Resampling.BILINEAR "PIL.Image.Resampling.BILINEAR")|⭐|⭐|⭐⭐⭐|
|[`Resampling.HAMMING`](https://pillow.readthedocs.io/en/stable/reference/Image.html#PIL.Image.Resampling.HAMMING "PIL.Image.Resampling.HAMMING")|⭐⭐||⭐⭐⭐|
|[`Resampling.BICUBIC`](https://pillow.readthedocs.io/en/stable/reference/Image.html#PIL.Image.Resampling.BICUBIC "PIL.Image.Resampling.BICUBIC")|⭐⭐⭐|⭐⭐⭐|⭐⭐|
|[`Resampling.LANCZOS`](https://pillow.readthedocs.io/en/stable/reference/Image.html#PIL.Image.Resampling.LANCZOS "PIL.Image.Resampling.LANCZOS")|⭐⭐⭐⭐|⭐⭐⭐⭐|⭐|











