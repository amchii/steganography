import sys

import cv2
import numpy as np


def get_mse(im_true: np.ndarray, im_test):
    assert im_true.shape == im_test.shape, "array shape not equal"
    diff = im_true - im_test
    mse = 0
    for i in range(3):
        mse += np.mean(np.square(diff[i]))
    mse /= 3
    print(f"均方误差MSE:{mse}")
    return mse


def get_psnr(im_true: np.ndarray, im_test, data_max):
    psnr = 10 * np.log10(data_max * data_max / get_mse(im_true, im_test))
    print(f'峰值信噪比PSNR:{psnr}')
    return psnr


def show_image(im, win_name=None, flags=cv2.WINDOW_AUTOSIZE, timeout=0):
    """
    展示图片
    BackSpace: 销毁当前窗口
    Enter: 销毁所有窗口
    Other: 保留当前窗口
    """
    win_name = win_name or "image"
    cv2.namedWindow(win_name, flags)
    cv2.imshow(win_name, im)
    key = cv2.waitKey(timeout * 1000)
    if key == 8:  # BackSpace
        cv2.destroyWindow(win_name)
    elif key == 13:  # Enter
        cv2.destroyAllWindows()


def real_im(im: np.ndarray):
    return np.uint8(im.real.clip(0, 255))


def bgr2rgb(im):
    return cv2.cvtColor(im, cv2.COLOR_BGR2RGB)


def rgb2bgr(im):
    return cv2.cvtColor(im, cv2.COLOR_RGB2BGR)


def img_uint82float(im: np.ndarray):
    assert im.dtype == np.dtype("uint8"), f"{im.dtype} is not 'uint8'"
    return np.double(im) / 255


def cyclic_repair(im: np.ndarray, imsize: tuple, l2r=True, t2b=True):
    """
    将图片循环补全至指定尺寸
    :param im:
    :param imsize: (height, width)
    :param l2r: 左到右
    :param t2b: 上到下
    """
    o_w = im.shape[1]
    o_h = im.shape[0]
    assert o_w <= imsize[1] and o_h <= imsize[0], f"图像尺寸: ({o_w}, {o_h}) 大于输入尺寸{imsize}"
    im_repair = np.zeros((imsize[0], imsize[1], 3), dtype="uint8")
    if l2r and t2b:
        im_repair[0: im.shape[0], 0: im.shape[1], :] = im
        show_image(im_repair)
        im_repair[im.shape[0]:, : im.shape[1], :] = im[
                                                    0: imsize[0] - im.shape[0], :, :
                                                    ]
        show_image(im_repair, "vertical repair")
        im = im_repair[:, : im.shape[1], :]
        im_repair[:, im.shape[1]:, :] = im[:, 0: imsize[1] - im.shape[1], :]
        show_image(im_repair, "horizontal repair")
    else:
        raise ValueError("not supported now ^_^")
    return im_repair


def encode_watermark(
        watermark: np.ndarray, im_shape: tuple, encode=True, encoded_file="encode.npz"
):
    """
    :param watermark: numpy.ndarray, 水印
    :param im_shape: 3-tuple, 底图shape
    :param encode: bool, 是否编码
    :param encoded_file: str,  保存编码序列的文件名
    :return: numpy.ndarray, shap=im_shape的对称水印
    """
    encoded_mark = np.zeros((int(im_shape[0] * 0.5), im_shape[1], im_shape[2]))
    origin_mark = encoded_mark.copy()
    mark_size = watermark.shape
    origin_mark[0: mark_size[0], 0: mark_size[1], :] = watermark
    if encode:
        # 随机编码序列
        h = np.random.permutation(int(im_shape[0] * 0.5))
        w = np.random.permutation(im_shape[1])
        np.savez(encoded_file, h=h, w=w)  # 保存本地

        # 编码
        for i in range(int(im_shape[0] * 0.5)):
            for j in range(im_shape[1]):
                encoded_mark[i, j, :] = origin_mark[h[i], w[j], :]
    else:
        encoded_mark = origin_mark

    # 对称
    mark_ = np.zeros(im_shape)
    mark_[0: int(im_shape[0] * 0.5), 0: im_shape[1], :] = encoded_mark
    for i in range(int(im_shape[0] * 0.5)):
        for j in range(im_shape[1]):
            mark_[im_shape[0] - 1 - i, im_shape[1] - 1 - j, :] = encoded_mark[i, j, :]

    return mark_


def add_watermark(im, watermark, alpha=1):
    """
    :param im:
    :param watermark:
    :param alpha: 能量系数
    :return: numpy.ndarray, 隐写后图片
    """
    im_fft = np.fft.fft2(im)
    show_image(real_im(im_fft), "spectrum of origin image")
    marked_fft = im_fft + alpha * watermark
    show_image(real_im(marked_fft), "spectrum of watermarked image")
    marked_ifft = np.fft.ifft2(marked_fft)
    return marked_ifft


def extract_watermark(img, origin_img, encoded_file=None, alpha=1):
    im, origin_im = img, origin_img
    if not isinstance(im, np.ndarray):
        im = cv2.imread(img)
    if not isinstance(origin_im, np.ndarray):
        origin_im = cv2.imread(origin_img)
    assert im is not None and origin_im is not None, "图像文件不存在"  # and被覆写
    encode_sequence = np.load(encoded_file)
    h = encode_sequence["h"]
    w = encode_sequence["w"]

    im_shape = origin_im.shape

    im_fft = np.fft.fft2(im)
    orgin_im_fft = np.fft.fft2(origin_im)
    encoded_mark = (im_fft - orgin_im_fft) / alpha
    mark = np.zeros(im_shape, dtype="complex128")
    for i in range(0, int(im_shape[0] * 0.5)):
        for j in range(0, im_shape[1]):
            mark[h[i], w[j], :] = encoded_mark[i, j, :]

    for i in range(0, int(im_shape[0] * 0.5)):
        for j in range(0, im_shape[1]):
            mark[im_shape[0] - 1 - h[i], im_shape[1] - 1 - w[j], :] = encoded_mark[
                                                                      im_shape[0] - 1 - i, im_shape[1] - 1 - j, :
                                                                      ]
    return mark


def show_extract(origin_img, edit_image, encoded_file='watermark_encode.npz'):
    im = cv2.imread(origin_img)
    show_image(im, "origin_image")
    edit_im = cv2.imread(edit_image)
    show_image(edit_im, "edit image")
    extract = extract_watermark(
        edit_image, origin_img, encoded_file, alpha=alpha
    )
    show_image(real_im(extract))


# -------------------------------------------attack-----------------------------------------------


def jpeg_test(jpg_file):
    im = cv2.imread(origin_img)
    show_image(im, "origin image")
    jpg_img = cv2.imread(jpg_file)
    show_image(jpg_img, "jpg image")
    extract = extract_watermark(
        jpg_img, origin_img, "watermark_encode.npz", alpha=alpha
    )
    show_image(real_im(extract))


def _crop_test(im_crop):
    im = cv2.imread(origin_img)
    show_image(real_im(im_crop), "croped")
    repaired_im = cyclic_repair(im_crop, im.shape[:2])
    repaired_extract_mark = extract_watermark(
        repaired_im, im, "watermark_encode.npz", alpha=alpha
    )
    show_image(real_im(repaired_extract_mark), "extracted watermark")


def shot_test(shot_file):
    im = cv2.imread(origin_img)
    shot = cv2.imread(shot_file)
    show_image(shot, "shot image")
    shot_resize = cv2.resize(shot, (im.shape[1], im.shape[0]))
    show_image(shot_resize, "resized shot image")
    extract = extract_watermark(
        shot_resize, origin_img, "watermark_encode.npz", alpha=alpha
    )
    show_image(real_im(extract))


def crop_test():
    im = cv2.imread(origin_img)

    # 裁剪攻击
    marked_im = cv2.imread("watermarked_img.png")
    # 横向裁剪
    im_crop = marked_im[
              :, 0: int(im.shape[1] * 0.8),
              ]
    _crop_test(im_crop)
    # 纵向裁剪
    im_crop = marked_im[
              0: int(im.shape[0] * 0.8), :,
              ]
    _crop_test(im_crop)
    # 横纵向裁剪
    im_crop = marked_im[
              0: int(im.shape[0] * 0.8), 0: int(im.shape[1] * 0.8),
              ]
    _crop_test(im_crop)


def write_and_read(encoded_file='watermark_encode.npz'):
    # 读取原图和水印
    im = cv2.imread(origin_img)
    show_image(im, "origin image")
    watermark = cv2.imread(water_mark)
    show_image(watermark, "watermark")

    # 编码水印
    encoded_mark = encode_watermark(
        watermark, im.shape, encoded_file=encoded_file
    )
    show_image(encoded_mark, win_name="encoded watermark")

    # 添加水印
    marked_im = add_watermark(im, encoded_mark, alpha=alpha)
    show_image(real_im(marked_im), "watermarked image")
    cv2.imwrite("watermarked_img.png", real_im(marked_im))

    # 解水印
    extract_mark = extract_watermark(
        "watermarked_img.png", origin_img, "watermark_encode.npz", alpha=alpha
    )
    show_image(real_im(extract_mark), "extracted watermark")
    return im, marked_im


def main():
    args = sys.argv
    if len(args) == 1:
        sys.exit()
    option = args[1]
    if option == 'run':
        if len(args) == 3:
            encoded_file = args[2]
            im, marked_im = write_and_read(encoded_file)
        else:
            im, marked_im = write_and_read()
        get_psnr(im, marked_im, 255)
    elif option == 'extract':
        assert len(args) == 5, '参数错误'
        origin_img, edit_image, encoded_file = args[2], args[3], args[4]
        show_extract(origin_img, edit_image, encoded_file)
    elif option == 'test_crop':
        crop_test()
    elif option == 'test_shot':
        assert len(args) == 3
        shot_file = args[2]
        shot_test(shot_file)


if __name__ == "__main__":
    alpha = 4
    origin_img = "woman.png"
    water_mark = "zhangsan.png"
    # im, marked_im = write_and_read()
    # get_psnr(im, marked_im, 255)
    # show_extract(origin_img, "smear.png")
    # shot_test('resize.png')
    # jpeg_test("watermarked_img.jpg")
    # crop_test()
    main()
