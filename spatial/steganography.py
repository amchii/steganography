"""
Use Pillow
"""
import os
import sys
import time

from PIL import Image, ImageDraw
from PIL.ImageFile import ImageFile

from crypt_ import AESCiper


def set_bit(n, i, bit):
    return (n & ~(1 << i)) | (bit << i)


def get_bit(n, i):
    return n >> i & 1


def calc_time(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        t = func(*args, **kwargs)
        print("%s的执行时间: %s" % (func.__name__, time.time() - start))
        return t

    return wrapper


# 快速转换bytes为bits
@calc_time
def bytes2bits(bytes_):
    num = int.from_bytes(bytes_, "big")
    bit_length = len(bytes_) * 8
    bits = bin(num)[2:].zfill(bit_length)
    return bits


@calc_time
def bytes2bits2(bytes_):
    # 字符串拼接的方式:str+=str 很慢
    bits = []
    for b in bytes_:
        bits.append(bin(b).replace("0b", "").zfill(8))
    return "".join(bits)


@calc_time
def _write(img_file, payload, password=None):
    img: ImageFile = Image.open(img_file)
    draw = ImageDraw.Draw(img)
    width, height = img.size

    if password:
        print("加密ing......")
        ciper = AESCiper(password)
        payload = ciper.encrypt(payload)

    payload_length = len(payload)
    max_payload_length = int(width * height * 3 / 8) - 3  # 最后3个字节存储payload字节长度
    print(f"最大写入长度: {max_payload_length} bytes")
    if payload_length > max_payload_length:
        print(f"超过最大写入长度: {max_payload_length} bytes")
        sys.exit()

    # 写入payload
    payload_bits = bytes2bits(payload)
    idx = 0
    flag = False
    for h in range(height):
        for w in range(width):
            (r, g, b) = img.getpixel((w, h))[:3]
            try:
                r = set_bit(r, 0, int(payload_bits[idx + 0]))
                g = set_bit(g, 0, int(payload_bits[idx + 1]))
                b = set_bit(b, 0, int(payload_bits[idx + 2]))
                idx += 3
            except IndexError:
                flag = True
                break
            finally:
                draw.point((w, h), (r, g, b))
        if flag:
            break

    # 末尾3字节存储payload字节长度
    length_bytes = payload_length.to_bytes(3, "big")
    length_bits = bytes2bits(length_bytes)

    idx = 0
    flag = False
    for h in range(height - 1, -1, -1):
        for w in range(width - 1, -1, -1):
            (r, g, b) = img.getpixel((w, h))[:3]
            if idx >= 3 * 8:
                flag = True
                break
            r = set_bit(r, 0, int(length_bits[idx + 0]))
            g = set_bit(g, 0, int(length_bits[idx + 1]))
            b = set_bit(b, 0, int(length_bits[idx + 2]))
            draw.point((w, h), (r, g, b))
            idx += 3
        if flag:
            break

    file_name = "stego_" + os.path.splitext(img_file)[0] + ".png"
    img.save(file_name, "png")  # png无损保存
    print("写入成功!")
    return file_name


@calc_time
def _read(img_file, password=None):
    img: ImageFile = Image.open(img_file)
    width, height = img.size

    # 读取payload字节长度
    v = []
    idx = 0
    flag = False
    for h in range(height - 1, -1, -1):
        for w in range(width - 1, -1, -1):
            if idx >= 3 * 8:
                flag = True
                break
            (r, g, b) = img.getpixel((w, h))[:3]
            r = str(get_bit(r, 0))
            g = str(get_bit(g, 0))
            b = str(get_bit(b, 0))
            v.extend([r, g, b])
            idx += 3
        if flag:
            break
    payload_length = int("".join(v), base=2)
    bit_length = payload_length * 8
    print(f"隐写数据大小: {payload_length} bytes ")

    # 读取前bit_length个bit
    v = []
    flag = False
    for h in range(height):
        for w in range(width):
            (r, g, b) = img.getpixel((w, h))[:3]
            r = str(get_bit(r, 0))
            g = str(get_bit(g, 0))
            b = str(get_bit(b, 0))
            v.extend([r, g, b])
            if len(v) >= bit_length:
                flag = True
                break
        if flag:
            break
    payload = int("".join(v[:bit_length]), base=2).to_bytes(payload_length, "big")
    if password:
        print("解密ing......")
        ciper = AESCiper(password)
        payload = ciper.decrypt(payload)
    return payload


def write_file_to_img(img_file, payload_file, password=None):
    """
    将文件写入图片中
    :param img_file: str
    :param payload_file: str
    :param password: str
    :return: file's name, str
    """
    with open(payload_file, "rb") as fp:
        data = fp.read()
    # 前8个字节存储文件扩展名
    ext = os.path.splitext(payload_file)[-1]
    payload = ext.encode("utf-8").rjust(8, b"\x00")[-8:] + data
    return _write(img_file, payload, password)


def read_from_img(img_file, password=None):
    """
    :param img_file: str
    :param password:
    :return: 2-tuple, (<文件扩展名>, <文件内容>)
    """
    payload = _read(img_file, password)
    ext = payload[:8].strip(b"\x00").decode("utf-8")
    return ext, payload[8:]


@calc_time
def main():
    file = write_file_to_img("a.jpg", "a.txt", "secret")
    ext, payload = read_from_img(file, "secret")
    print(ext)
    with open(f"out{ext}", "wb") as f:
        f.write(payload)


if __name__ == "__main__":
    main()
