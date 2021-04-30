"""
Use Opencv
"""
import os
import sys
import time

import cv2
import numpy


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


def bytes2bits2(bytes_):
    # 字符串拼接的方式:str+=str 很慢
    bits = []
    for b in bytes_:
        bits.append(bin(b).replace("0b", "").zfill(8))
    return "".join(bits)


@calc_time
def _write(img_file, payload, password=None):
    im: numpy.ndarray = cv2.imread(img_file)
    height, width = im.shape[:2]

    payload_length = len(payload)
    max_payload_length = int(width * height * 3 / 8) - 3  # 最后3个字节存储payload字节长度
    print(f"max size for write: {max_payload_length} bytes")
    if payload_length > max_payload_length:
        print(f"payload is too big for max length: {max_payload_length} bytes")
        sys.exit()

    # 写入payload
    payload_bits = bytes2bits(payload)
    idx = 0
    flag = False
    for h in range(height):
        for w in range(width):
            (b, g, r) = im[h][w]
            try:
                b = set_bit(b, 0, int(payload_bits[idx + 0]))
                g = set_bit(g, 0, int(payload_bits[idx + 1]))
                r = set_bit(r, 0, int(payload_bits[idx + 2]))
                idx += 3
            except IndexError:
                flag = True
                break
            finally:
                im[h][w] = [b, g, r]
        if flag:
            break
    # 末尾3字节存储payload字节长度
    length_bytes = payload_length.to_bytes(3, "big")
    length_bits = bytes2bits(length_bytes)

    idx = 0
    flag = False
    for h in range(height - 1, -1, -1):
        for w in range(width - 1, -1, -1):
            (b, g, r) = im[h][w]
            if idx >= 3 * 8:
                flag = True
                break
            b = set_bit(b, 0, int(length_bits[idx + 0]))
            g = set_bit(g, 0, int(length_bits[idx + 1]))
            r = set_bit(r, 0, int(length_bits[idx + 2]))
            idx += 3
            im[h][w] = [b, g, r]
        if flag:
            break
    file_name = "stego_" + os.path.splitext(img_file)[0] + ".png"
    cv2.imwrite(file_name, im)
    print("hide success!")
    return file_name


@calc_time
def _read(img_file, password=None):
    im: numpy.ndarray = cv2.imread(img_file)
    height, width = im.shape[:2]

    # 读取payload字节长度
    v = []
    idx = 0
    flag = False
    for h in range(height - 1, -1, -1):
        for w in range(width - 1, -1, -1):
            if idx >= 3 * 8:
                flag = True
                break
            (b, g, r) = im[h][w]
            b = str(get_bit(b, 0))
            g = str(get_bit(g, 0))
            r = str(get_bit(r, 0))
            v.extend([b, g, r])
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
            (b, g, r) = im[h][w]
            b = str(get_bit(b, 0))
            g = str(get_bit(g, 0))
            r = str(get_bit(r, 0))
            v.extend([b, g, r])
            if len(v) >= bit_length:
                flag = True
                break
        if flag:
            break
    payload = int("".join(v[:bit_length]), base=2).to_bytes(payload_length, "big")
    return payload


def write_file_to_img(img_file, payload_file, password=None):
    with open(payload_file, "rb") as fp:
        data = fp.read()
    # 前8个字节存储文件扩展名
    ext = os.path.splitext(payload_file)[-1]
    payload = ext.encode().rjust(8, b"\x00")[-8:] + data
    return _write(img_file, payload, password)


def read_from_img(img_file, password=None):
    payload = _read(img_file)
    ext = payload[:8].strip(b"\x00").decode("utf-8")
    return ext, payload[8:]


@calc_time
def main():
    if len(sys.argv) != 3:
        print(f"""
    USAGE:  python {sys.argv[0]} <ORIGIN_IMG> <PAYLOAD_FILE>
            eg: python {sys.argv[0]} a.png a.txt
        """)
        sys.exit()

    origin_img = sys.argv[1]
    payload_file = sys.argv[2]
    file_name = write_file_to_img(origin_img, payload_file)
    ext, payload = read_from_img(file_name)
    with open(f"output{ext}", "wb") as fp:
        fp.write(payload)


if __name__ == "__main__":
    main()
