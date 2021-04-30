import hashlib

from Crypto.Cipher import AES
from Crypto import Random


class AESCiper:
    def __init__(self, key):
        self.key = hashlib.md5(key.encode("utf-8")).digest()
        self.block_size = AES.block_size

    def _pad(self, s):
        pad_size = self.block_size - len(s) % self.block_size
        return s + pad_size * chr(pad_size).encode("utf-8")

    @staticmethod
    def _unpad(s):
        return s[: -ord(s[-1:])]

    def encrypt(self, payload):
        iv = Random.new().read(self.block_size)
        ciper = AES.new(self.key, AES.MODE_CBC, iv)
        return iv + ciper.encrypt(self._pad(payload))

    def decrypt(self, payload):
        iv = payload[: self.block_size]
        ciper = AES.new(self.key, AES.MODE_CBC, iv)
        return self._unpad(ciper.decrypt(payload[self.block_size :]))
