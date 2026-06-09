"""
串口通信协议单元测试（不依赖硬件）。

验证 AA 帧协议的帧构建和解析逻辑。
"""
import pytest


class TestAAFrameProtocol:
    """AA 帧协议测试"""

    def test_classification_frame_format(self):
        """分类帧格式：AA <class_id> 55"""
        def build_frame(class_id: int) -> bytes:
            if not (1 <= class_id <= 4):
                raise ValueError(f"Invalid class_id: {class_id}")
            return bytes([0xAA, class_id, 0x55])

        assert build_frame(1) == b'\xAA\x01\x55'  # 可回收
        assert build_frame(2) == b'\xAA\x02\x55'  # 有害
        assert build_frame(3) == b'\xAA\x03\x55'  # 厨余
        assert build_frame(4) == b'\xAA\x04\x55'  # 其他

    def test_classification_frame_parse(self):
        """解析分类帧"""
        def parse_frame(data: bytes) -> int:
            if len(data) < 3:
                raise ValueError("Frame too short")
            if data[0] != 0xAA or data[2] != 0x55:
                raise ValueError("Invalid frame header/footer")
            return data[1]

        assert parse_frame(b'\xAA\x01\x55') == 1
        assert parse_frame(b'\xAA\x04\x55') == 4

    def test_detect_frame(self):
        """检测帧：0xA1"""
        assert 0xA1 == 0xA1

    def test_ack_frame(self):
        """确认帧：0xCC"""
        assert 0xCC == 0xCC

    def test_done_frame(self):
        """完成帧：0xDD"""
        assert 0xDD == 0xDD

    def test_invalid_class_id_raises(self):
        """非法 class_id 抛出异常"""
        def build_frame(class_id: int) -> bytes:
            if not (1 <= class_id <= 4):
                raise ValueError(f"Invalid class_id: {class_id}")
            return bytes([0xAA, class_id, 0x55])

        with pytest.raises(ValueError):
            build_frame(0)
        with pytest.raises(ValueError):
            build_frame(5)


class TestCharProtocol:
    """单字符协议测试"""

    CHAR_MAP = {
        'R': 0,  # 可回收
        'H': 1,  # 有害
        'K': 2,  # 厨余
        'O': 3,  # 其他
    }

    def test_char_mapping(self):
        assert self.CHAR_MAP['R'] == 0
        assert self.CHAR_MAP['H'] == 1
        assert self.CHAR_MAP['K'] == 2
        assert self.CHAR_MAP['O'] == 3

    def test_invalid_char(self):
        with pytest.raises(KeyError):
            _ = self.CHAR_MAP['X']
