import math
import os
import random
from typing import Optional

try:
	import numpy as np
except Exception:
	np = None

try:
	import torch
except Exception:
	torch = None


def set_random_seed(seed: Optional[int] = None, deterministic: bool = True) -> int:
	"""랜덤 시드를 설정해서 실험 재현성을 돕습니다.

	- seed가 None이면 시스템 랜덤으로 시드를 생성합니다.
	- Python `random`, `numpy`, `torch`(있을 경우), 그리고 `PYTHONHASHSEED`를 설정합니다.
	- `deterministic=True`이면 PyTorch의 경우 가능한 결정적(deterministic) 동작을 설정합니다.

	Returns:
		사용된 정수 시드

	사용 예:
		seed = set_random_seed(42)
		seed = set_random_seed()  # 자동 시드 생성
	"""
	if seed is None:
		# os.urandom으로부터 4바이트를 얻어 int 변환 (안전한 랜덤)
		seed = int.from_bytes(os.urandom(4), byteorder="little")

	# 환경 변수로 해시 시드 고정 (파이썬 해시 관련 비결정성 방지)
	os.environ.setdefault("PYTHONHASHSEED", str(seed))

	# 표준 random
	random.seed(seed)

	# numpy
	if np is not None:
		try:
			np.random.seed(seed)
		except Exception:
			pass

	# torch (있을 경우)
	if torch is not None:
		try:
			torch.manual_seed(seed)
			if torch.cuda.is_available():
				torch.cuda.manual_seed_all(seed)
			if deterministic:
				# 가능한 경우 결정적 연산 사용
				torch.backends.cudnn.deterministic = True
				torch.backends.cudnn.benchmark = False
		except Exception:
			pass

	return int(seed)


if __name__ == "__main__":
	# 간단한 데모: 자동 시드 생성 및 출력
	s = set_random_seed(None)
	print(f"사용된 랜덤 시드: {s}")

