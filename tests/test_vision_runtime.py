import cv2
import numpy as np
import pytesseract


def test_tesseract_recognizes_a_synthetic_counter() -> None:
    image = np.full((60, 180), 255, dtype=np.uint8)
    cv2.putText(
        image, "123", (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 2
    )

    recognized = pytesseract.image_to_string(
        image,
        config="--psm 7 -c tessedit_char_whitelist=0123456789",
    ).strip()

    assert recognized == "123"
