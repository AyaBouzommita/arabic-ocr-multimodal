import pytest
from pipeline.evaluation.evaluator import (
    normalize_arabic_text,
    calculate_cer,
    calculate_wer,
)
from pipeline.utils.validator import (
    validate_ocr_output,
    validate_detection_output,
    validate_fusion_input,
    SchemaValidationError,
)

# --- Tests for Arabic Text Normalization ---


def test_normalize_arabic_diacritics():
    # Test stripping of Fatha, Damma, Kasra, Sukun, Shadda, Tanween, Tatweel
    text_with_diacritics = "كَتَبَ الرَّجُلُ الكِتَابَـ"
    expected = "كتب الرجل الكتاب"
    assert (
        normalize_arabic_text(
            text_with_diacritics,
            remove_diacritics=True,
            normalize_alef_ta_ya=False,
            remove_punctuation=False,
        )
        == expected
    )


def test_normalize_arabic_alef_ta_ya():
    # Test unifying Alef forms (أ, إ, آ -> ا), Teh Marbuta (ة -> ه), and Alef Maksura (ى -> ي)
    text = "أحمد ذهب إلى المدرسةِ ورأى الفتى"
    # أ -> ا, إ -> ا, ة -> ه, ى -> ي
    expected = "احمد ذهب الي المدرسه وراي الفتي"
    res = normalize_arabic_text(
        text,
        remove_diacritics=True,
        normalize_alef_ta_ya=True,
        remove_punctuation=False,
    )
    assert res == expected


def test_normalize_arabic_punctuation():
    # Test removing both Latin and Arabic punctuation
    text = "هل ذهبت؟ نعم، كتبت الدرس."
    expected = "هل ذهبت نعم كتبت الدرس"
    assert (
        normalize_arabic_text(
            text,
            remove_diacritics=False,
            normalize_alef_ta_ya=False,
            remove_punctuation=True,
        )
        == expected
    )


def test_normalize_multiple_spaces():
    text = "  كتب \t \n  الدرس   "
    expected = "كتب الدرس"
    assert normalize_arabic_text(text) == expected


# --- Tests for CER / WER Calculations & Edge Cases ---


def test_cer_wer_perfect_match():
    ref = "كتب الطالب الدرس في القسم"
    hyp = "كتب الطالب الدرس في القسم"
    assert calculate_cer(ref, hyp) == 0.0
    assert calculate_wer(ref, hyp) == 0.0


def test_cer_wer_simple_substitution():
    ref = "كتب الطالب الدرس"
    hyp = "كتب الكاتب الدرس"
    # Word level: 1 substitution out of 3 words = 1/3 = 0.3333...
    assert calculate_wer(ref, hyp) == pytest.approx(1.0 / 3.0)
    # Character level: Ref len is 16. 'الطالب' vs 'الكاتب' has 2 substitutions ('ط', 'ل' -> 'ك', 'ا').
    # Ref:  ك ت ب _ ا ل ط ا ل ب _ ا ل د ر س
    # Hyp:  ك ت ب _ ا ل ك ا ت ب _ ا ل د ر س
    # Subst: 'ط' -> 'ك', 'ل' -> 't' (wait, 'الطالب' vs 'الكاتب':
    # 'ا ل ط ا ل ب' vs 'ا ل ك ا ت ب'
    # 'ط' -> 'ك', 'ل' -> 'ا', 'ا' -> 'ت'. Actually it's 3 substitutions: 'ط'->'ك', 'ا'->'ا' (same), 'ل'->'ت', 'ب'->'ب' (same).
    # Wait, 'الـ طـ ا لـ ب' (6 chars) vs 'الـ كـ ا تـ ب' (6 chars).
    # Positions:
    # 1: ا, 2: ل, 3: ط -> ك, 4: ا -> ا, 5: ل -> ت, 6: ب -> ب.
    # So 2 substitutions: 'ط'->'ك' and 'ل'->'t'. Total diff is 2 characters.
    # Total characters in ref clean: 16.
    # 2/16 = 0.125
    assert calculate_cer(ref, hyp) == pytest.approx(2.0 / 16.0)


def test_cer_wer_empty_inputs():
    # Both empty
    assert calculate_cer("", "") == 0.0
    assert calculate_wer("", "") == 0.0

    # Reference empty
    assert calculate_cer("", "كتب") == 1.0
    assert calculate_wer("", "كتب") == 1.0

    # Hypothesis empty
    assert calculate_cer("كتب", "") == 1.0
    assert calculate_wer("كتب", "") == 1.0


def test_cer_wer_normalization_effect():
    ref = "كَتَبَ أحمدُ الدرسَ."
    hyp = "كتب احمد الدرس"
    # Without normalization:
    # ref becomes "كَتَبَ أحمدُ الدرسَ." (with diacritics, alef hamza, period)
    # hyp is "كتب احمد الدرس"
    # They will be very different.
    # With normalization:
    # ref_clean = "كتب احمد الدرس"
    # hyp_clean = "كتب احمد الدرس"
    # They should match perfectly!
    assert calculate_cer(ref, hyp, normalize=True) == 0.0
    assert calculate_wer(ref, hyp, normalize=True) == 0.0


# --- Tests for Schema Validation ---


def test_ocr_schema_validation_success():
    valid_ocr_payload = {
        "document_id": "doc_001",
        "engine": "tesseract",
        "raw_text": "كتب الطالب الدرس",
        "tokens": [
            {"text": "كتب", "bbox": [10, 20, 50, 40], "confidence": 0.95},
            {"text": "الطالب", "bbox": [60, 20, 120, 40], "confidence": 0.88},
            {"text": "الدرس", "bbox": [130, 20, 180, 40], "confidence": 0.99},
        ],
        "processing_time_ms": 120,
    }
    # Should not raise any exceptions
    validate_ocr_output(valid_ocr_payload)


def test_ocr_schema_validation_failure():
    invalid_ocr_payload = {
        "document_id": "doc_001",
        "engine": "invalid_engine_name",  # Must be tesseract, easyocr, or paddleocr
        "raw_text": "كتب الطالب الدرس",
        "tokens": [
            {
                "text": "كتب",
                "bbox": [10, 20, 50],
                "confidence": 0.95,
            }  # Bbox has only 3 items, needs 4
        ],
        "processing_time_ms": -10,  # Must be >= 0
    }
    with pytest.raises(SchemaValidationError) as exc_info:
        validate_ocr_output(invalid_ocr_payload)
    assert "engine" in str(exc_info.value)
    assert "bbox" in str(exc_info.value)


def test_detection_schema_validation_success():
    valid_detection_payload = {
        "document_id": "doc_001",
        "model": "yolov8",
        "objects": [
            {
                "label": "Stamp",
                "bbox": [100.0, 150.0, 200.0, 250.0],
                "confidence": 0.91,
                "context_tag": "administrative",
            },
            {
                "label": "Logo",
                "bbox": [50.0, 50.0, 120.0, 120.0],
                "confidence": 0.85,
                "context_tag": "financial",
            },
        ],
    }
    validate_detection_output(valid_detection_payload)


def test_fusion_input_schema_validation_success():
    valid_fusion_payload = {
        "document_id": "doc_001",
        "ocr": {
            "document_id": "doc_001",
            "engine": "tesseract",
            "raw_text": "كتب الطالب الدرس",
            "tokens": [
                {"text": "كتب", "bbox": [10, 20, 50, 40], "confidence": 0.95},
                {"text": "الطالب", "bbox": [60, 20, 120, 40], "confidence": 0.88},
                {"text": "الدرس", "bbox": [130, 20, 180, 40], "confidence": 0.99},
            ],
            "processing_time_ms": 120,
        },
        "detected_objects": [
            {
                "label": "Stamp",
                "bbox": [100.0, 150.0, 200.0, 250.0],
                "confidence": 0.91,
                "context_tag": "administrative",
            }
        ],
        "correction_candidates": [
            {
                "original_word": "الطالب",
                "position": [4, 10],
                "candidates": [
                    {"text": "الكاتب", "lm_score": 0.12},
                    {"text": "الطالب", "lm_score": 0.88},
                ],
            }
        ],
    }
    validate_fusion_input(valid_fusion_payload)
