import editdistance


# Don't forget to support cases when target_text == ''


def calc_cer(target_text: str, predicted_text: str) -> float:
    dist = editdistance.eval(target_text, predicted_text)
    return 1. if len(target_text) == 0 else dist / len(target_text)


def calc_wer(target_text: str, predicted_text: str) -> float:
    target_text = target_text.split()
    predicted_text = predicted_text.split()
    dist = editdistance.eval(target_text, predicted_text)
    return 1. if len(target_text) == 0 else dist / len(target_text)
