import json
import sys

from src.inference import predict_text


if __name__ == '__main__':
    if len(sys.argv) < 2:
        raise SystemExit('Usage: python predict.py "Your message text here"')
    text = ' '.join(sys.argv[1:])
    result = predict_text(text)
    print(json.dumps(result, indent=2))
