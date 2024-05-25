username=$1
python scripts/detection.py --benchmark gsm8k --type rephrase --username $username &
python scripts/detection.py --benchmark gsm8k --type synthetic --username $username &
python scripts/detection.py --benchmark gsm8k --type reference --username $username &
python scripts/detection.py --benchmark arc --type rephrase --username $username &
python scripts/detection.py --benchmark arc --type synthetic --username $username &
python scripts/detection.py --benchmark arc --type reference --username $username &
python scripts/detection.py --benchmark mmlu --type rephrase --username $username &
python scripts/detection.py --benchmark mmlu --type synthetic --username $username &
python scripts/detection.py --benchmark hellaswag --type rephrase --username $username &
python scripts/detection.py --benchmark hellaswag --type synthetic --username $username &
python scripts/detection.py --benchmark hellaswag --type reference --username $username &