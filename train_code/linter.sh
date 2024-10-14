rm -rf `find -type d -name .ipynb_checkpoints`
isort models/ov_blip_detr.py
flake8 models/ov_blip_detr.py