.PHONY: install plan

install:
	python -m pip install -e '.[all]'

plan:
	python reproduce.py plan
