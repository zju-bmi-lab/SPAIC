# --- base
.PHONY:init
init:
	pip install -r requirements.txt

# ----pypi----

.PHONY:clear_package
clear_package:
	-sudo rm -rf ./dist ./build ./SPAIC.egg-info

.PHONY:package # package to wheel
package:clear_package
	python setup.py sdist bdist_wheel

.PHONY:pypitest\:upload
pypitest\:upload:package
	- python3 -m twine upload --repository pypitest dist/*
	make clear_package

.PHONY:pypitest\:upload
pypi\:upload:package
	- python3 -m twine upload --repository pypi dist/*
	make clear_package

# --- pypi end ---