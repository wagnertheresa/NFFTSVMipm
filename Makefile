make:
	rm -rf build
	python setup.py build
	python setup.py install

check:
	python test/test.py
