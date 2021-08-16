format:
	black pyrayt tinygfx

tests:
	python -m unittest discover -v -s ./test -p test_*.py

itests:
	python -m unittest discover -v -s ./test -p int_test_*.py