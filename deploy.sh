#!bash

python3 setup.py bdist_wheel

if [ -z "${PYPI_TOKEN}" ]; then
    twine upload dist/*
else
    twine upload dist/* -u __token__ -p $PYPI_TOKEN
fi

rm -rf build
rm -rf dist
