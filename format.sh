#!/bin/bash

# Formats the code.

black *.py
black tests/*.py
black examples/*.py
black swiftemulator/*.py
black swiftemulator/*/*.py
