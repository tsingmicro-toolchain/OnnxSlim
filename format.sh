#!/bin/sh
set -ex
ufmt format -- .
autoflake --in-place --recursive .
