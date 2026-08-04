#!/usr/bin/env bash

bash ./build.sh

docker save psmareg_convexadam | gzip -c > psmareg_convexadam.tar.gz
