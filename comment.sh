#!/bin/bash
cat <<EOF > report.md
# $1

This is a comment created by cml.

![Bug](./img.png)

$(date)

EOF
