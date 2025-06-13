#!/bin/bash
PlYI2M_BUILD_VERSION=$(awk -F"-" '{ print $2 }' <<< $(ls dist/ | head -1))
PYI2M_VERSIONS_STRING=$(pip index versions iso2mesh | grep versions:)
PYI2M_VERSIONS_STRING=${PYI2M_VERSIONS_STRING#*:}
UPLOAD_TO_PYPI=1
while IFS=', ' read -ra PYI2M_VERSIONS_ARRAY; do
  for VERSION in "${PYI2M_VERSIONS_ARRAY[@]}"; do
    if [ "$PYI2M_BUILD_VERSION" = "$VERSION" ]; then
      UPLOAD_TO_PYPI=0
    fi
  done;
done <<< "$PYI2M_VERSIONS_STRING"
if [ "$UPLOAD_TO_PYPI" = 1 ]; then
  echo "Wheel version wasn't found on PyPi.";
else
  echo "Wheel was found on PyPi.";
fi
echo "perform_pypi_upload=$UPLOAD_TO_PYPI" >> $GITHUB_OUTPUT
