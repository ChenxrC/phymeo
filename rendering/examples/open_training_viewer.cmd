@echo off
setlocal
pushd "%~dp0..\.." || exit /b 1
set "PYTHONPATH=%CD%\src;%CD%\rendering\src;%CD%\rendering\examples;%CD%\rendering\tests;%CD%\artifacts\physics-vulkan-interop\p0-20261124\cp313;%APPDATA%\Python\Python313\site-packages;%PYTHONPATH%"
if not defined VK_LAYER_PATH set "VK_LAYER_PATH=%CD%\artifacts\rendering\g3-table\toolchain\Validation_Layers_Windows_64"
if not defined PHYVIEW_DEVICE set "PHYVIEW_DEVICE=1"
if not defined PHYVIEW_LIBRARY set "PHYVIEW_LIBRARY=%CD%\artifacts\physics-vulkan-interop\p1-raster-20261124\build\phymeo_native.dll"
py -3.13 -B rendering\examples\view_training.py --library "%PHYVIEW_LIBRARY%" --device-index "%PHYVIEW_DEVICE%" %*
set "viewer_exit=%errorlevel%"
popd
if not "%viewer_exit%"=="0" pause
exit /b %viewer_exit%
