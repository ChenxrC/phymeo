@echo off
setlocal
pushd "%~dp0..\.." || exit /b 1
set "PYTHONPATH=%CD%\src;%CD%\rendering\src;%CD%\rendering\examples;%APPDATA%\Python\Python313\site-packages;%PYTHONPATH%"
if not defined PHYVIEW_DEVICE set "PHYVIEW_DEVICE=1"
if not defined PHYVIEW_LIBRARY set "PHYVIEW_LIBRARY=%CD%\artifacts\physics-vulkan-interop\p1-raster-20261124\build\phymeo_native.dll"
py -3.13 -B rendering\examples\view_background_simulation.py --library "%PHYVIEW_LIBRARY%" --device-index "%PHYVIEW_DEVICE%" %*
set "viewer_exit=%errorlevel%"
popd
exit /b %viewer_exit%
