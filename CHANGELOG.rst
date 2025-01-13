=========
Changelog
=========

Unreleased Changes
==================
-

Version 0.3.2
=============
- Fixed an issue when downloading activate data, because the variable
 keyword in the API has changed (`soil_moisture_saturation` -> `surface_soil_moisture`)
- The default version for download is now v202312

Version 0.3.1
=============
- Fixing CDS API access (new token and API url)

Version 0.3.0
=============
- Added CLI module
- Added a program for regular dataset updates (pulling new
  images and time series creation)
- Version specific metadata was dropped
- Dockerfile added
- Tutorials submodule was updated

Version 0.2.0
=============
- Support C3S v202212
- Add tutorials submodule to docs
- Allow reading and reshuffling spatial subsets (bbox)
- Add support for new versions v201912, v202012
- Add module to generate metadata for time series files
- Add Reader for image stacks
- CI runs in Github Actions now

Version 0.1.2
=============
- Update pyscaffold structure
- Add automatic pypi deploy
- Add option to remove time zone info from time series

Version 0.1.1
=============
- Contains extended time series reader

Version 0.1
===========
- Add smecv grid support
- Add reader from images and image stacks
- Add reshuffle module
- Add first version of metadata module
- Add basic documentation
- Add unit tests for all modules
